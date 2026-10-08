# 🏗️ Top 2 Architectures from Solution III (Sections 4.1–4.9)
### For Datasets: SAUD · UWIQA · UUIQ | Baseline Model Selection

> **Context**: Solution III — *Multi-Scale Transformer-Based Patch Quality with Hierarchical Perceptual Pooling* — is the most expressive of the three solutions. It combines multi-scale patch representations, cross-patch Transformer context, hierarchical pooling, and cross-scale consistency. All components below come **strictly** from §4.1–§4.9.

---

## 🗺️ Component Map (§4.1–§4.9)

| Section | Component | Role |
|---------|-----------|------|
| **§4.1** | Motivation | Context & problem statement |
| **§4.2** | Multi-Scale Patch Extraction (p=32,64,128) | Input representation |
| **§4.3** | Scale-Specific Quality Estimation | Per-scale quality head |
| **§4.4** | Transformer-Based Cross-Patch Interaction | Global context |
| **§4.5** | Semantic Importance from Transformer Features | Importance weighting |
| **§4.6** | Hierarchical Pooling (patch→region→image) | Structured aggregation |
| **§4.7** | Cross-Scale Attention (fine-scale queries coarse-scale) | Multi-scale fusion |
| **§4.8** | Scale-Adaptive Quality Fusion | Learned scale weighting |
| **§4.9** | Cross-Scale Consistency Constraint | Regularization loss |

---

## 🥇 Architecture 1 — Full Hierarchical Multi-Scale Transformer (HMST)
### *Sections used: §4.2 + §4.4 + §4.5 + §4.6 + §4.8 + §4.9*

```
Input Image
    │
    ▼
§4.2  Multi-Scale Patch Extraction
      ├─ Scale s1: p=32  → N1 patches  (fine-grained distortion detail)
      ├─ Scale s2: p=64  → N2 patches  (mid-level semantic)
      └─ Scale s3: p=128 → N3 patches  (coarse object context)
    │
    ▼ (per scale, shared CNN backbone — e.g., ResNet-50 / EfficientNet-B3)
§4.3  Scale-Specific Feature Extraction + Local Quality Head
      q_i^(s) = σ(W_q · f_i^(s) + b_q)      ∈ [0,1]
    │
    ▼
§4.4  Transformer Self-Attention (per scale, or shared)
      X' = Softmax(QK^T / √d) · V
      → Each patch representation is now context-aware
    │
    ▼
§4.5  Semantic Importance Head (on Transformer output h_i)
      e_i = MLP(h_i)
      α_i = softmax(e_i)         ← learned patch importance
    │
    ▼
§4.6  Hierarchical Pooling (two-level)
      LEVEL 1: Q_Rk = Σ_{i∈Gk} α_i·q_i / Σα_i    (region quality)
      LEVEL 2: β_k  = softmax(g(R_k))               (region importance)
               Q     = Σ β_k · Q_Rk                 (image quality)
    │
    ▼
§4.8  Scale-Adaptive Fusion
      λ_s = softmax(MLP(G_s))
      Q̂   = Σ_s λ_s · Σ_i α_i^(s) · q_i^(s)
    │
    ▼
§4.9  Cross-Scale Consistency Loss
      L_cons = Σ_{i,j} M_ij · (q_i^(32) − q_j^(128))²
      L_total = λ₁·L_MOS + λ₂·L_rank + λ₃·L_cons
    │
    ▼
  Global UIQA Score Q̂
```

### Why this works on SAUD, UWIQA, UUIQ

| Dataset Characteristic | How this architecture handles it |
|------------------------|----------------------------------|
| **SAUD** — Pairwise ranking labels (binary preference) | L_rank loss natively handles relative preference; Transformer context learns inter-image ordering cues |
| **UWIQA** — Absolute MOS, 890 images, diverse degradation | Multi-scale patches capture both color casts (fine) and low-visibility regions (coarse); hierarchical pooling aggregates correctly |
| **UUIQ** — Unknown distribution, likely similar to UWIQA | Scale-adaptive fusion (§4.8) auto-reweights scales based on the content; cross-scale consistency prevents overfitting |

### Evaluation Metric Predictions

| Metric | Expected behaviour | Rationale |
|--------|-------------------|-----------|
| **PLCC** ↑ | **Highest** among all variants | Hierarchical pooling + Transformer context = smooth, linear-correlated predictions |
| **SROCC** ↑ | **Highest** | L_rank loss directly optimizes ranking agreement |
| **KRCC** ↑ | High | Consistent with SROCC; Kendall benefits from globally consistent importance ordering |
| **RMSE** ↓ | Lowest | L_cons regularization prevents erratic patch-level predictions |

### ✅ Pros
- **Complete modeling**: captures fine distortions (32-px), mid-level semantics (64-px), and coarse scene context (128-px)
- **Cross-patch context** via Transformer → patches are NOT processed blindly in isolation; neighboring patches inform each other
- **Hierarchical pooling** matches human perception → humans judge regions, not independent pixels
- **Scale-adaptive fusion** (§4.8) means the model self-learns which scale matters most per image — critical for UUIQ where degradation varies
- **L_cons** (§4.9) prevents inconsistency between fine- and coarse-scale quality predictions — strong regularizer for small datasets
- **SAUD-compatible**: L_rank loss built into the training objective

### ❌ Cons
- **Computational cost**: Three parallel CNN streams + Transformer + hierarchical pooling → high memory & FLOPs
- **Training data hunger**: Transformers need more data; SAUD has only 1000 images (800 training) — may require pretraining or careful augmentation
- **Hyperparameter sensitivity**: λ₁, λ₂, λ₃ in the combined loss need careful tuning
- **Region grouping** in §4.6 is not self-learned (Gk must be predefined or require additional module)

### 📌 Baseline Recommendation
> **Use this as your PRIMARY baseline.** It is the full expression of Solution III. All ablation studies (removing §4.4, §4.6, §4.9 one at a time) can validate each component's contribution — giving you a clean research story.

---

## 🥈 Architecture 2 — Cross-Scale Transformer with Adaptive Fusion (CSTAF)
### *Sections used: §4.2 + §4.3 + §4.4 + §4.5 + §4.7 + §4.8*

```
Input Image
    │
    ▼
§4.2  Multi-Scale Patch Extraction
      ├─ F^(32)  → fine-grained tokens
      ├─ F^(64)  → mid tokens
      └─ F^(128) → coarse tokens
    │
    ▼
§4.3  Scale-Specific Quality Head (per scale)
      q_i^(s) per patch per scale
    │
    ▼
§4.4  Within-Scale Transformer Self-Attention
      X'^(s) = Attention(X^(s))       ← context within each scale
    │
    ▼
§4.7  Cross-Scale Attention (Between Scales)
      A_{32→128} = Softmax(Q_3 · K_2^T_128 / √d)
      H_{32→128} = A_{32→128} · V_128
      → Fine-scale patch "sees" what large object/region it belongs to
    │
    ▼
§4.5  Semantic Importance from Cross-Scale Transformer output
      α_i^(s) = softmax(MLP(h_i^(s)))
    │
    ▼
§4.8  Scale-Adaptive Quality Fusion
      λ_s = softmax(MLP(G_s))
      Q̂   = Σ_s λ_s · Σ_i α_i^(s) · q_i^(s)
    │
    ▼
  Loss: L = λ₁·L_MOS + λ₂·L_rank   (No L_cons)
    │
    ▼
  Global UIQA Score Q̂
```

### Why this works on SAUD, UWIQA, UUIQ

| Dataset | Advantage |
|---------|-----------|
| **SAUD** | Cross-scale attention captures global context required for pairwise judgment; a distorted fish region (fine scale) is contextualized against surrounding water (coarse scale) |
| **UWIQA** | Cross-scale (§4.7) explicitly lets the 32-px distortion patches "understand" which semantic object they belong to — excellent for color-cast vs. visibility degradation |
| **UUIQ** | Scale-adaptive fusion (§4.8) self-reweights → generalizes well even if UUIQ has different degradation distribution than SAUD/UWIQA |

### Evaluation Metric Predictions

| Metric | Expected behaviour | Rationale |
|--------|-------------------|-----------|
| **PLCC** ↑ | High (slightly below Arch 1) | No hierarchical pooling reduces structural richness |
| **SROCC** ↑ | High | Cross-scale attention learns semantic ordering well |
| **KRCC** ↑ | Moderate-high | Consistent with SROCC performance |
| **RMSE** ↓ | Moderate | No L_cons → slightly higher variance in patch predictions |

### ✅ Pros
- **Cross-scale attention** (§4.7) is the key differentiator: fine-scale patches can "query" the semantic meaning of coarse patches — no other solution has this
- **Lighter than Arch 1**: no two-level hierarchical pooling, no consistency loss — easier to train on small datasets
- **Scale-adaptive fusion** (§4.8) adds flexibility without complexity
- **Better for color-cast degradation** (SAUD biased-color subset): cross-scale allows spatial context that pure patch-independent models miss
- **Simpler to implement** as a baseline — fewer modules to debug

### ❌ Cons
- **No L_cons** (§4.9): predictions between scales can diverge, reducing RMSE performance
- **No hierarchical pooling** (§4.6): treats all patches at a flat level per scale — less aligned with human regional perception
- **Cross-scale attention** (§4.7) is computationally non-trivial (query from one scale, key/value from another) — requires careful dimension alignment

### 📌 Baseline Recommendation
> **Use this as your SECONDARY baseline / strong ablation variant.** It is a leaner, equally novel architecture. You can compare Arch 1 vs Arch 2 to specifically isolate the contribution of **hierarchical pooling + consistency loss** — a publishable ablation finding.

---

## ⚖️ Head-to-Head Comparison

| Criterion | Arch 1 (HMST) | Arch 2 (CSTAF) |
|-----------|---------------|----------------|
| Sections used | §4.2, 4.3, 4.4, 4.5, 4.6, 4.8, 4.9 | §4.2, 4.3, 4.4, 4.5, 4.7, 4.8 |
| Key differentiator | Hierarchical pooling + L_cons | Cross-scale attention (4.7) |
| PLCC (predicted) | ★★★★★ | ★★★★☆ |
| SROCC (predicted) | ★★★★★ | ★★★★☆ |
| RMSE (predicted) | ★★★★★ | ★★★★☆ |
| Training stability | Moderate (3-loss) | High (2-loss) |
| Compute cost | High | Moderate |
| SAUD compatibility | ✅ Native (L_rank) | ✅ Native (L_rank) |
| UWIQA compatibility | ✅ Strong | ✅ Strong |
| UUIQ generalization | ✅ L_cons helps | ✅ Scale-adaptive helps |
| Novelty for research | ⭐⭐⭐ (most complete) | ⭐⭐ (novel cross-scale) |
| Ablation potential | ✅ High | ✅ High |

---

## 🔬 Why NOT to use certain section combinations

| Omitted Component | Why it hurts |
|-------------------|-------------|
| **Omit §4.4 (Transformer)** | Patches become independent → fails to capture spatial context; SAUD pairwise comparison heavily relies on contextual quality differences |
| **Omit §4.8 (Scale-adaptive fusion)** | Uniform scale weighting (λ_s = 1/3) fails when one scale dominates — e.g., UWIQA low-visibility images where coarse scale matters more |
| **Omit §4.2 (Multi-scale)** | Single scale cannot simultaneously capture fine distortions AND semantic structure → confirmed inferior by §10.5 Scale Hypothesis |
| **Use only §4.6 (no §4.4)** | Hierarchical pooling without Transformer context = structurally richer than uniform but lacks cross-patch context — suboptimal |

---

## 📐 Training Recipe Summary

### For SAUD Dataset
- Loss: `L = λ₁·L_MOS_Rank + λ₂·L_rank + λ₃·L_cons` (use ranking loss since labels are binary preference)
- β (preference bias) = 0.5 (per EDANet paper)
- Data split: 800 train / 200 test (per EDANet paper)

### For UWIQA Dataset
- Loss: `L = λ₁·L_MAE + λ₂·L_rank + λ₃·L_cons`
- Data split: 712 train / 178 test (80/20)
- Quality scores: 0.1 – 1.0 range

### For UUIQ Dataset
- Adapt the same recipe as UWIQA (MOS-based), adjusting normalization based on dataset's score range

### Common Settings (matching EDANet paper baseline)
- Backbone: ResNet-50 (pretrained ImageNet)
- Image resize: 224×224
- Optimizer: Adam, lr=1e-4
- Epochs: 50 (DQENet stage)
- Batch size: 10
- Evaluation: 10× random train-test splits, report median PLCC/SROCC/KRCC/RMSE

---

## 🎯 Final Recommendation for Your Research

> **Start with Architecture 1 (HMST) as your primary baseline.** It represents the full Solution III pipeline and gives you the most defensible baseline with the highest expected metrics.
>
> **Use Architecture 2 (CSTAF) as your ablation target** — by removing §4.6 and §4.9 from Arch 1 and adding §4.7 cross-scale attention, you can demonstrate precisely what each component contributes.
>
> Both architectures are novel, well-grounded in the paper's formulations, and fully compatible with all three datasets (SAUD, UWIQA, UUIQ).
