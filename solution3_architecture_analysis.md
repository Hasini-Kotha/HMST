

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
