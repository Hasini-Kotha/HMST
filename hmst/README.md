# HMST — Hierarchical Multi-Scale Transformer for UIQA

**Architecture 1** from *Solution III: Multi-Scale Transformer-Based Patch Quality with Hierarchical Perceptual Pooling* (`Underwater_ideas.pdf`, §4.2–§4.9).

---

## Architecture

```
Input Image [B, 3, 224, 224]
        │
§4.2    Multi-Scale Patch Extraction
        ├─ Scale s1: p=32  → 49 patches  [B, 49, 3, 32, 32]
        ├─ Scale s2: p=64  → 9 patches   [B, 9,  3, 64, 64]
        └─ Scale s3: p=128 → 1 patch     [B, 1,  3, 128, 128]
        │
        ResNet-50 backbone (shared, pretrained ImageNet)
        Feature projection → [B, Ns, 512]
        │
§4.3    Scale-Specific Quality Head (shared)
        q_i^(s) = σ(W_q · f_i^(s)) ∈ [0,1]  [B, Ns, 1]
        │
§4.4    Transformer Self-Attention (shared, per scale)
        X' = Softmax(QK^T / √d) · V           [B, Ns, 512]
        │
§4.5    Semantic Importance Head (shared)
        α_i = softmax(MLP(h_i))               [B, Ns, 1]
        │
§4.6    Hierarchical Pooling (per scale, two-level)
        LEVEL 1: Q_Rk = Σ_{i∈Gk} α_i·q_i / Σα_i  (region quality)
        LEVEL 2: β_k  = softmax(g(R_k))             (region importance)
                 Q_s  = Σ β_k · Q_Rk                (scale quality)
        │
§4.8    Scale-Adaptive Quality Fusion
        λ_s = softmax(MLP(G_s))
        Q̂   = Σ_s λ_s · Σ_i α_i^(s) · q_i^(s)
        │
§4.9    Cross-Scale Consistency Loss (training only)
        L_cons = Σ_{i,j} M_ij · (q_i^(32) − q_j^(128))²
        L_total = λ₁·L_MOS + λ₂·L_rank + λ₃·L_cons
        │
        Global UIQA Score Q̂ ∈ [0, 1]
```

---

## Datasets

### UWIQA
| Property | Value |
|----------|-------|
| Location | `../UWIQA/UWIQA dataset/UWIQA/` |
| Score file | `../UWIQA/UWIQA dataset/IQA Value.xlsx` |
| Images | 890 PNG files (0001.png – 0890.png) |
| Score range | 0.2 – 1.0 (MOS) |
| Split | 80/20 → 712 train / 178 test |

### SAUD
| Property | Value |
|----------|-------|
| Location | `../SAUD/SAUD/Benchmark/Enhanced/` |
| Score file | `../SAUD/SAUD/sub_bt_scores.mat` |
| Images | 1000 enhanced images (100 raw × 10 UIE methods) |
| Score | BT scores (normalized to [0,1]) |
| Split | 800 train / 200 test (per EDANet paper) |

---

## Project Structure

```
hmst/
├── configs/
│   ├── test.yaml          # Local debug config
│   └── full.yaml          # Server training config
├── data/
│   ├── dataset.py         # UWIQA + SAUD dataset classes
│   ├── transforms.py      # Preprocessing
│   └── __init__.py
├── models/
│   ├── architecture1.py   # HMST main model
│   ├── modules.py         # Sub-modules (§4.2–§4.8)
│   ├── losses.py          # L_MOS, L_rank, L_cons, HMSTLoss
│   └── __init__.py
├── training/
│   ├── trainer.py         # Training loop
│   ├── metrics.py         # PLCC, SROCC, KRCC, RMSE
│   ├── checkpoint.py      # Save/load checkpoints
│   └── __init__.py
├── scripts/
│   ├── train.py           # Main entry point
│   └── evaluate.py        # Evaluation script
├── checkpoints/
│   ├── test/              # Test mode checkpoints
│   └── full/              # Full mode checkpoints
├── outputs/
│   ├── test/              # history.csv, metrics.json
│   └── full/
└── requirements.txt
```

---

## Installation

```bash
cd hmst
pip install -r requirements.txt
```

---

## Local Test Mode

Verify the full pipeline on a small subset (60 train / 20 val / 20 test):

```bash
cd hmst
python scripts/train.py --mode test
```

Prints tensor shapes through the architecture and validates the complete pipeline.

---

## Full Server Training

```bash
cd hmst
python scripts/train.py --mode full
```

---

## Resume Training

```bash
python scripts/train.py --mode full --resume checkpoints/full/latest.pt
```

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/full/best.pt --mode full
```

---

## Checkpoints

| File | Description |
|------|-------------|
| `latest.pt` | Updated every epoch |
| `best.pt` | Best validation SROCC |
| `epoch_NNN.pt` | Per-epoch save (test mode only) |

---

## Configuration

Key hyperparameters (from EDANet paper + §4.9):

| Parameter | Value | Source |
|-----------|-------|--------|
| Backbone | ResNet-50 (ImageNet pretrained) | EDANet baseline |
| Image resize | 224×224 | EDANet baseline |
| Patch sizes | 32, 64, 128 | §4.2 |
| Optimizer | Adam | EDANet baseline |
| Learning rate | 1e-4 | EDANet baseline |
| Batch size | 10 (full) / 4 (test) | EDANet baseline |
| Epochs | 50 (full) / 3 (test) | EDANet baseline |
| λ_MOS | 1.0 | §4.9 |
| λ_rank | 1.0 | §4.9 |
| λ_cons | 0.1 | §4.9 |
| Transformer d_model | 512 | Architecture |
| Transformer heads | 8 | Architecture |
| Transformer layers | 2 | Architecture |

---

## Metrics

| Metric | Target |
|--------|--------|
| PLCC ↑ | Pearson linear correlation |
| SROCC ↑ | Spearman rank-order correlation |
| KRCC ↑ | Kendall rank correlation |
| RMSE ↓ | Root mean square error |
