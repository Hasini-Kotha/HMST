# Underwater Image Quality Assessment (UIQA) Research

This repository implements deep learning architectures for **Underwater Image Quality Assessment (UIQA)** based on *Solution III: Multi-Scale Transformer-Based Patch Quality with Hierarchical Perceptual Pooling* (`Underwater_ideas.pdf`, §4.1–§4.9).

The primary model implemented is **HMST (Hierarchical Multi-Scale Transformer - Architecture 1)** located in the [`hmst/`](file:///s:/SNU/5th%20sem/dl%20research/hmst) directory.

---

## 📑 Quick Navigation
* **Complete Step-by-Step Architecture Guide (Simple English)**: [ARCHITECTURE_EXPLAINED.md](file:///c:/Users/Hasini/HMST/ARCHITECTURE_EXPLAINED.md)
* **Full HMST Architecture & Run Guide**: [hmst/README.md](file:///c:/Users/Hasini/HMST/hmst/README.md)
* **Architecture Specifications & Paper Mapping**: [solution3_architecture_analysis.md](file:///c:/Users/Hasini/HMST/solution3_architecture_analysis.md)
* **Code Implementation**:
  - Model Architecture: [hmst/models/architecture1.py](file:///s:/SNU/5th%20sem/dl%20research/hmst/models/architecture1.py)
  - Core Modules: [hmst/models/modules.py](file:///s:/SNU/5th%20sem/dl%20research/hmst/models/modules.py)
  - Multi-Objective Loss: [hmst/models/losses.py](file:///s:/SNU/5th%20sem/dl%20research/hmst/models/losses.py)
  - Training Engine: [hmst/training/trainer.py](file:///s:/SNU/5th%20sem/dl%20research/hmst/training/trainer.py)
  - Training Script: [hmst/scripts/train.py](file:///s:/SNU/5th%20sem/dl%20research/hmst/scripts/train.py)

---

## 🏗️ Architecture Flow (Input → Output)

```
                              Input Image: X ∈ R^[B, 3, 224, 224]
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
            Scale s1 (p = 32)       Scale s2 (p = 64)      Scale s3 (p = 128)
             N1 = 49 patches         N2 = 9 patches          N3 = 1 patch
                      │                       │                       │
                      └───────────────────────┼───────────────────────┘
                                              │
                                              ▼
                             Shared ResNet-50 Backbone (§2.3)
                           + Linear Projection → [B, N_s, 512]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          §4.3 Local Quality Head                          §4.4 Transformer Encoder
         q_i^(s) = σ(W_q·f_i) ∈ [0,1]                      Multi-Head Self-Attention
                      │                                    h_i^(s) = MSA(f_i^(s))
                      │                                               │
                      │                                               ▼
                      │                                    §4.5 Semantic Importance
                      │                                    α_i^(s) = softmax(MLP(h_i))
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                           §4.6 Hierarchical Perceptual Pooling
                           Level 1: Patches → 4 Spatial Quadrants (R_k)
                           Level 2: Quadrants → Scale Quality Q_s
                                              │
                                              ▼
                           §4.8 Scale-Adaptive Quality Fusion
                           Scale weights λ_s = softmax(MLP([G_1, G_2, G_3]))
                           Final Predicted MOS: Q̂ = Σ λ_s · Q_s ∈ [0, 1]
                                              │
                                              ▼
                           §4.9 Multi-Objective Loss Function
                           L_total = λ_MOS · L_MOS + λ_rank · L_rank + λ_cons · L_cons
```

---

## 📦 Required Libraries

| Library | Role |
|---------|------|
| **`torch`** ($\ge 2.0.0$) | Deep learning, tensor autograd, GPU acceleration, mixed precision |
| **`torchvision`** ($\ge 0.15.0$) | Pretrained ResNet-50 weights, image augmentations |
| **`timm`** ($\ge 0.9.0$) | Vision architecture model utilities |
| **`scipy`** ($\ge 1.10.0$) | Spearman Rank ($\text{SROCC}$) & Kendall Tau ($\text{KRCC}$) correlation metrics |
| **`openpyxl`** ($\ge 3.1.0$) | Reading Excel score sheets (`IQA Value.xlsx` for UWIQA) |
| **`numpy`** ($\ge 1.24.0$) | Numerical operations and Pearson Linear ($\text{PLCC}$) correlation |
| **`Pillow`** ($\ge 9.5.0$) | Image loading and preprocessing |
| **`PyYAML`** ($\ge 6.0$) | Configuration file loading (`test.yaml` / `full.yaml`) |

---

## 🚀 Commands to Run

### Installation
```powershell
cd "s:\SNU\5th sem\dl research\hmst"
pip install -r requirements.txt
```

### 1. Test Mode (Quick CPU/GPU verification on subset)
```powershell
python scripts/train.py --mode test
```

### 2. Full Mode (Complete training on full dataset)
```powershell
python scripts/train.py --mode full
```

### 3. Resume Training
```powershell
python scripts/train.py --mode full --resume checkpoints/full/latest.pt
```

### 4. Evaluate Checkpoint
```powershell
python scripts/evaluate.py --checkpoint checkpoints/full/best.pt --mode full
```

---

