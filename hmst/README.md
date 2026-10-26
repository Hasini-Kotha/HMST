# HMST — Hierarchical Multi-Scale Transformer for Underwater Image Quality Assessment (UIQA)

Implementation of **Architecture 1** from **Solution III: Multi-Scale Transformer-Based Patch Quality with Hierarchical Perceptual Pooling** (based on `Underwater_ideas.pdf`, §4.1–§4.9).

---

## Table of Contents
0. [📘 Complete Architecture Guide in Simple English](../ARCHITECTURE_EXPLAINED.md)
1. [Architecture Overview & Flow (Input → Output)](#architecture-overview--flow-input--output)
2. [Step-by-Step Mathematical Formulation](#step-by-step-mathematical-formulation)
3. [Libraries & Dependencies](#libraries--dependencies)
4. [Datasets Supported](#datasets-supported)
5. [Project Structure](#project-structure)
6. [How to Run (Test Mode vs Full Mode)](#how-to-run-test-mode-vs-full-mode)
7. [Evaluation & Metrics](#evaluation--metrics)
8. [Git Commands to Push Changes](#git-commands-to-push-changes)

---

## Architecture Overview & Flow (Input → Output)

Underwater images suffer from non-uniform degradation: wavelength-dependent light attenuation (green/blue color cast), forward scattering (blur and contrast loss), and backward scattering (veiling glare/fog). Global pooling or single-scale CNNs miss fine-grained artifacts or fail to account for semantic salience.

HMST solves this through a **three-scale hierarchical pipeline**:

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
                      Shared ResNet-50 or ConvNeXt-Tiny Backbone
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

## Step-by-Step Mathematical Formulation

### 1. §4.2 Multi-Scale Patch Extraction
Input image $X \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ is partitioned on regular grids into patches of three complementary scales:
* **Fine scale ($s_1, p=32$)**: $7 \times 7 = 49$ patches $[B, 49, 3, 32, 32]$. Captures localized edge blur, high-frequency textures, and sensor noise.
* **Medium scale ($s_2, p=64$)**: $3 \times 3 = 9$ patches $[B, 9, 3, 64, 64]$. Captures mid-level object boundaries and structural distortions.
* **Coarse scale ($s_3, p=128$)**: $1 \times 1 = 1$ patch $[B, 1, 3, 128, 128]$ (centered crop). Captures global color cast, transmission haze, and background water illumination.

### 2. §4.3 Shared Feature Extraction & Local Quality Head
All patches across all scales pass through the selected, ImageNet-pretrained backbone, followed by a linear projection. ResNet-50 produces 2048 features; ConvNeXt-Tiny produces 768:
$$f_i^{(s)} = \text{Linear}(\text{Backbone}(x_i^{(s)})) \in \mathbb{R}^{d}$$

The **Local Quality Head** directly estimates an initial scalar quality score for each patch:
$$q_i^{(s)} = \sigma(W_q f_i^{(s)} + b_q) \in [0, 1]$$

### 3. §4.4 Transformer-Based Cross-Patch Interaction
Patches cannot be judged in isolation; human visual perception compares degraded regions against cleaner areas. A shared multi-head Transformer encoder ($L=2$ layers, $H=8$ heads) enables patches within a scale to exchange information:
$$h^{(s)} = \text{TransformerEncoder}(f^{(s)}) \in \mathbb{R}^{B \times N_s \times 512}$$

### 4. §4.5 Semantic Importance Weighting
Degradation in visually salient foreground objects (marine life, divers, reefs) affects perceived quality far more than uniform background water. A 2-layer MLP computes normalized patch importance weights:
$$e_i^{(s)} = W_2 \cdot \text{GELU}(W_1 h_i^{(s)} + b_1) + b_2$$
$$\alpha_i^{(s)} = \frac{\exp(e_i^{(s)})}{\sum_{j=1}^{N_s} \exp(e_j^{(s)})}, \quad \sum_{i=1}^{N_s} \alpha_i^{(s)} = 1$$

### 5. §4.6 Two-Level Hierarchical Perceptual Pooling
Rather than flat averaging, spatial aggregation is performed hierarchically:
* **Level 1 (Patch → Region)**: Patches are assigned to $K=4$ quadrant regions $\{G_1, G_2, G_3, G_4\}$ (top-left, top-right, bottom-left, bottom-right). Region quality is computed via importance-weighted pooling:
  $$Q_{R_k}^{(s)} = \frac{\sum_{i \in G_k} \alpha_i^{(s)} q_i^{(s)}}{\sum_{i \in G_k} \alpha_i^{(s)} + \epsilon}$$
* **Level 2 (Region → Scale Quality)**: Region feature vectors $R_k$ are evaluated by a region MLP to produce region weights $\beta_k$, yielding scale quality $Q_s$:
  $$\beta^{(s)} = \text{softmax}(\text{MLP}([R_1, \dots, R_K]))$$
  $$Q_s = \sum_{k=1}^K \beta_k^{(s)} Q_{R_k}^{(s)}$$

### 6. §4.8 Scale-Adaptive Quality Fusion
A fusion network inspects global scale summary tokens $G_s = \sum_i \alpha_i^{(s)} h_i^{(s)}$ to dynamically assign importance weights $\lambda_s$ to each scale:
$$[\lambda_1, \lambda_2, \lambda_3] = \text{softmax}(\text{MLP}([G_1, G_2, G_3]))$$
$$\hat{Q} = \sum_{s=1}^3 \lambda_s \cdot Q_s \in [0, 1]$$

### 7. §4.9 Multi-Objective Loss Function
Trained end-to-end using three complementary loss terms:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{MOS}} \mathcal{L}_{\text{MOS}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$
* **$\mathcal{L}_{\text{MOS}}$**: Smooth L1 loss between predicted $\hat{Q}$ and ground-truth MOS $y$.
* **$\mathcal{L}_{\text{rank}}$**: Pairwise ranking loss enforcing relative quality ordering between image pairs $(i, j)$:
  $$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max(0, - \text{sign}(y_i - y_j) \cdot (\hat{Q}_i - \hat{Q}_j) + \delta)$$
* **$\mathcal{L}_{\text{cons}}$**: Cross-scale consistency constraint. Uses a spatial correspondence matrix $M_{ij} \in \{0, 1\}$ mapping fine patches to overlapping coarse regions:
  $$\mathcal{L}_{\text{cons}} = \frac{1}{\sum M_{ij}} \sum_{i,j} M_{ij} \left( q_i^{(32)} - q_j^{(128)} \right)^2$$

---

## Libraries & Dependencies

All required packages are specified in `requirements.txt`:

| Library | Version | Role in Project |
|---------|---------|-----------------|
| **`torch`** | $\ge 2.0.0$ | Core deep learning framework, autograd engine, GPU tensor acceleration, AMP (Automatic Mixed Precision). |
| **`torchvision`** | $\ge 0.15.0$ | Pretrained ResNet-50 backbone weights (`IMAGENET1K_V1`), image augmentations and transforms. |
| **`timm`** | $\ge 0.9.0$ | Vision architecture utilities and model building helpers. |
| **`scipy`** | $\ge 1.10.0$ | Benchmark correlation metric computations: Spearman Rank ($\text{SROCC}$), Kendall Tau ($\text{KRCC}$). |
| **`openpyxl`** | $\ge 3.1.0$ | Excel parser to read `IQA Value.xlsx` containing human MOS labels for the UWIQA dataset. |
| **`numpy`** | $\ge 1.24.0$ | Array manipulations, metrics math, linear regression for Pearson Correlation ($\text{PLCC}$). |
| **`Pillow`** | $\ge 9.5.0$ | Image I/O reading RGB underwater image files (`.png`, `.jpg`). |
| **`PyYAML`** | $\ge 6.0$ | Configuration file parser loading `configs/test.yaml` and `configs/full.yaml`. |

---

## Datasets Supported

### 1. UWIQA Dataset
* **Path**: `../UWIQA/UWIQA dataset/UWIQA/` (890 real underwater images).
* **Ground Truth**: `../UWIQA/UWIQA dataset/IQA Value.xlsx` (MOS range: 0.2 – 1.0).
* **Train / Val / Test Split**: seeded 80/20 train-candidate/test split, then 80/20 train/validation split within the train-candidate set (approximately 570/142/178 images).

### 2. SAUD Dataset
* **Path**: `../SAUD/SAUD/Benchmark/Enhanced/` (1000 enhanced underwater images across 10 UIE algorithms).
* **Ground Truth**: `../SAUD/SAUD/sub_bt_scores.mat` (Bradley-Terry preference scores scaled from training-split extrema).
* **Train / Val / Test Split**: seeded 80/20 raw-scene split, then a raw-scene-disjoint 80/20 train/validation split within the training scenes (640/160/200 images when each of 100 scenes has 10 enhancements).

---

## Project Structure

```
hmst/
├── configs/
│   ├── test.yaml          # Quick test & verification configuration (subset, CPU/GPU)
│   └── full.yaml          # Full production training configuration (full dataset, GPU)
├── data/
│   ├── dataset.py         # PyTorch Dataset implementations for UWIQA and SAUD
│   ├── transforms.py      # Resize, Normalize, and Augmentation pipelines
│   └── __init__.py
├── models/
│   ├── architecture1.py   # Main HMST model assembly (Sections 4.2–4.8)
│   ├── modules.py         # PatchExtractor, QualityHead, Transformer, HierarchicalPool
│   ├── losses.py          # L_MOS, L_rank, L_cons and joint HMSTLoss (§4.9)
│   └── __init__.py
├── training/
│   ├── trainer.py         # Full training loop, validation, logging, checkpointing
│   ├── metrics.py         # PLCC, SROCC, KRCC, RMSE evaluation metrics
│   ├── checkpoint.py      # Safe checkpoint save and resume utilities
│   └── __init__.py
├── scripts/
│   ├── train.py           # Main training entry point
│   └── evaluate.py        # Standalone checkpoint evaluation script
├── checkpoints/           # Saved model weights, namespaced by dataset/mode/backbone
├── outputs/               # Training logs, namespaced by dataset/mode/backbone
└── requirements.txt       # Python environment dependencies
```

---

## How to Run (Test Mode vs Full Mode)

### Step 1: Install Dependencies
```powershell
# Navigate to the hmst directory
cd "s:\SNU\5th sem\dl research\hmst"

# Install requirements
pip install -r requirements.txt
```

---

### Step 2: Run in Test Mode (Quick Subset / CPU Safe)
Test mode runs a rapid sanity check on a small subset (60 train / 20 val / 20 test images) across 3 epochs to verify the pipeline end-to-end without taking hours:

```powershell
python scripts/train.py --mode test
```

* **Purpose**: Verifies data loading, loss convergence, backward pass, checkpoint saving, and metric calculation.
* **Hardware**: Runs safely on standard CPU or GPU.
* **Runtime**: ~2 to 5 minutes.
* **Outputs**: Checkpoints and logs are namespaced by dataset and mode; ConvNeXt outputs use `checkpoints/<dataset>/test/convnext/` and `outputs/<dataset>/test/convnext/`.

---

### Step 3: Run in Full Mode (Production Training on Full Dataset)
Full mode trains on the entire dataset for 50 epochs with automatic mixed precision (AMP) enabled:

```powershell
python scripts/train.py --mode full
```

* **Purpose**: Full model convergence for research benchmark evaluation.
* **Hardware**: Recommended on GPU (CUDA).
* **Outputs**: ConvNeXt on UWIQA saves to `checkpoints/uwiqa/full/convnext/` and `outputs/uwiqa/full/convnext/`; SAUD uses the corresponding `saud` paths. Start new runs with `resume: false`.

For a custom full-run YAML, set `mode: full`, `dataset: uwiqa` or `saud`, and `backbone: convnext`; include the matching dataset path block and set `resume: false` for the first run. Relative dataset paths are resolved from the HMST project root; absolute server paths are also accepted.

```powershell
python scripts/train.py --config configs/my_uwiqa_full.yaml
python scripts/train.py --config configs/my_saud_full.yaml
```

---

### Step 4: Resume Interrupted Training
After a checkpoint exists, set `resume: true` in the same dataset-specific YAML and rerun:

```powershell
python scripts/train.py --config configs/my_uwiqa_full.yaml
```

The checkpoint is resolved under that dataset/mode/backbone directory. The script errors if the expected checkpoint is absent or if the checkpoint records another dataset. Use separate YAMLs for UWIQA and SAUD.

---

### Step 5: Standalone Checkpoint Evaluation
To evaluate any trained model checkpoint on the validation and test sets without retraining:

```powershell
python scripts/evaluate.py --checkpoint checkpoints/uwiqa/full/convnext/best.pt --mode full
```

---

## Evaluation & Metrics

The pipeline evaluates predictions against human MOS scores using PLCC, SROCC,
KRCC, and RMSE. Validation metrics retain the configured IQA 4-parameter
logistic mapping for benchmark-style monitoring. Final test metrics report raw
PLCC/RMSE as the primary values; JSON also records
`same_split_fitted_plcc` and `same_split_fitted_rmse` for the 4-parameter fit
performed on that same test split. Those fitted values are benchmark-style
statistics, not out-of-sample calibration or raw deployment performance.

* **PLCC (Pearson Linear Correlation Coefficient)** $\in [-1, 1]$ $\uparrow$: Measures prediction accuracy and linearity.
* **SROCC (Spearman Rank-Order Correlation Coefficient)** $\in [-1, 1]$ $\uparrow$: Measures monotonic ranking consistency.
* **KRCC (Kendall Rank Correlation Coefficient)** $\in [-1, 1]$ $\uparrow$: Measures pairwise rank agreement.
* **RMSE (Root Mean Square Error)** $\in [0, \infty)$ $\downarrow$: Measures absolute prediction deviation.

---

## Git Commands to Push Changes

Follow these commands to stage, commit, and push your changes to your Git repository:

```powershell
# 1. Navigate to the repository root
cd "s:\SNU\5th sem\dl research"

# 2. Check current status of modified files
git status

# 3. Stage the modified code, configs, and documentation
git add hmst/configs/ hmst/models/ hmst/scripts/ hmst/training/ hmst/data/ hmst/README.md

# (Optional: If you want to commit all tracked updates)
git add -u

# 4. Commit your changes with a descriptive message
git commit -m "feat(hmst): implement Architecture 1 with ResNet-50 backbone, fix repeat shape logs, and update documentation"

# 5. Push to GitHub
git push origin main
```
