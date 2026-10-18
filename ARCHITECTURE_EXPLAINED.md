# HMST Architecture Guide: From Raw Image to Quality Score (Step-by-Step)

> **HMST**: **H**ierarchical **M**ulti-**S**cale **T**ransformer for Underwater Image Quality Assessment (UIQA)  
> *Based on Solution III: Multi-Scale Transformer-Based Patch Quality with Hierarchical Perceptual Pooling*

---

## 🌊 The Big Picture: Why is Underwater Image Quality Assessment Hard?

When humans look at an underwater photo, they don't judge it like a standard photo on social media. Underwater environments suffer from three unique physical challenges:
1. **Color Cast (Wavelength Absorption)**: Red light is absorbed within 3–5 meters of water, causing underwater photos to look overwhelmingly blue or green.
2. **Haze and Low Contrast (Forward Scattering)**: Suspended sand, plankton, and particles scatter light forward, washing out dark areas and making objects look foggy.
3. **Blur & Noise (Backward Scattering)**: Particles reflect camera flash or sunlight directly into the lens, causing bright specks (marine snow) and loss of sharp edges.

### Why Standard AI Models Fail:
- **Global Average Pooling** (looking at the whole photo at once) misses small, sharp textures or localized blur.
- **Single-Scale Patch Models** (looking only at small crops) lose track of the big picture (e.g., they can't tell if a green tint is just a small coral or the entire ocean water).
- **Ignoring Human Attention**: If an empty corner of water is blurry, a human viewer doesn't care much. But if a sea turtle or diver in the center is blurry, the human immediately gives the image a low quality score!

---

## 🗺️ High-Level Architecture Flowchart

```
                          [ Input Image: 224 × 224 RGB ]
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
       [ Scale 1: Fine ]          [ Scale 2: Mid ]          [ Scale 3: Coarse ]
       Patch size = 32×32         Patch size = 64×64        Patch size = 128×128
       49 patches (7×7)           9 patches (3×3)           1 patch (1×1)
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                                         ▼
                     [ Shared ResNet-50 Feature Extractor ]
                         Extracts 2048-dim features
                                         │
                                         ▼
                     [ Feature Projection: 2048 → 512 / 256 ]
                       Produces clean patch tokens f_i
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              ▼                                                     ▼
     [ Local Quality Head ]                               [ Transformer Encoder ]
   "How good is this patch on                            "Let patches communicate
     its own?" (0.0 to 1.0)                              with each other across
              │                                            the whole photo"
              │                                                     │
              │                                                     ▼
              │                                         [ Semantic Importance Head ]
              │                                         "Does this patch contain an
              │                                           important subject or just
              │                                           empty background water?" (α_i)
              │                                                     │
              └──────────────────────────┬──────────────────────────┘
                                         │
                                         ▼
                   [ Two-Level Hierarchical Perceptual Pooling ]
                   • Level 1: Group patches into 4 spatial quadrants (Regions)
                   • Level 2: Weight regions to compute scale quality Q_s
                                         │
                                         ▼
                         [ Scale-Adaptive Quality Fusion ]
                    Dynamically decides which scale matters most:
                    e.g., 50% Fine + 30% Mid + 20% Coarse
                                         │
                                         ▼
                       [ Final Predicted MOS: Q̂ ∈ [0, 1] ]
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              │                                                     │
    [ Training Phase: 3-Part Loss ]                         [ Testing Phase ]
    1. L_MOS: Matches human score (MAE)                     Test-Time Augmentation
    2. L_rank: Gets ranking order right                     (4-view TTA: flips & rotations)
    3. L_cons: Consistency across scales                             │
                                                            ▼
                                                    [ Final Evaluation ]
                                                    SROCC, PLCC, KRCC, RMSE
```

---

## 🚶 Step-by-Step Breakdown (In Simple English)

---

### Step 1: Input Preprocessing & Sizing
- **What happens**: The raw underwater image is loaded from disk and resized to a standard square format: **$224 \times 224$ pixels** with 3 color channels (Red, Green, Blue).
- **Tensor Shape**: `[Batch_Size, 3, 224, 224]`
- **Simple Analogy**: Think of this as putting a standard square frame around the photo so our neural network always receives the exact same canvas size.

---

### Step 2: Multi-Scale Patch Extraction (§4.2)
- **What happens**: Instead of analyzing the image at only one zoom level, HMST cuts the image into square puzzle pieces ("patches") at **three complementary zoom levels (scales)**:
  1. **Scale 1 (Fine - $p = 32\times32$)**: Cuts the image into a $7 \times 7$ grid = **49 tiny patches**.
     - *What it looks for*: Fine edge sharpness, pixel noise, fine sand grain, and tiny high-frequency textures.
  2. **Scale 2 (Medium - $p = 64\times64$)**: Cuts the image into a $3 \times 3$ grid = **9 medium patches**.
     - *What it looks for*: Object boundaries, fish outlines, coral shapes, and local contrast.
  3. **Scale 3 (Coarse - $p = 128\times128$)**: Takes a large center window = **1 large patch**.
     - *What it looks for*: Global water color cast (is it murky green or deep blue?), overall lighting, and background fog.
- **Output Shapes**:
  - Scale 1: `[Batch_Size, 49, 3, 32, 32]`
  - Scale 2: `[Batch_Size, 9, 3, 64, 64]`
  - Scale 3: `[Batch_Size, 1, 3, 128, 128]`
- **Simple Analogy**: 
  - Scale 1 is a **microscope** looking at dust and edges.
  - Scale 2 is a **magnifying glass** looking at objects.
  - Scale 3 is your **own eyes** stepping back to look at the whole underwater scene.

---

### Step 3: Shared Backbone Feature Extraction & Projection
- **What happens**:
  1. All patches from all scales are fed through a **shared ResNet-50 backbone** (pre-trained on ImageNet).
  2. For every patch, ResNet-50 extracts a high-level representation vector of length **2048**.
  3. A linear layer projects this 2048-dimensional vector down to a compact token dimension: **$d_{\text{model}} = 512$** (or **$256$** in tuned mode).
- **Output Shape per scale**: `[Batch_Size, Num_Patches, d_model]`
- **Why Shared?**: By using the exact same ResNet-50 weights for all scales, the model doesn't need 3 separate heavy networks. This saves GPU memory, prevents overfitting on small datasets, and ensures that features across different scales share a common visual vocabulary.

---

### Step 4: Local Quality Head (§4.3)
- **What happens**: A small linear layer followed by a Sigmoid function inspects each patch's feature vector $f_i$ and predicts an initial quality score $q_i$ between $0.0$ (terrible quality) and $1.0$ (perfect quality):
  $$q_i^{(s)} = \sigma(W_q \cdot f_i^{(s)} + b_q) \in [0, 1]$$
- **Output Shape**: `[Batch_Size, Num_Patches, 1]`
- **Simple English**: The network asks each patch in isolation: *"Based purely on what you see in your tiny window, how clear or distorted do you look?"*

---

### Step 5: Transformer Cross-Patch Attention (§4.4)
- **What happens**: Patches cannot be accurately evaluated in isolation. A patch of plain green water might look "distorted" on its own, but if the whole photo is an underwater dive, that green tint is just the natural environment!
  - We feed all patch feature vectors of a scale into a **Multi-Head Self-Attention Transformer Encoder** (with 2–3 layers and 4–8 attention heads).
  - Each patch computes queries ($Q$), keys ($K$), and values ($V$) to compare itself to all other patches in the image:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
- **Output Shape**: `[Batch_Size, Num_Patches, d_model]` (contextualized features $h_i$)
- **Simple English**: The patches "talk to each other." Patch #1 says: *"I'm seeing a lot of green fog here, what do you see?"* Patch #25 says: *"I have a bright orange clownfish here with sharp edges!"* Now every patch understands the broader context of the entire photograph.

---

### Step 6: Semantic Importance Head (§4.5)
- **What happens**: A 2-layer Multi-Layer Perceptron (MLP) takes the contextualized features $h_i$ from the Transformer and computes a single importance weight $\alpha_i$ for each patch:
  $$e_i = \text{MLP}(h_i), \quad \alpha_i = \frac{\exp(e_i)}{\sum_{j} \exp(e_j)}$$
  All $\alpha_i$ values across the patches sum up to $1.0$.
- **Output Shape**: `[Batch_Size, Num_Patches, 1]`
- **Simple English**: This acts like the human gaze. When people score a photo, they look at the main subject (fish, coral, diver) and ignore empty water. Patches with the main subject get **high importance ($\alpha \approx 0.15$)**, while plain murky background patches get **low importance ($\alpha \approx 0.005$)**.

---

### Step 7: Two-Level Hierarchical Perceptual Pooling (§4.6)
Instead of simply averaging all patch scores together, HMST aggregates quality in two structured, human-like stages:

#### Level 1: Patch → Quadrant Regions ($R_k$)
- The image is divided into $K = 4$ spatial quadrants:
  - $R_1$: Top-Left
  - $R_2$: Top-Right
  - $R_3$: Bottom-Left
  - $R_4$: Bottom-Right
- For each quadrant $G_k$, we compute its regional quality $Q_{R_k}$ by taking an importance-weighted average of the patches inside it:
  $$Q_{R_k} = \frac{\sum_{i \in G_k} \alpha_i \cdot q_i}{\sum_{i \in G_k} \alpha_i + \epsilon}$$

#### Level 2: Quadrant Regions → Single Scale Quality ($Q_s$)
- A Region MLP inspects the features of each region and decides how important each quadrant is ($\beta_k$, where $\sum \beta_k = 1.0$).
- The final quality for this zoom scale is:
  $$Q_s = \sum_{k=1}^{4} \beta_k \cdot Q_{R_k}$$
- **Simple English**: Humans judge an image region by region. If the bottom-half of the image has a beautiful sharp coral reef, but the top-half is empty water, the model weights the coral region higher to decide the overall quality of that zoom level.

---

### Step 8: Scale-Adaptive Quality Fusion (§4.8)
- **What happens**: We now have three scale quality scores:
  - $Q_{32}$ from the Fine scale (details/noise)
  - $Q_{64}$ from the Mid scale (objects/contrast)
  - $Q_{128}$ from the Coarse scale (color cast/haze)
- Which scale is most important? It depends on the image!
  - If an image has severe green cast, the **Coarse scale** is most informative.
  - If an image has motion blur or sensor noise, the **Fine scale** is most informative.
- A Scale-Fusion MLP computes dynamic weights $\lambda = [\lambda_{32}, \lambda_{64}, \lambda_{128}]$ where $\sum \lambda_s = 1.0$:
  $$\lambda = \text{softmax}(\text{MLP}([G_{32}, G_{64}, G_{128}]))$$
- The final predicted Mean Opinion Score (MOS) $\hat{Q}$ is:
  $$\hat{Q} = \lambda_{32} Q_{32} + \lambda_{64} Q_{64} + \lambda_{128} Q_{128} \quad \in [0, 1]$$
- **Simple English**: The model acts like an expert panel where 3 judges (Fine, Medium, Coarse) give their scores, and a chief judge dynamically decides whose opinion to trust most for this specific picture!

---

### Step 9: Multi-Objective Training Loss Functions (§4.9)
During training, HMST is guided by three simultaneous loss functions:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{MOS}} \mathcal{L}_{\text{MOS}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$

1. **MOS Regression Loss ($\mathcal{L}_{\text{MOS}}$)**:
   - Measures how close the predicted score $\hat{Q}$ is to the actual human score $y$:
     $$\mathcal{L}_{\text{MOS}} = \frac{1}{B} \sum_{b=1}^{B} |\hat{Q}_b - y_b| \quad (\text{Mean Absolute Error})$$
2. **Pairwise Ranking Loss ($\mathcal{L}_{\text{rank}}$)**:
   - Evaluates pairs of images $(A, B)$ in a batch. If Image A has a higher human quality score than Image B ($y_A > y_B$), the model must predict $\hat{Q}_A > \hat{Q}_B$ by at least a safety margin:
     $$\mathcal{L}_{\text{rank}} = \max(0, \text{margin} - (\hat{Q}_A - \hat{Q}_B))$$
   - *Why it matters*: Humans are much better at comparing which of two photos looks better than assigning an exact number like 0.742. Ranking loss directly boosts **SROCC** and **KRCC**.
3. **Cross-Scale Consistency Loss ($\mathcal{L}_{\text{cons}}$)**:
   - A fine-scale patch ($32\times32$) that sits physically inside a coarse-scale region ($128\times128$) shouldn't have a wildly contradictory quality rating.
   - We construct a binary correspondence matrix $M_{ij} \in \{0, 1\}$ and penalize severe disagreements:
     $$\mathcal{L}_{\text{cons}} = \sum_{i,j} M_{ij} \cdot (q_i^{(32)} - q_j^{(128)})^2$$
   - *Why it matters*: Acts as a powerful regularizer that prevents the model from hallucinating noisy patch scores on small datasets.

---

### Step 10: Test-Time Augmentation (4-View TTA)
- **What happens at test time**: When evaluating an unseen test image, instead of running just one forward pass:
  1. View 1: Original Image
  2. View 2: Horizontal Flip
  3. View 3: Vertical Flip
  4. View 4: 180° Rotation (Horizontal + Vertical Flip)
- We average the predicted quality scores from all 4 views:
  $$\hat{Q}_{\text{final}} = \frac{1}{4} (\hat{Q}_1 + \hat{Q}_2 + \hat{Q}_3 + \hat{Q}_4)$$
- **Why it helps**: Underwater images often have asymmetric lighting or directional backscatter. Averaging 4 orientations eliminates directional bias and gives an extra **+2% to +3% boost to SROCC and PLCC** with zero extra training required!

---

## 📐 Summary of Tensor Shapes at Every Layer

| Step | Operation / Layer | Input Tensor Shape | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **1** | Raw Input Image | Camera RGB file | `[B, 3, 224, 224]` |
| **2** | Patch Extractor ($s_1: 32$) | `[B, 3, 224, 224]` | `[B, 49, 3, 32, 32]` |
| | Patch Extractor ($s_2: 64$) | `[B, 3, 224, 224]` | `[B, 9, 3, 64, 64]` |
| | Patch Extractor ($s_3: 128$) | `[B, 3, 224, 224]` | `[B, 1, 3, 128, 128]` |
| **3** | ResNet-50 Backbone | Flattened patches `[B·N, 3, p, p]` | `[B·N, 2048]` |
| | Linear Projection | `[B·N, 2048]` | `[B, N, 512]` |
| **4** | Local Quality Head | `[B, N, 512]` | `[B, N, 1]` |
| **5** | Transformer Cross-Patch | `[B, N, 512]` | `[B, N, 512]` |
| **6** | Semantic Importance Head | `[B, N, 512]` | `[B, N, 1]` ($\sum \alpha = 1$) |
| **7** | Hierarchical Pooling (L1) | `[B, N, 1]` & `[B, N, 512]` | `[B, 4]` (4 Quadrants) |
| | Hierarchical Pooling (L2) | `[B, 4]` quadrant qualities | `[B]` ($Q_s$ per scale) |
| **8** | Scale Adaptive Fusion | 3 scale qualities + features | `[B]` (Final Score $\hat{Q}$) |

---

## 🏆 What Do the Evaluation Numbers Mean?

| Metric | Full Name | Ideal Value | What it Measures in Plain English |
| :--- | :--- | :---: | :--- |
| **SROCC** | Spearman Rank Order Correlation | $+1.0$ | **Relative Ranking**: If Photo A looks better to humans than Photo B, does the AI also rank Photo A higher than Photo B? |
| **PLCC** | Pearson Linear Correlation | $+1.0$ | **Linear Accuracy**: Do the predicted numbers follow a straight line with the actual human scores? |
| **KRCC** | Kendall Rank Correlation | $+1.0$ | **Pairwise Agreement**: Out of all possible pairs of images, what fraction did the model rank in the correct relative order? |
| **RMSE** | Root Mean Square Error | $0.0$ | **Numerical Distance**: On average, how far off is the AI's predicted score from the real human score? |
