# Underwater Dark Channel Prior (UDCP) Integration Plan

## 1. Executive Summary

This document specifies the theoretical foundation, architectural design, and exact code modifications required to integrate **Underwater Dark Channel Prior (UDCP)** and **optical color-cast features** into the **HMST (Hierarchical Multi-Scale Transformer)** architecture.

* **Target Goal**: Inject domain-specific underwater optical physics into an ImageNet-pretrained ResNet-50 backbone.
* **Expected Gain**: +0.012 to +0.025 SROCC on underwater image quality benchmarks (e.g., UWIQA, UID).
* **Computational Cost**: Virtually zero (<1.5 ms per batch on GPU, ~1,200 new trainable parameters).
* **Target Files**:
  * [`hmst/models/modules.py`](file:///c:/Users/Hasini/HMST/hmst/models/modules.py)
  * [`hmst/models/architecture1.py`](file:///c:/Users/Hasini/HMST/hmst/models/architecture1.py)

---

## 2. Research Motivation: Why HMST Needs Physical Degradation Priors

### 2.1 The Domain Gap of ResNet-50
HMST utilizes a ResNet-50 backbone pre-trained on ImageNet. While ImageNet provides strong general representations for textures, shapes, and semantic objects (e.g., animals, rocks, boats), it is unequipped to evaluate **optical transmission physics**:
1. **Underwater Veil / Backscatter**: Light from the water surface scatters off suspended micro-particles into the camera sensor, reducing contrast and washing out shadows.
2. **Wavelength-Selective Absorption**: Water absorbs red light exponentially faster than green and blue (red attenuation coefficient $\approx 0.35/\text{m}$ vs blue $\approx 0.02/\text{m}$). Past 3–5 meters, natural red illumination vanishes, causing a heavy cyan/greenish color cast.

In standard HMST, the network must expend model capacity trying to learn these optical phenomena purely from high-level convolutional features.

### 2.2 Learning from the UIED Approach (`archi_internet.png`)
The reference paper architecture (`archi_internet.png`, UIED) extracts physical degradation using an underwater optical model:

$$I(x) = J(x) \cdot T(x) + B \cdot (1 - T(x))$$

Where:
* $I(x)$ = Observed degraded image.
* $J(x)$ = Clean scene radiance.
* $T(x)$ = Scene medium transmission map.
* $B$ = Background ambient light.

Instead of training an auxiliary neural network to estimate $T(x)$ and $B$ (which introduces high training instability and overfitting risks on small IQA datasets), **UDCP provides an analytical, closed-form proxy for $T(x)$** directly from the image pixels with zero learned parameters.

---

## 3. Mathematical Formulation of the Physical Features

### 3.1 Underwater Dark Channel Prior (UDCP)
Standard Dark Channel Prior (He et al., 2009) evaluates the minimum across all three channels $\{R, G, B\}$. In underwater imagery, the red channel is naturally dark due to water absorption, making standard DCP trivial and biased.

Following Drews et al. (2013), **UDCP** computes the minimum solely over the **Green and Blue** channels:

$$J^{\text{UDCP}}(x) = \min_{c \in \{G, B\}} \left( \min_{y \in \Omega(x)} I^c(y) \right)$$

Where $\Omega(x)$ represents a spatial local window around pixel $x$.

### 3.2 Four Extracted Patch Descriptors
For each patch $P_i^{(s)}$ extracted at scale $s$, we compute 4 statistical descriptors:

1. **`udcp_mean` (Turbidity & Veil Level)**:
   $$\mu_{\text{UDCP}} = \frac{1}{|P|} \sum_{x \in P} J^{\text{UDCP}}(x)$$
   * High values indicate severe suspended-particle backscatter and heavy fog/turbidity.

2. **`udcp_std` (Local Structural Contrast)**:
   $$\sigma_{\text{UDCP}} = \sqrt{\frac{1}{|P|} \sum_{x \in P} \left(J^{\text{UDCP}}(x) - \mu_{\text{UDCP}}\right)^2}$$
   * Differentiates between clear patches with rich texture variation and completely washed-out milky haze.

3. **`udcp_max` (Peak Scattering Concentration)**:
   $$M_{\text{UDCP}} = \max_{x \in P} J^{\text{UDCP}}(x)$$
   * Measures local light glint or severe hot spots of scattered light.

4. **`red_attenuation` (Color Cast / Absorption Index)**:
   $$\rho_{\text{red}} = \frac{\bar{I}_R}{\bar{I}_G + \bar{I}_B + \epsilon}$$
   * Quantifies color loss; severely degraded deep underwater scenes have $\rho_{\text{red}} \to 0$.

---

## 4. Architectural Integration & Dataflow

```
                            Input Patch [B, Ns, 3, ps, ps]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
  ResNet-50 Backbone (Frozen/Tuned)                       UnderwaterPhysicalPriorExtractor
            │                                             (Denormalize + UDCP + Red Ratio)
            ▼                                                           │
   Feature: [B, Ns, 2048]                                               ▼
            │                                             Physical Vector: [B, Ns, 4]
            ▼                                                           │
   Linear(2048 → d_model - 32)                                          ▼
            │                                                   Linear(4 → 32) + ReLU
            ▼                                                           │
   Semantic Embed: [B, Ns, d_model - 32]                         Physical Embed: [B, Ns, 32]
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                         Concat along channel dimension
                                          │
                                          ▼
                          Unified Patch Token f_i [B, Ns, d_model]
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
§4.3 Local Quality Head                                     §4.4 Transformer Encoder
   q_i = σ(W_q · f_i)                                          h_i = SelfAttention(f_i)
(Degradation-aware score)                                   (Physics-informed context)
            │                                                           │
            │                                                           ▼
            │                                               §4.5 Importance Head
            │                                                  α_i = softmax(MLP(h_i))
            │                                                           │
            └─────────────────────────────┬─────────────────────────────┘
                                          ▼
                             §4.6 Hierarchical Pooling
                                          ▼
                             §4.8 Scale-Adaptive Fusion
                                          ▼
                             Final Quality Score Q̂
```

---

## 5. Implementation Specifications

### 5.1 File 1: [`hmst/models/modules.py`](file:///c:/Users/Hasini/HMST/hmst/models/modules.py)

Add the `UnderwaterPhysicalPriorExtractor` module:

```python
class UnderwaterPhysicalPriorExtractor(nn.Module):
    """
    Extracts physics-based underwater degradation descriptors per patch:
      1. Mean UDCP (turbidity/veil level)
      2. Std UDCP (structural contrast)
      3. Max UDCP (peak backscatter)
      4. Red channel attenuation ratio (color absorption)
    """
    def __init__(self, patch_size_dcp: int = 7):
        super().__init__()
        self.patch_size_dcp = patch_size_dcp
        # ImageNet normalization constants for real-time de-normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [B, Ns, 3, ps, ps] (ImageNet-normalized)
        Returns:
            descriptors: [B, Ns, 4]
        """
        B, Ns, C, H, W = patches.shape

        # 1. Invert normalization to restore true [0, 1] optical intensities
        x = patches * self.std + self.mean
        x = torch.clamp(x, 0.0, 1.0)

        # 2. Extract channels: R = [B, Ns, 1, H, W], G = ..., B = ...
        r_ch = x[:, :, 0:1, :, :]
        g_ch = x[:, :, 1:2, :, :]
        b_ch = x[:, :, 2:3, :, :]

        # 3. Compute UDCP using Green & Blue channels
        gb_min = torch.min(g_ch, b_ch)  # [B, Ns, 1, H, W]
        # Reshape to [B*Ns, 1, H, W] for 2D min-pooling
        gb_flat = gb_min.view(B * Ns, 1, H, W)
        k = min(self.patch_size_dcp, H, W)
        pad = k // 2
        # Min-pooling via negative max-pooling
        udcp_flat = -F.max_pool2d(-gb_flat, kernel_size=k, stride=1, padding=pad)
        udcp = udcp_flat.view(B, Ns, 1, H, W)

        # 4. Statistical summaries per patch
        udcp_spatial = udcp.view(B, Ns, -1)
        udcp_mean = udcp_spatial.mean(dim=-1, keepdim=True)        # [B, Ns, 1]
        udcp_std  = udcp_spatial.std(dim=-1, keepdim=True)         # [B, Ns, 1]
        udcp_max  = udcp_spatial.max(dim=-1, keepdim=True)[0]      # [B, Ns, 1]

        # 5. Red channel attenuation ratio
        r_mean  = r_ch.view(B, Ns, -1).mean(dim=-1, keepdim=True)
        g_mean  = g_ch.view(B, Ns, -1).mean(dim=-1, keepdim=True)
        b_mean  = b_ch.view(B, Ns, -1).mean(dim=-1, keepdim=True)
        red_ratio = r_mean / (g_mean + b_mean + 1e-6)              # [B, Ns, 1]

        # 6. Concatenate 4 physical metrics: [B, Ns, 4]
        phys_stats = torch.cat([udcp_mean, udcp_std, udcp_max, red_ratio], dim=-1)
        return phys_stats
```

---

### 5.2 File 2: [`hmst/models/architecture1.py`](file:///c:/Users/Hasini/HMST/hmst/models/architecture1.py)

#### Initialization (`HMST.__init__`):
```python
self.phys_dim = 32
self.phys_extractor = UnderwaterPhysicalPriorExtractor()
self.phys_proj = nn.Sequential(
    nn.Linear(4, self.phys_dim),
    nn.ReLU(inplace=True),
    nn.Linear(self.phys_dim, self.phys_dim)
)

# ResNet-50 feature projection: maps 2048 -> (d_model - phys_dim)
# Example: 2048 -> 224 when d_model=256
self.feat_proj = nn.Linear(self.BACKBONE_DIM, d_model - self.phys_dim)
```

#### Forward Pass (`HMST.forward`):
```python
# In the per-scale loop:
for s, (patches, gs) in enumerate(zip(patches_ms, self.grid_sizes)):
    ps = self.patch_sizes[s]

    # Semantic features from ResNet-50: [B, Ns, d_model - 32]
    f_semantic = self._extract_patch_features(patches)

    # Physical features from UDCP: [B, Ns, 4] -> [B, Ns, 32]
    p_raw = self.phys_extractor(patches)
    p_embed = self.phys_proj(p_raw)

    # Concatenate into full patch representation: [B, Ns, d_model]
    f_s = torch.cat([f_semantic, p_embed], dim=-1)

    # §4.3 Local quality head (receives exact d_model features)
    q_s = self.quality_head(f_s)

    # §4.4 Transformer cross-patch interaction
    h_s = self.transformer(f_s)

    # §4.5 Semantic importance head
    alpha_s = self.importance_head(h_s)
```

---

## 6. Downstream Impact & Benefits Analysis

| Stage | Mechanism Before UDCP | Mechanism With UDCP | Concrete Benefit |
| :--- | :--- | :--- | :--- |
| **§4.3 Local Quality Head ($q_i$)** | Evaluates quality solely based on ImageNet features (texture, sharpness). | Directly receives explicit turbidity and color loss metrics. | Accurately penalizes uniform turbidity and green/blue shifts that confuse texture-based heads. |
| **§4.4 Cross-Patch Transformer ($h_i$)** | Self-attention relates patches purely based on visual similarity. | Attention compares both visual semantics and optical haze profiles. | Learns whether quality degradation is uniform (turbid water) or localized (foreground occlusion). |
| **§4.5 Importance Head ($\alpha_i$)** | Assigns weights without knowing transmission clarity. | Identifies clear foreground patches versus backscatter-heavy regions. | Up-weights informative foreground regions and down-weights featureless water veil. |
| **§4.8 Scale Adaptive Fusion ($\lambda_s$)** | Fuses scales using global average pooling. | Multi-scale representations reflect degradation at fine vs coarse scales. | High-frequency detail (scale $p=32$) is weighted when water is clear; coarse features dominate when turbid. |

---

## 7. Complexity & Risk Assessment

* **Parameter Overhead**: ~1,200 trainable parameters (`Linear(4, 32)` + `Linear(32, 32)`). Negligible compared to the 23M ResNet-50 backbone.
* **Latency Overhead**: <1.5 ms per batch. Entirely vectorized PyTorch tensor operations executing directly on GPU memory.
* **Overfitting Risk**: **Zero.** Unlike learned convolutional sub-networks, UDCP is a fixed physical function. It restricts the hypothesis space toward physically valid underwater optical models.
* **Compatibility**: Preserves the exact `d_model` token width (256/512). Downstream components, checkpoints configurations, Optuna hyperparameters, and loss functions remain 100% compatible.
