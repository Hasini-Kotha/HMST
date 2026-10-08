"""
models/modules.py
-----------------
Sub-modules for Architecture 1 — HMST
(Hierarchical Multi-Scale Transformer)

All components implemented STRICTLY from §4.2–§4.9 of
Underwater_ideas.pdf and solution3_architecture_analysis.md.

§4.2  MultiScalePatchExtractor
§4.4  TransformerCrossPatchInteraction
§4.5  SemanticImportanceHead
§4.6  HierarchicalPooling
§4.8  ScaleAdaptiveFusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# §4.2  Multi-Scale Patch Extractor
# ─────────────────────────────────────────────────────────────────────────────

class MultiScalePatchExtractor(nn.Module):
    """
    §4.2 — Multi-Scale Patch Extraction.

    Splits a 224×224 image into non-overlapping patches at three scales:
        s1: p=32  → (224//32)² = 7×7 = 49 patches
        s2: p=64  → (224//64)² = 3×3 = 9 patches  (224//64=3, 3×64=192, remaining 32 discarded)
        s3: p=128 → (224//128)² = 1×1 = 1 patch  (224//128=1)

    Note: to ensure complete coverage and a reasonable patch count:
        s1: p=32  → 49 patches
        s2: p=64  → 9 patches  (we crop to 192×192 for exact tiling)
        s3: p=128 → 1 patch    (we crop to 128×128 for exact tiling)

    Each patch is extracted and returned as shape [B, Ns, C, p, p].
    Feature extraction by the backbone happens in the architecture module.
    """

    def __init__(self, image_size: int = 224, patch_sizes: List[int] = [32, 64, 128]):
        super().__init__()
        self.image_size = image_size
        self.patch_sizes = patch_sizes

        # Compute how many patches per scale (floor division)
        self.grid_sizes = [image_size // p for p in patch_sizes]
        self.num_patches = [g * g for g in self.grid_sizes]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] where H=W=image_size

        Returns:
            list of tensors, one per scale:
              patches_s[s] → [B, Ns, C, ps, ps]
        """
        B, C, H, W = x.shape
        patches_per_scale = []

        for ps, gs in zip(self.patch_sizes, self.grid_sizes):
            # Crop to the largest multiple of ps that fits
            crop = gs * ps  # e.g. 7*32=224, 3*64=192, 1*128=128
            x_crop = x[:, :, :crop, :crop]  # [B, C, crop, crop]

            # unfold: extract non-overlapping ps×ps patches
            # Result shape: [B, C, gs, ps, gs, ps]
            patches = x_crop.unfold(2, ps, ps).unfold(3, ps, ps)
            # [B, C, gs, gs, ps, ps]
            patches = patches.contiguous().view(B, C, gs * gs, ps, ps)
            # [B, Ns, C, ps, ps]
            patches = patches.permute(0, 2, 1, 3, 4).contiguous()

            patches_per_scale.append(patches)

        return patches_per_scale


# ─────────────────────────────────────────────────────────────────────────────
# §4.3  Scale-Specific Quality Head
# ─────────────────────────────────────────────────────────────────────────────

class LocalQualityHead(nn.Module):
    """
    §4.3 — Scale-specific quality estimation.

    q_i^(s) = σ( W_q · f_i^(s) + b_q )   ∈ [0, 1]

    Maps a feature vector of size `feat_dim` to a scalar quality score.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, D]

        Returns:
            q: [B, N, 1]  — per-patch quality scores in [0,1]
        """
        q = torch.sigmoid(self.head(features))  # [B, N, 1]
        return q


# ─────────────────────────────────────────────────────────────────────────────
# §4.4  Transformer-Based Cross-Patch Interaction
# ─────────────────────────────────────────────────────────────────────────────

class TransformerCrossPatchInteraction(nn.Module):
    """
    §4.4 — Transformer self-attention over patch tokens.

    X' = Softmax(QK^T / √d) · V

    Standard multi-head self-attention as specified.
    Each patch representation becomes context-aware.

    Args:
        d_model:  Token embedding dimension
        nhead:    Number of attention heads
        nlayers:  Number of Transformer encoder layers
        dropout:  Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        nlayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # [B, N, D]
            norm_first=True,   # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]  — patch token sequence

        Returns:
            x': [B, N, D]  — context-aware patch representations
        """
        return self.transformer(x)


# ─────────────────────────────────────────────────────────────────────────────
# §4.5  Semantic Importance Head
# ─────────────────────────────────────────────────────────────────────────────

class SemanticImportanceHead(nn.Module):
    """
    §4.5 — Semantic importance from Transformer features.

    e_i = MLP(h_i)
    α_i = softmax(e_i)   ← learned patch importance

    Args:
        d_model:  Input feature dimension (Transformer output size)
        hidden:   Hidden layer size of the MLP
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, D]  — Transformer output (contextualized features)

        Returns:
            alpha: [B, N, 1]  — normalized patch importance weights
        """
        e = self.mlp(h)           # [B, N, 1]
        alpha = F.softmax(e, dim=1)  # [B, N, 1]
        return alpha


# ─────────────────────────────────────────────────────────────────────────────
# §4.6  Hierarchical Pooling
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPooling(nn.Module):
    """
    §4.6 — Two-level hierarchical pooling.

    LEVEL 1: Region quality
        Q_Rk = Σ_{i∈Gk} α_i·q_i / Σ_{i∈Gk} α_i

    LEVEL 2: Image quality
        β_k  = softmax(g(R_k))   ← region importance
        Q     = Σ β_k · Q_Rk

    Patch grouping (Gk): spatial grid groups.
    For N patches on a grid of size gs×gs, we group into
    n_regions = max(1, gs//2)² spatial blocks.

    Args:
        d_model:   Feature dimension (for region representation g(R_k))
        n_regions: Number of regions to pool into
    """

    def __init__(self, d_model: int, n_regions: int = 4):
        super().__init__()
        # Region importance MLP: g(R_k) → scalar
        self.region_importance = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.n_regions = n_regions

    def _make_groups(self, N: int, grid_size: int) -> List[List[int]]:
        """
        Partition N patches (arranged as grid_size×grid_size) into
        self.n_regions spatial groups.

        Returns list of lists of patch indices.
        """
        n_r = self.n_regions
        # Compute how many grid cells per region block
        # n_r must be a perfect square for even spatial coverage
        r_side = max(1, int(math.isqrt(n_r)))
        # clamp n_r so it divides the grid evenly
        r_side = min(r_side, grid_size)
        cell_size = max(1, grid_size // r_side)

        groups = []
        for ri in range(r_side):
            for rj in range(r_side):
                patch_indices = []
                for pi in range(ri * cell_size, min((ri + 1) * cell_size, grid_size)):
                    for pj in range(rj * cell_size, min((rj + 1) * cell_size, grid_size)):
                        patch_indices.append(pi * grid_size + pj)
                if patch_indices:
                    groups.append(patch_indices)
        return groups

    def forward(
        self,
        h: torch.Tensor,
        alpha: torch.Tensor,
        q: torch.Tensor,
        grid_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:         [B, N, D] — Transformer patch features
            alpha:     [B, N, 1] — patch importance weights
            q:         [B, N, 1] — per-patch quality scores
            grid_size: sqrt(N), the spatial grid dimension

        Returns:
            Q_image:    [B] — final image quality score from this scale
            Q_regions:  [B, K] — region quality scores (for consistency loss)
        """
        B, N, D = h.shape

        groups = self._make_groups(N, grid_size)
        K = len(groups)

        Q_region_list = []
        R_features = []

        for group_indices in groups:
            # Gather patches in this group
            idx = torch.tensor(group_indices, dtype=torch.long, device=h.device)

            h_g = h[:, idx, :]       # [B, len_g, D]
            alpha_g = alpha[:, idx, :]  # [B, len_g, 1]
            q_g = q[:, idx, :]          # [B, len_g, 1]

            # Level 1: weighted average quality for this region
            w_sum = alpha_g.sum(dim=1)       # [B, 1]
            Q_Rk = (alpha_g * q_g).sum(dim=1) / (w_sum + 1e-8)  # [B, 1]
            Q_region_list.append(Q_Rk)

            # Region feature = mean of patch features (for g(R_k))
            R_feat = h_g.mean(dim=1)  # [B, D]
            R_features.append(R_feat)

        # Level 2: region importance
        Q_regions_t = torch.cat(Q_region_list, dim=1)  # [B, K]
        R_features_t = torch.stack(R_features, dim=1)  # [B, K, D]

        # g(R_k) → scalar importance for each region
        e_k = self.region_importance(R_features_t)  # [B, K, 1]
        beta = F.softmax(e_k, dim=1)                # [B, K, 1]

        # Image quality: Σ β_k · Q_Rk
        Q_image = (beta.squeeze(-1) * Q_regions_t).sum(dim=1)  # [B]

        return Q_image, Q_regions_t


# ─────────────────────────────────────────────────────────────────────────────
# §4.8  Scale-Adaptive Fusion
# ─────────────────────────────────────────────────────────────────────────────

class ScaleAdaptiveFusion(nn.Module):
    """
    §4.8 — Scale-adaptive quality fusion.

    G_s = global representation of scale s (mean of all patch features)
    e_s = MLP(G_s)
    λ_s = softmax(e_s)
    Q̂   = Σ_s λ_s · (Σ_i α_i^(s) · q_i^(s))

    Args:
        d_model:   Feature dimension
        n_scales:  Number of scales (3 for p=32,64,128)
    """

    def __init__(self, d_model: int, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales
        self.scale_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        scale_features: List[torch.Tensor],   # [B, Ns, D] per scale
        scale_alphas:   List[torch.Tensor],   # [B, Ns, 1] per scale
        scale_qualities: List[torch.Tensor],  # [B, Ns, 1] per scale
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Q_hat:       [B] — final fused quality score
            lambda_s:    [B, S] — per-scale weights (for logging)
        """
        S = len(scale_features)
        e_s_list = []
        Q_s_list = []

        for s in range(S):
            h_s = scale_features[s]    # [B, Ns, D]
            a_s = scale_alphas[s]      # [B, Ns, 1]
            q_s = scale_qualities[s]   # [B, Ns, 1]

            # G_s = global mean of patch features for this scale
            G_s = h_s.mean(dim=1)      # [B, D]
            e_s = self.scale_mlp(G_s)  # [B, 1]
            e_s_list.append(e_s)

            # Scale-level quality: Σ_i α_i^(s) · q_i^(s)
            Q_s = (a_s * q_s).sum(dim=1)  # [B, 1]
            Q_s_list.append(Q_s)

        e_s_t = torch.cat(e_s_list, dim=1)   # [B, S]
        Q_s_t = torch.cat(Q_s_list, dim=1)   # [B, S]

        lambda_s = F.softmax(e_s_t, dim=1)   # [B, S]
        Q_hat = (lambda_s * Q_s_t).sum(dim=1)  # [B]

        return Q_hat, lambda_s
