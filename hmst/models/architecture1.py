"""
models/architecture1.py
-----------------------
Architecture 1 — HMST: Hierarchical Multi-Scale Transformer
for Underwater Image Quality Assessment.

Implements §4.2 + §4.4 + §4.5 + §4.6 + §4.8 + §4.9 exactly as
specified in solution3_architecture_analysis.md and §4.2–§4.9 of
Underwater_ideas.pdf.

Architecture flow:
    Input Image [B, 3, 224, 224]
          │
    §4.2  Multi-Scale Patch Extraction
          ├─ s1: p=32  → 49 patches  [B, 49, 3, 32, 32]
          ├─ s2: p=64  → 9 patches   [B, 9,  3, 64, 64]
          └─ s3: p=128 → 1 patch     [B, 1,  3, 128, 128]
          │
    ResNet-50 backbone (shared, pretrained ImageNet)
          → Per-patch feature vectors [B, Ns, 2048]
          → Project to d_model [B, Ns, D]
          │
    §4.3  Local Quality Head
          q_i^(s) = σ(W_q · f_i^(s)) ∈ [0,1]  [B, Ns, 1]
          │
    §4.4  Transformer Self-Attention (per scale, shared weights)
          X' = Softmax(QK^T / √d) · V
          → [B, Ns, D]
          │
    §4.5  Semantic Importance Head
          α_i = softmax(MLP(h_i))  [B, Ns, 1]
          │
    §4.6  Hierarchical Pooling (two-level)
          LEVEL 1: Q_Rk = Σ_{i∈Gk} α_i·q_i / Σα_i
          LEVEL 2: Q_s  = Σ β_k · Q_Rk
          │
    §4.8  Scale-Adaptive Fusion
          λ_s = softmax(MLP(G_s))
          Q̂   = Σ_s λ_s · Q_s, using the hierarchical per-scale quality Q_s
          │
    §4.9  Cross-Scale Consistency Loss (training only)
          L_cons = Σ_{i,j} M_ij · (q_i^(32) - q_j^(128))²
          │
    Global UIQA Score Q̂ ∈ [0, 1]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

import torchvision.models as tv_models

from .convnext_backbone import ConvNeXtTinyBackbone, CONVNEXT_TINY_DIM

from .modules import (
    MultiScalePatchExtractor,
    LocalQualityHead,
    TransformerCrossPatchInteraction,
    SemanticImportanceHead,
    HierarchicalPooling,
    ScaleAdaptiveFusion,
)


class HMST(nn.Module):
    """
    Hierarchical Multi-Scale Transformer (HMST) — Architecture 1.

    Args:
        image_size:   Input image size (224)
        patch_sizes:  [32, 64, 128]
        d_model:      Transformer token dimension (512)
        nhead:        Number of attention heads (8)
        nlayers:      Number of Transformer encoder layers (2)
        n_regions:    Number of hierarchical pooling regions per scale (4)
        dropout:      Dropout rate (0.1)
        print_shapes: Print tensor shapes for debugging (test mode)
        backbone:     Backbone name: 'resnet50' (default) | 'convnext'
                      - 'resnet50'  → pretrained ResNet-50, output dim 2048
                      - 'convnext'  → pretrained ConvNeXt-Tiny, output dim 768
    """

    # Supported backbone names and their output dimensions
    _BACKBONE_DIMS = {
        "resnet50": 2048,
        "convnext": CONVNEXT_TINY_DIM,  # 768
    }

    # Kept for backward compatibility (default: ResNet-50)
    BACKBONE_DIM = 2048

    def __init__(
        self,
        image_size: int = 224,
        patch_sizes: List[int] = [32, 64, 128],
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 2,
        n_regions: int = 4,
        dropout: float = 0.1,
        print_shapes: bool = False,
        backbone: str = "resnet50",
    ):
        super().__init__()

        # ── Validate backbone selection ────────────────────────────────────
        backbone_key = backbone.lower().strip().replace("-", "_")
        if backbone_key == "convnext_tiny":
            backbone_key = "convnext"
        if backbone_key not in self._BACKBONE_DIMS:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Supported values: {sorted(self._BACKBONE_DIMS.keys())}"
            )
        self.backbone_name = backbone_key

        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.print_shapes = print_shapes
        self.n_scales = len(patch_sizes)

        # ── §4.2  Multi-Scale Patch Extractor ─────────────────────────────
        self.patch_extractor = MultiScalePatchExtractor(image_size, patch_sizes)
        self.grid_sizes = self.patch_extractor.grid_sizes  # [7, 3, 1]
        self.num_patches = self.patch_extractor.num_patches  # [49, 9, 1]

        # ── Backbone: selected pretrained backbone (shared across all patches) ─
        if backbone_key == "convnext":
            # ConvNeXt-Tiny pretrained on ImageNet-1K
            # Output: [B*N, 768]  (after built-in AdaptiveAvgPool2d + flatten)
            self.backbone = ConvNeXtTinyBackbone()
            self.BACKBONE_DIM = CONVNEXT_TINY_DIM  # 768
        else:
            # ResNet-50 pretrained on ImageNet-1K — default, unchanged path
            # (§2.3 of Underwater_ideas.pdf lists ResNet as candidate)
            _rn = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove the final FC layer — keep avgpool which gives [2048]
            self.backbone = nn.Sequential(*list(_rn.children())[:-1])
            # backbone output: [B*N, 2048, 1, 1]
            self.BACKBONE_DIM = 2048

        # ── Feature projection: BACKBONE_DIM → d_model ────────────────────
        # One shared projector across all scales.
        # For resnet50:  Linear(2048 → d_model)
        # For convnext:  Linear(768  → d_model)
        self.feat_proj = nn.Linear(self.BACKBONE_DIM, d_model)

        # ── §4.3  Scale-specific quality heads (one per scale) ────────────
        # Shared quality head across scales (per architecture description)
        self.quality_head = LocalQualityHead(feat_dim=d_model)

        # ── §4.4  Transformer cross-patch interaction (shared) ────────────
        self.transformer = TransformerCrossPatchInteraction(
            d_model=d_model,
            nhead=nhead,
            nlayers=nlayers,
            dropout=dropout,
        )

        # ── §4.5  Semantic importance head (shared) ───────────────────────
        self.importance_head = SemanticImportanceHead(
            d_model=d_model,
            hidden=256,
        )

        # ── §4.6  Hierarchical pooling (one per scale) ────────────────────
        # Each scale has its own pooling module with its own region MLP
        self.hierarchical_pools = nn.ModuleList([
            HierarchicalPooling(d_model=d_model, n_regions=n_regions)
            for _ in patch_sizes
        ])

        # ── §4.8  Scale-adaptive fusion ───────────────────────────────────
        self.scale_fusion = ScaleAdaptiveFusion(d_model=d_model, n_scales=self.n_scales)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: run backbone on a batch of patches
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_patch_features(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Run the shared backbone on all patches of one scale.

        Works for both backbone variants:
          - resnet50:  backbone returns [B*N, 2048, 1, 1], needs .view to flatten
          - convnext:  ConvNeXtTinyBackbone.forward() already returns [B*N, 768]

        Args:
            patches: [B, N, C, ps, ps]

        Returns:
            features: [B, N, d_model]
        """
        B, N, C, ps, _ = patches.shape

        # Reshape to [B*N, C, ps, ps] for batch processing
        flat = patches.view(B * N, C, ps, ps)

        # Backbone expects images ≥ 32×32 (both backbones handle 32×32 fine)
        # If ps < 32, upsample (not needed here since min ps=32)
        if ps < 32:
            flat = F.interpolate(flat, size=32, mode="bilinear", align_corners=False)

        # Run backbone
        if self.backbone_name == "convnext":
            # ConvNeXtTinyBackbone.forward() returns [B*N, 768] directly
            feat = self.backbone(flat)               # [B*N, 768]
        else:
            # ResNet-50 backbone → [B*N, 2048, 1, 1] — unchanged original path
            feat = self.backbone(flat)
            feat = feat.view(B * N, self.BACKBONE_DIM)  # [B*N, 2048]

        # Project to d_model
        feat = self.feat_proj(feat)                  # [B*N, d_model]
        feat = feat.view(B, N, self.d_model)         # [B, N, d_model]

        return feat

    # ─────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]  where H=W=224

        Returns:
            dict with keys:
              'Q_hat'      : [B]        final quality score
              'lambda_s'   : [B, S]     scale weights (§4.8)
              'alphas'     : list of [B, Ns, 1] per scale (§4.5)
              'qualities'  : list of [B, Ns, 1] per scale (§4.3)
              'Q_regions'  : list of [B, K] per scale (§4.6, for §4.9 loss)
        """
        B = x.shape[0]
        dbg = self.print_shapes
        if dbg:
            self.print_shapes = False  # Only print shapes once to avoid spamming terminal on every batch
            print(f"\n{'='*60}")
            print(f"[HMST] Backbone:       {self.backbone_name} (dim={self.BACKBONE_DIM})")
            print(f"[HMST] Input:          {list(x.shape)}")

        # ── §4.2  Multi-Scale Patch Extraction ────────────────────────────
        patches_ms = self.patch_extractor(x)
        # patches_ms[s] : [B, Ns, C, ps, ps]

        if dbg:
            for s, (patches, ps) in enumerate(zip(patches_ms, self.patch_sizes)):
                print(f"[S4.2] Scale s{s+1} p={ps:3d}: patches {list(patches.shape)}")

        # ── Per-scale processing ───────────────────────────────────────────
        scale_features  = []  # [B, Ns, D] Transformer outputs h_i
        scale_alphas    = []  # [B, Ns, 1] importance weights α_i
        scale_qualities = []  # [B, Ns, 1] quality scores q_i

        for s, (patches, gs) in enumerate(zip(patches_ms, self.grid_sizes)):
            ps = self.patch_sizes[s]

            # ── Backbone feature extraction ────────────────────────────────
            f_s = self._extract_patch_features(patches)  # [B, Ns, D]

            if dbg:
                print(f"[S4.3] Scale s{s+1} backbone feat: {list(f_s.shape)}")

            # ── §4.3  Local quality head ───────────────────────────────────
            q_s = self.quality_head(f_s)  # [B, Ns, 1]  in [0,1]

            if dbg:
                print(f"[S4.3] Scale s{s+1} quality q_i:   {list(q_s.shape)}")

            # ── §4.4  Transformer cross-patch interaction ──────────────────
            h_s = self.transformer(f_s)   # [B, Ns, D]

            if dbg:
                print(f"[S4.4] Scale s{s+1} Transformer h:  {list(h_s.shape)}")

            # ── §4.5  Semantic importance ──────────────────────────────────
            alpha_s = self.importance_head(h_s)  # [B, Ns, 1]

            if dbg:
                print(f"[S4.5] Scale s{s+1} importance alpha: {list(alpha_s.shape)}")

            scale_features.append(h_s)
            scale_alphas.append(alpha_s)
            scale_qualities.append(q_s)

        # ── §4.6  Hierarchical Pooling (per scale) ─────────────────────────
        Q_regions_per_scale = []
        Q_image_per_scale   = []

        for s in range(self.n_scales):
            gs = self.grid_sizes[s]
            Q_img_s, Q_reg_s = self.hierarchical_pools[s](
                h=scale_features[s],
                alpha=scale_alphas[s],
                q=scale_qualities[s],
                grid_size=gs,
            )
            Q_image_per_scale.append(Q_img_s)   # [B]
            Q_regions_per_scale.append(Q_reg_s)  # [B, K]

            if dbg:
                print(f"[S4.6] Scale s{s+1} Q_image: {list(Q_img_s.shape)}  "
                      f"Q_regions: {list(Q_reg_s.shape)}")

        # ── §4.8  Scale-Adaptive Fusion ────────────────────────────────────
        Q_hat, lambda_s = self.scale_fusion(
            scale_features=scale_features,
            scale_scores=Q_image_per_scale,
        )

        if dbg:
            print(f"[S4.8] Q_hat (fused):      {list(Q_hat.shape)}")
            print(f"[S4.8] lambda_s (scale wts): {list(lambda_s.shape)}")
            print(f"{'='*60}\n")

        return {
            "Q_hat":      Q_hat,           # [B]
            "lambda_s":   lambda_s,        # [B, S]
            "alphas":     scale_alphas,    # list of [B, Ns, 1]
            "qualities":  scale_qualities, # list of [B, Ns, 1]
            "Q_regions":  Q_regions_per_scale,  # list of [B, K]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Scale Consistency Correspondence
# ─────────────────────────────────────────────────────────────────────────────

def build_correspondence_matrix(
    num_patches_fine: int,   # N1 (49 for p=32)
    num_patches_coarse: int, # N3 (1 for p=128)
    grid_fine: int,          # 7
    grid_coarse: int,        # 1
) -> torch.Tensor:
    """
    Build M_ij ∈ {0,1}: M_ij=1 iff fine-scale patch i belongs to
    coarse-scale region j.

    §4.9: L_cons = Σ_{i,j} M_ij · (q_i^(32) − q_j^(128))²

    Returns:
        M: [N_fine, N_coarse]  binary tensor
    """
    M = torch.zeros(num_patches_fine, num_patches_coarse)
    ratio = grid_fine // max(grid_coarse, 1)

    for fi in range(grid_fine):
        for fj in range(grid_fine):
            fine_idx = fi * grid_fine + fj
            ci = fi // ratio
            cj = fj // ratio
            ci = min(ci, grid_coarse - 1)
            cj = min(cj, grid_coarse - 1)
            coarse_idx = ci * grid_coarse + cj
            if coarse_idx < num_patches_coarse:
                M[fine_idx, coarse_idx] = 1.0

    return M
