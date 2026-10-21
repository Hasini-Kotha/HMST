"""
models/convnext_backbone.py
---------------------------
ConvNeXt-Tiny feature-extraction backbone for HMST (Hierarchical Multi-Scale Transformer).

Wraps the pretrained TorchVision ConvNeXt-Tiny model (IMAGENET1K_V1) so that it produces
a flat 768-d feature vector from image patches, matching the interface expected
by `HMST._extract_patch_features`. It also exposes intermediate hierarchical feature
maps from each stage.

Architecture Overview of ConvNeXt-Tiny:
    Input [B, 3, H, W]  (Default 224 × 224, or patch sizes: 32, 64, 128)
          │
    features[0]  – Stem: Conv2d(3, 96, kernel=4, stride=4) + LayerNorm
    features[1]  – Stage 1: 3 ConvNeXt Blocks            → [B,  96, H/4,  W/4]
    features[2]  – Downsample: LayerNorm + Conv2d(96, 192, kernel=2, stride=2)
    features[3]  – Stage 2: 3 ConvNeXt Blocks            → [B, 192, H/8,  W/8]
    features[4]  – Downsample: LayerNorm + Conv2d(192, 384, kernel=2, stride=2)
    features[5]  – Stage 3: 9 ConvNeXt Blocks            → [B, 384, H/16, W/16]
    features[6]  – Downsample: LayerNorm + Conv2d(384, 768, kernel=2, stride=2)
    features[7]  – Stage 4: 3 ConvNeXt Blocks            → [B, 768, H/32, W/32]
          │
    AdaptiveAvgPool2d((1, 1))                            → [B, 768, 1, 1]
          │
    LayerNorm2d((768,))  (model.classifier[0])           → [B, 768, 1, 1]
          │
    Flatten(start_dim=1)                                 → [B, 768]

Stage Channels & Spatial Dimensions for 224 × 224 Input:
    - Stage 1: 96 channels,  56 × 56 spatial
    - Stage 2: 192 channels, 28 × 28 spatial
    - Stage 3: 384 channels, 14 × 14 spatial
    - Stage 4: 768 channels, 7 × 7 spatial
    - Output vector: 768-d (L2-ready patch embedding, LayerNorm applied)

For Multi-Scale Patches in HMST:
    - p = 32:  Stage 1 [B, 96, 8, 8]   → Stage 4 [B, 768, 1, 1] → [B, 768]
    - p = 64:  Stage 1 [B, 96, 16, 16] → Stage 4 [B, 768, 2, 2] → [B, 768]
    - p = 128: Stage 1 [B, 96, 32, 32] → Stage 4 [B, 768, 4, 4] → [B, 768]

Normalization:
    ConvNeXt-Tiny's IMAGENET1K_V1 weights require ImageNet standard normalization:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    This is already applied by `get_train_transforms` and `get_val_transforms`
    in `hmst/data/transforms.py`. No additional or duplicate normalization is applied.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv_models

# Public constant used by architecture1.py
CONVNEXT_TINY_DIM: int = 768
CONVNEXT_STAGE_CHANNELS: Tuple[int, ...] = (96, 192, 384, 768)


class ConvNeXtTinyBackbone(nn.Module):
    """
    Pretrained ConvNeXt-Tiny backbone for HMST.

    Extracts both global pooled feature vectors ([B, 768]) and intermediate
    hierarchical feature maps across the 4 stages.

    Usage:
        backbone = ConvNeXtTinyBackbone()
        # Flat feature vector:
        feat = backbone(patches)                     # [B, 768]
        # Intermediate hierarchical stage features:
        stages = backbone.extract_stages(images)     # dict of stage1..stage4
    """

    OUTPUT_DIM: int = CONVNEXT_TINY_DIM
    STAGE_CHANNELS: Tuple[int, ...] = CONVNEXT_STAGE_CHANNELS

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # Use official IMAGENET1K_V1 weights if pretrained is True
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)

        # model.features has 8 children:
        #   features[0]: stem (conv 4x4, stride 4, LayerNorm)
        #   features[1]: stage 1 blocks (x3)
        #   features[2]: downsample (LayerNorm, conv 2x2, stride 2)
        #   features[3]: stage 2 blocks (x3)
        #   features[4]: downsample (LayerNorm, conv 2x2, stride 2)
        #   features[5]: stage 3 blocks (x9)
        #   features[6]: downsample (LayerNorm, conv 2x2, stride 2)
        #   features[7]: stage 4 blocks (x3)
        self.features = model.features
        self.avgpool = model.avgpool  # AdaptiveAvgPool2d((1, 1))

        # model.classifier is Sequential:
        #   classifier[0]: LayerNorm2d((768,), eps=1e-06)
        #   classifier[1]: Flatten(start_dim=1)
        #   classifier[2]: Linear(768, 1000)
        # We retain classifier[0] so the global representation preserves the
        # exact pretrained ImageNet feature normalization and affine parameters,
        # and discard the classification Linear head.
        self.norm = model.classifier[0]

    def extract_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate hierarchical feature maps from all 4 stages.

        Args:
            x: [B, 3, H, W] where H, W >= 32

        Returns:
            dict with:
                'stage1': [B, 96,  H/4,  W/4]
                'stage2': [B, 192, H/8,  W/8]
                'stage3': [B, 384, H/16, W/16]
                'stage4': [B, 768, H/32, W/32]
        """
        # Stage 1: stem + stage1 blocks
        x1 = self.features[0](x)
        x1 = self.features[1](x1)

        # Stage 2: downsample + stage2 blocks
        x2 = self.features[2](x1)
        x2 = self.features[3](x2)

        # Stage 3: downsample + stage3 blocks
        x3 = self.features[4](x2)
        x3 = self.features[5](x3)

        # Stage 4: downsample + stage4 blocks
        x4 = self.features[6](x3)
        x4 = self.features[7](x4)

        return {
            "stage1": x1,
            "stage2": x2,
            "stage3": x3,
            "stage4": x4,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch feature extraction.

        Args:
            x: [B, 3, H, W] - normalized RGB patch tensor

        Returns:
            feat: [B, 768] - normalized global average pooled feature vector
        """
        feat = self.features(x)           # [B, 768, H/32, W/32]
        feat = self.avgpool(feat)         # [B, 768, 1, 1]
        feat = self.norm(feat)            # [B, 768, 1, 1] - pretrained LayerNorm2d
        feat = feat.flatten(start_dim=1)  # [B, 768]
        return feat
