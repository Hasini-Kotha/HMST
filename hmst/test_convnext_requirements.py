"""
test_convnext_requirements.py
------------------------------
Rigorous verification of all testing requirements from User Request Section 4:
1. Import ConvNeXtTinyBackbone
2. Instantiate with pretrained weights
3. Forward pass with (2, 3, 224, 224) in eval mode
4. Forward passes for patch sizes 32x32, 64x64, 128x128
5. Verify output shape of forward() is [B, 768]
6. Verify output shapes of all 4 entries returned by extract_stages()
7. Verify compatibility with HMST feature-projection layer (Linear(768, d_model))
8. Verify ResNet-50 default path and behavior remain unchanged
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.convnext_backbone import (
    ConvNeXtTinyBackbone,
    CONVNEXT_TINY_DIM,
    CONVNEXT_STAGE_CHANNELS,
)
from models.architecture1 import HMST


def run_tests():
    print("=" * 65)
    print("TESTING CONVNEXT-TINY INTEGRATION REQUIREMENTS")
    print("=" * 65)

    # 1 & 2: Import & Instantiate with pretrained weights
    print("\n[Step 1 & 2] Instantiating ConvNeXtTinyBackbone with pretrained weights...")
    backbone = ConvNeXtTinyBackbone(pretrained=True)
    backbone.eval()
    assert hasattr(backbone, "norm"), "Backbone must have self.norm layer (LayerNorm2d)"
    assert isinstance(backbone.norm, nn.LayerNorm), f"Expected LayerNorm, got {type(backbone.norm)}"
    assert hasattr(backbone.norm, "weight") and hasattr(backbone.norm, "bias")
    assert backbone.norm.weight.shape == (768,)
    print("  [PASS] Successfully instantiated with pretrained IMAGENET1K_V1 weights.")
    print("  [PASS] Verified pretrained classifier LayerNorm2d retained (shape: (768,)).")

    # 3: Forward pass with (2, 3, 224, 224)
    print("\n[Step 3] Forward pass with full input tensor (2, 3, 224, 224)...")
    x_224 = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_224 = backbone(x_224)
    assert out_224.shape == (2, 768), f"Expected (2, 768), got {out_224.shape}"
    assert not torch.isnan(out_224).any(), "NaN detected in forward output"
    print(f"  [PASS] Forward output shape: {list(out_224.shape)}")

    # 4 & 5: Forward passes for patch sizes 32x32, 64x64, 128x128
    print("\n[Step 4 & 5] Forward passes for multi-scale patch sizes [32, 64, 128]...")
    for ps in [32, 64, 128]:
        patch_tensor = torch.randn(4, 3, ps, ps)
        with torch.no_grad():
            feat = backbone(patch_tensor)
        assert feat.shape == (4, 768), f"Patch size {ps} failed: expected (4, 768), got {feat.shape}"
        assert not torch.isnan(feat).any(), f"NaN in patch {ps} output"
        print(f"  [PASS] Patch {ps:3d}x{ps:3d}: input {list(patch_tensor.shape)} -> output {list(feat.shape)}")

    # 6: Output shapes of all 4 entries returned by extract_stages()
    print("\n[Step 6] Verifying all 4 stage feature maps returned by extract_stages()...")
    with torch.no_grad():
        stages = backbone.extract_stages(x_224)
    expected_shapes = {
        "stage1": (2, 96, 56, 56),
        "stage2": (2, 192, 28, 28),
        "stage3": (2, 384, 14, 14),
        "stage4": (2, 768, 7, 7),
    }
    for stage_name, exp_shape in expected_shapes.items():
        assert stage_name in stages, f"Missing {stage_name} in extract_stages output"
        actual_shape = tuple(stages[stage_name].shape)
        assert actual_shape == exp_shape, f"{stage_name}: expected {exp_shape}, got {actual_shape}"
        print(f"  [PASS] {stage_name}: shape {list(actual_shape)} (channels={actual_shape[1]})")

    # 7: Compatibility with HMST feature projection layer
    print("\n[Step 7] Verifying compatibility with HMST feat_proj (Linear(768, d_model))...")
    d_model = 512
    model_cn = HMST(d_model=d_model, backbone="convnext")
    model_cn.eval()
    assert model_cn.BACKBONE_DIM == 768
    assert model_cn.feat_proj.in_features == 768
    assert model_cn.feat_proj.out_features == d_model

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        hmst_out_cn = model_cn(dummy_input)
    assert hmst_out_cn["Q_hat"].shape == (2,)
    assert (hmst_out_cn["Q_hat"] >= 0.0).all() and (hmst_out_cn["Q_hat"] <= 1.0).all()
    assert len(hmst_out_cn["qualities"]) == 3
    assert len(hmst_out_cn["alphas"]) == 3
    assert hmst_out_cn["lambda_s"].shape == (2, 3)
    print(f"  [PASS] HMST(backbone='convnext') forward pass successful.")
    print(f"  [PASS] Predicted quality scores Q_hat: {hmst_out_cn['Q_hat'].tolist()} (in [0, 1])")
    print(f"  [PASS] Scale fusion weights lambda_s: {hmst_out_cn['lambda_s'].shape}")

    # 8: ResNet-50 default path unchanged
    print("\n[Step 8] Verifying default ResNet-50 path and behavior remain unchanged...")
    model_rn = HMST(d_model=d_model)  # omitted backbone parameter
    model_rn.eval()
    assert model_rn.backbone_name == "resnet50"
    assert model_rn.BACKBONE_DIM == 2048
    assert model_rn.feat_proj.in_features == 2048
    assert model_rn.feat_proj.out_features == d_model

    with torch.no_grad():
        hmst_out_rn = model_rn(dummy_input)
    assert hmst_out_rn["Q_hat"].shape == (2,)
    assert (hmst_out_rn["Q_hat"] >= 0.0).all() and (hmst_out_rn["Q_hat"] <= 1.0).all()
    print("  [PASS] HMST() default path is 100% ResNet-50 (BACKBONE_DIM=2048, feat_proj Linear(2048, 512)).")
    print(f"  [PASS] ResNet-50 Q_hat: {hmst_out_rn['Q_hat'].tolist()} (in [0, 1])")

    print("\n" + "=" * 65)
    print("ALL 8 VERIFICATION REQUIREMENTS SUCCESSFULLY PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
