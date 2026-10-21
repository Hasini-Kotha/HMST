"""
verify_backbones.py
-------------------
Comprehensive test suite for HMST backbone integration (ResNet-50 & ConvNeXt-Tiny).
"""

import sys
import os
import shutil
import tempfile
import torch
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models import HMST, ConvNeXtTinyBackbone, CONVNEXT_TINY_DIM
from training.checkpoint import save_checkpoint, load_checkpoint
from scripts.train import (
    dataset_output_dir,
    dataset_checkpoint_dir,
    save_results_for_plotting,
    update_summary_aggregates,
)


def test_1_backbone_defaults_and_validation():
    print("=== Test 1: Backbone Defaults and Validation ===")
    
    # 1. Default must be resnet50
    m_default = HMST()
    assert m_default.backbone_name == "resnet50", f"Default backbone should be resnet50, got {m_default.backbone_name}"
    assert m_default.BACKBONE_DIM == 2048, f"Expected 2048 for resnet50, got {m_default.BACKBONE_DIM}"
    print("  [PASS] Default backbone is resnet50 (dim=2048)")

    # 2. Explicit 'resnet50'
    m_rn = HMST(backbone="resnet50")
    assert m_rn.backbone_name == "resnet50"
    assert m_rn.BACKBONE_DIM == 2048
    print("  [PASS] Explicit 'resnet50' selects ResNet-50 (dim=2048)")

    # 3. Explicit 'convnext'
    m_cn = HMST(backbone="convnext")
    assert m_cn.backbone_name == "convnext"
    assert m_cn.BACKBONE_DIM == 768
    print("  [PASS] Explicit 'convnext' selects ConvNeXt-Tiny (dim=768)")

    # 4. Alias 'convnext_tiny' and 'convnext-tiny'
    m_cn_alias1 = HMST(backbone="convnext_tiny")
    assert m_cn_alias1.backbone_name == "convnext"
    m_cn_alias2 = HMST(backbone="convnext-tiny")
    assert m_cn_alias2.backbone_name == "convnext"
    print("  [PASS] Aliases 'convnext_tiny' and 'convnext-tiny' normalize to 'convnext'")

    # 5. Unsupported backbone raises ValueError
    unsupported_raised = False
    try:
        HMST(backbone="unsupported_model_xyz")
    except ValueError as e:
        unsupported_raised = True
        print(f"  [PASS] Unsupported backbone raised expected ValueError: {e}")
    assert unsupported_raised, "Unsupported backbone should raise ValueError!"


def test_2_convnext_stage_extraction_and_shapes():
    print("\n=== Test 2: ConvNeXt-Tiny Intermediate Stage Extraction ===")
    bb = ConvNeXtTinyBackbone(pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    stages = bb.extract_stages(x)
    
    assert "stage1" in stages and stages["stage1"].shape == (2, 96, 56, 56)
    assert "stage2" in stages and stages["stage2"].shape == (2, 192, 28, 28)
    assert "stage3" in stages and stages["stage3"].shape == (2, 384, 14, 14)
    assert "stage4" in stages and stages["stage4"].shape == (2, 768, 7, 7)
    print("  [PASS] Stage 1 shape: [2, 96, 56, 56]")
    print("  [PASS] Stage 2 shape: [2, 192, 28, 28]")
    print("  [PASS] Stage 3 shape: [2, 384, 14, 14]")
    print("  [PASS] Stage 4 shape: [2, 768, 7, 7]")

    feat = bb(x)
    assert feat.shape == (2, 768), f"Expected [2, 768], got {feat.shape}"
    print("  [PASS] Global pooled feature shape: [2, 768]")


def test_3_forward_passes_and_prediction_ranges():
    print("\n=== Test 3: Forward Passes with Dummy Tensor (1, 3, 224, 224) ===")
    dummy = torch.randn(1, 3, 224, 224)

    # 1. ResNet-50
    m_rn = HMST(backbone="resnet50")
    m_rn.eval()
    with torch.no_grad():
        out_rn = m_rn(dummy)
    assert out_rn["Q_hat"].shape == (1,), f"Expected [1], got {out_rn['Q_hat'].shape}"
    q_rn = out_rn["Q_hat"].item()
    assert 0.0 <= q_rn <= 1.0, f"Predicted score {q_rn} outside [0, 1]"
    assert len(out_rn["qualities"]) == 3
    assert len(out_rn["alphas"]) == 3
    assert out_rn["lambda_s"].shape == (1, 3)
    print(f"  [PASS] ResNet-50 forward pass PASS: Q_hat={q_rn:.4f} in [0, 1]")

    # 2. ConvNeXt-Tiny
    m_cn = HMST(backbone="convnext")
    m_cn.eval()
    with torch.no_grad():
        out_cn = m_cn(dummy)
    assert out_cn["Q_hat"].shape == (1,), f"Expected [1], got {out_cn['Q_hat'].shape}"
    q_cn = out_cn["Q_hat"].item()
    assert 0.0 <= q_cn <= 1.0, f"Predicted score {q_cn} outside [0, 1]"
    assert len(out_cn["qualities"]) == 3
    assert len(out_cn["alphas"]) == 3
    assert out_cn["lambda_s"].shape == (1, 3)
    print(f"  [PASS] ConvNeXt-Tiny forward pass PASS: Q_hat={q_cn:.4f} in [0, 1]")

    # Batch of 4 images
    dummy_b4 = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out_b4 = m_cn(dummy_b4)
    assert out_b4["Q_hat"].shape == (4,)
    print("  [PASS] Batch size 4 forward pass PASS: Q_hat shape [4]")


def test_4_checkpoint_compatibility_safeguard():
    print("\n=== Test 4: Checkpoint Management and Incompatibility Safeguard ===")
    temp_dir = tempfile.mkdtemp()
    try:
        rn_model = HMST(backbone="resnet50")
        cn_model = HMST(backbone="convnext")
        opt = torch.optim.Adam(rn_model.parameters(), lr=1e-4)

        rn_ckpt_path = os.path.join(temp_dir, "resnet50_best.pt")
        save_checkpoint(
            path=rn_ckpt_path,
            epoch=5,
            model=rn_model,
            optimizer=opt,
            best_metric=0.85,
            training_history=[],
            config={"backbone": "resnet50", "dataset": "uwiqa"},
        )
        print("  [PASS] Successfully saved ResNet-50 checkpoint")

        # Loading into ResNet-50 model should succeed
        rn_loaded = HMST(backbone="resnet50")
        load_checkpoint(rn_ckpt_path, rn_loaded)
        print("  [PASS] Loading ResNet-50 checkpoint into ResNet-50 model SUCCEEDED")

        # Loading into ConvNeXt-Tiny model must fail with RuntimeError
        cross_load_failed = False
        try:
            load_checkpoint(rn_ckpt_path, cn_model)
        except RuntimeError as e:
            cross_load_failed = True
            print(f"  [PASS] Loading ResNet-50 checkpoint into ConvNeXt-Tiny rejected with error:\n    '{e}'")
        assert cross_load_failed, "Loading ResNet-50 checkpoint into ConvNeXt should raise RuntimeError!"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_5_directory_separation_and_plotting_results():
    print("\n=== Test 5: Directory Structure and Results Logging ===")
    temp_dir = tempfile.mkdtemp()
    try:
        # Check output directory naming
        rn_out = dataset_output_dir("uwiqa", "full", temp_dir, backbone="resnet50")
        cn_out = dataset_output_dir("uwiqa", "full", temp_dir, backbone="convnext")
        rn_ckpt = dataset_checkpoint_dir("uwiqa", "full", temp_dir, backbone="resnet50")
        cn_ckpt = dataset_checkpoint_dir("uwiqa", "full", temp_dir, backbone="convnext")

        # Verify separation: ResNet preserves existing paths, ConvNeXt uses scoped paths
        assert os.path.normpath(rn_out) == os.path.normpath(os.path.join(temp_dir, "outputs/uwiqa/full"))
        assert os.path.normpath(cn_out) == os.path.normpath(os.path.join(temp_dir, "outputs/uwiqa_convnext/full"))
        assert os.path.normpath(rn_ckpt) == os.path.normpath(os.path.join(temp_dir, "checkpoints/uwiqa/full"))
        assert os.path.normpath(cn_ckpt) == os.path.normpath(os.path.join(temp_dir, "checkpoints/uwiqa_convnext/full"))
        print("  [PASS] ResNet-50 output dir:      ", rn_out)
        print("  [PASS] ConvNeXt-Tiny output dir:  ", cn_out)
        print("  [PASS] ResNet-50 checkpoint dir:  ", rn_ckpt)
        print("  [PASS] ConvNeXt-Tiny ckpt dir:    ", cn_ckpt)

        # Mock results saving for both backbones across 2 iterations
        dummy_metrics_tta = {"srocc": 0.88, "plcc": 0.89, "krcc": 0.70, "rmse": 0.08, "mae": 0.06, "medae": 0.05}
        dummy_metrics_no_tta = {"srocc": 0.86, "plcc": 0.87, "krcc": 0.68, "rmse": 0.09, "mae": 0.07, "medae": 0.06}
        dummy_history = [
            {"epoch": 1, "train_loss": 0.45, "val_loss": 0.40, "val_srocc": 0.81, "val_plcc": 0.82, "val_rmse": 0.12, "learning_rate": 1e-4},
            {"epoch": 2, "train_loss": 0.35, "val_loss": 0.30, "val_srocc": 0.88, "val_plcc": 0.89, "val_rmse": 0.08, "learning_rate": 1e-4},
        ]
        dummy_preds = [
            {"image": "img1.png", "actual_mos": 0.75, "pred_mos_tta": 0.73, "pred_mos_no_tta": 0.71, "abs_error_tta": 0.02, "error_tta": -0.02, "path": "UWIQA/img1.png"},
            {"image": "img2.png", "actual_mos": 0.40, "pred_mos_tta": 0.42, "pred_mos_no_tta": 0.44, "abs_error_tta": 0.02, "error_tta": 0.02, "path": "UWIQA/img2.png"},
        ]
        dummy_cfg = {"image_size": 224, "patch_sizes": [32, 64, 128], "d_model": 512, "batch_size": 4, "epochs": 2}

        # Save resnet50 iter 1
        save_results_for_plotting(temp_dir, "resnet50", "uwiqa", 1, 42, dummy_metrics_tta, dummy_metrics_no_tta, dummy_history, dummy_preds, dummy_cfg, best_epoch=2, best_val_metric=0.88, test_sample_count=2)
        # Save convnext iter 1
        save_results_for_plotting(temp_dir, "convnext", "uwiqa", 1, 42, dummy_metrics_tta, dummy_metrics_no_tta, dummy_history, dummy_preds, dummy_cfg, best_epoch=2, best_val_metric=0.88, test_sample_count=2)
        update_summary_aggregates(temp_dir)

        # Verify directory structure
        rn_res = os.path.join(temp_dir, "results/resnet50/UWIQA/iteration_01")
        cn_res = os.path.join(temp_dir, "results/convnext_tiny/UWIQA/iteration_01")
        assert os.path.exists(os.path.join(rn_res, "metrics.json"))
        assert os.path.exists(os.path.join(rn_res, "epoch_history.csv"))
        assert os.path.exists(os.path.join(rn_res, "predictions.csv"))
        assert os.path.exists(os.path.join(cn_res, "metrics.json"))
        assert os.path.exists(os.path.join(cn_res, "epoch_history.csv"))
        assert os.path.exists(os.path.join(cn_res, "predictions.csv"))
        print("  [PASS] All required result files exist in results/ structure:")
        print("    -", os.path.join(rn_res, "metrics.json"))
        print("    -", os.path.join(cn_res, "predictions.csv"))

        # Verify pandas readability of summary and predictions
        summary_csv = os.path.join(temp_dir, "results/summary.csv")
        df_summary = pd.read_csv(summary_csv)
        assert len(df_summary) == 2, f"Expected 2 rows in summary.csv, got {len(df_summary)}"
        assert set(df_summary["backbone"]) == {"resnet50", "convnext_tiny"}
        assert "srocc_tta" in df_summary.columns
        assert "srocc_no_tta" in df_summary.columns
        print(f"  [PASS] pandas successfully loaded summary.csv ({len(df_summary)} rows):")
        print(df_summary[["backbone", "dataset", "iteration", "srocc_tta", "srocc_no_tta"]])

        df_preds = pd.read_csv(os.path.join(cn_res, "predictions.csv"))
        assert len(df_preds) == 2
        assert "actual_mos" in df_preds.columns
        assert "pred_mos_tta" in df_preds.columns
        assert "pred_mos_no_tta" in df_preds.columns
        print("  [PASS] pandas successfully loaded predictions.csv")

        agg_csv = os.path.join(temp_dir, "results/summary_aggregates.csv")
        df_agg = pd.read_csv(agg_csv)
        assert len(df_agg) == 2
        print("  [PASS] pandas successfully loaded summary_aggregates.csv")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING HMST CONVNEXT & RESNET BACKBONE VALIDATION SUITE")
    print("=" * 60)
    test_1_backbone_defaults_and_validation()
    test_2_convnext_stage_extraction_and_shapes()
    test_3_forward_passes_and_prediction_ranges()
    test_4_checkpoint_compatibility_safeguard()
    test_5_directory_separation_and_plotting_results()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 60)
