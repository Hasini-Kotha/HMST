"""
scripts/evaluate.py
--------------------
Evaluate a trained HMST model on the test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/full/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/test/best.pt --mode test
"""

import sys
import os
import argparse
import json
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
import yaml

from scripts.train import (
    load_config,
    build_datasets,
    get_device,
    make_absolute,
    set_seed,
    dataset_output_dir,
    build_full_evaluation_dataset,
    resolve_config_path,
    validate_training_config,
)
from models import HMST, build_correspondence_matrix
from models.losses import HMSTLoss
from training.checkpoint import load_checkpoint
from training.metrics import (
    compute_metrics,
    compute_test_metrics,
    collect_predictions,
    collect_predictions_tta,
)
from training.plots import save_error_analysis_plots, save_quality_plots


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMST Architecture 1")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration YAML file (e.g. configs/uwiqa_tuned.yaml)")
    parser.add_argument("--mode", type=str, choices=["test", "full"], default="full",
                        help="Config mode fallback if --config is not specified")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override evaluation batch size (defaults to batch_size in config)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["uwiqa", "saud"])
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["resnet50", "convnext", "convnext_tiny"],
                        help="Override backbone architecture: 'resnet50' (default) or 'convnext'")
    args = parser.parse_args()

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    # Load configuration: --config has priority over --mode
    if args.config:
        cfg_path = resolve_config_path(args.config, project_root)
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        validate_training_config(config, cfg_path)
        print(f"[Config] Successfully loaded and validated custom config: {cfg_path}")
    else:
        config = load_config(args.mode)
        cfg_path = os.path.normpath(os.path.join(project_root, "configs", f"{args.mode}.yaml"))
        validate_training_config(config, cfg_path)

    if args.dataset:
        config["dataset"] = args.dataset

    mode    = config.get("mode", args.mode or "full")
    dataset = config.get("dataset", "uwiqa")

    # Read the checkpoint's stored config early to determine dataset and backbone
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg  = ckpt_meta.get("config", {})
    ckpt_ds   = ckpt_cfg.get("dataset")
    if ckpt_ds is not None and ckpt_ds != dataset:
        raise RuntimeError(
            f"Checkpoint '{args.checkpoint}' records dataset '{ckpt_ds}' "
            f"but evaluation is configured for dataset '{dataset}'."
        )

    # Determine backbone: CLI flag > checkpoint config > YAML config > 'resnet50'
    if args.backbone is not None:
        selected_backbone = args.backbone.lower().strip().replace("-", "_")
        if selected_backbone == "convnext_tiny":
            selected_backbone = "convnext"
    else:
        cfg_bb = ckpt_cfg.get("backbone", config.get("backbone", "resnet50"))
        selected_backbone = str(cfg_bb).lower().strip().replace("-", "_")
        if selected_backbone == "convnext_tiny":
            selected_backbone = "convnext"

    if selected_backbone not in ("resnet50", "convnext"):
        raise ValueError(
            f"Unsupported backbone '{args.backbone or ckpt_cfg.get('backbone')}'. "
            f"Supported values: ['resnet50', 'convnext']"
        )
    backbone = selected_backbone
    out_dir = dataset_output_dir(dataset, mode, project_root, backbone=backbone)

    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(config)

    # Resolve evaluation batch size
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError(f"Invalid --batch-size {args.batch_size}. Must be a positive integer.")
        batch_size = args.batch_size
        batch_source = f"CLI override (--batch-size {batch_size})"
    else:
        batch_size = config.get("batch_size")
        if batch_size is None or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Invalid or missing batch_size in config '{cfg_path}': {batch_size}")
        batch_source = f"Config ({cfg_path})"

    print(f"\n{'='*60}")
    print(f"Evaluating HMST Architecture 1")
    print(f"{'='*60}")
    print(f"Config:      {cfg_path}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Backbone:    {backbone}")
    print(f"Device:      {device}")
    print(f"Dataset:     {dataset}")
    print(f"Batch Size:  {batch_size} ({batch_source})")
    print(f"Output dir:  {out_dir}")

    # Dataset
    _, val_ds, test_ds = build_datasets(config, project_root)
    num_workers = config.get("num_workers", 0)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Model
    patch_sizes = config.get("patch_sizes", [32, 64, 128])
    image_size  = config.get("image_size", 224)

    model = HMST(
        image_size=ckpt_cfg.get("image_size", image_size),
        patch_sizes=ckpt_cfg.get("patch_sizes", patch_sizes),
        d_model=int(ckpt_cfg.get("d_model", config.get("d_model", 512))),
        nhead=int(ckpt_cfg.get("nhead", config.get("nhead", 8))),
        nlayers=int(ckpt_cfg.get("nlayers", config.get("nlayers", 2))),
        n_regions=int(ckpt_cfg.get("n_regions", config.get("n_regions", 4))),
        dropout=float(ckpt_cfg.get("dropout", config.get("dropout", 0.3))),
        backbone=backbone,
    ).to(device)

    # Load checkpoint weights
    ckpt = load_checkpoint(args.checkpoint, model, device=device)
    trained_epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {trained_epoch}")

    os.makedirs(out_dir, exist_ok=True)

    # Evaluate on val and test
    plot_results = {}
    for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
        metric_fn = compute_test_metrics if split_name == "Test" else compute_metrics
        # Standard single-pass evaluation
        result = collect_predictions(model, loader, device)
        metrics = metric_fn(result["predictions"], result["targets"])

        # TTA evaluation (4-view flip average)
        result_tta = collect_predictions_tta(model, loader, device)
        metrics_tta = metric_fn(result_tta["predictions"], result_tta["targets"])
        plot_results[split_name.lower()] = (result, result_tta, metrics_tta)

        # Per-image absolute errors (TTA + no-TTA) -> MAE / MedAE
        actuals = result_tta["targets"]
        preds_tta = result_tta["predictions"]
        preds_no_tta = result["predictions"]
        abs_errors_tta = np.abs(preds_tta - actuals)
        abs_errors_no_tta = np.abs(preds_no_tta - actuals)
        mae_tta = float(np.mean(abs_errors_tta))
        medae_tta = float(np.median(abs_errors_tta))
        mae_no_tta = float(np.mean(abs_errors_no_tta))
        medae_no_tta = float(np.median(abs_errors_no_tta))

        # Persist MAE / MedAE into the metrics dict (both TTA and no-TTA)
        metrics_tta.update({"mae": mae_tta, "medae": medae_tta})
        metrics.update({"mae": mae_no_tta, "medae": medae_no_tta})

        print(f"\n{split_name} Metrics (n={len(result['predictions'])}):")
        print(f"  {'Metric':<12}  {'No TTA':>8}  {'TTA (4-view)':>12}")
        print(f"  {'-'*34}")
        plcc_label = "PLCC (raw)" if split_name == "Test" else "PLCC (4PL)"
        rmse_label = "RMSE (raw)" if split_name == "Test" else "RMSE (4PL)"
        print(f"  {plcc_label:<12}  {metrics['plcc']:>8.4f}  {metrics_tta['plcc']:>12.4f}")
        print(f"  {'SROCC':<8}  {metrics['srocc']:>8.4f}  {metrics_tta['srocc']:>12.4f}")
        print(f"  {'KRCC':<8}  {metrics['krcc']:>8.4f}  {metrics_tta['krcc']:>12.4f}")
        print(f"  {rmse_label:<12}  {metrics['rmse']:>8.4f}  {metrics_tta['rmse']:>12.4f}")
        if split_name == "Test":
            print(
                "  Same-split 4PL fit (not out-of-sample): "
                f"PLCC={metrics['same_split_fitted_plcc']:.4f}/"
                f"{metrics_tta['same_split_fitted_plcc']:.4f}, "
                f"RMSE={metrics['same_split_fitted_rmse']:.4f}/"
                f"{metrics_tta['same_split_fitted_rmse']:.4f} "
                "(No TTA/TTA)"
            )
        print(f"  {'MAE':<8}  {mae_no_tta:>8.4f}  {mae_tta:>12.4f}")
        print(f"  {'MedAE':<8}  {medae_no_tta:>8.4f}  {medae_tta:>12.4f}")

        # Save metrics JSON (TTA as primary, plus no-TTA reference)
        with open(os.path.join(out_dir, f"{split_name.lower()}_metrics.json"), "w") as f:
            json.dump(metrics_tta, f, indent=2)
        with open(os.path.join(out_dir, f"{split_name.lower()}_metrics_no_tta.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Per-image predictions CSV (Actual vs Predicted analysis)
        paths = result_tta.get("paths", result.get("paths", []))
        pred_csv_path = os.path.join(out_dir, f"{split_name.lower()}_predictions.csv")
        rows = []
        for idx in range(len(actuals)):
            full_path = paths[idx] if idx < len(paths) else f"{split_name.lower()}_{idx:04d}"
            rows.append({
                "image": os.path.basename(full_path),
                "actual_mos": float(actuals[idx]),
                "pred_mos_tta": float(preds_tta[idx]),
                "pred_mos_no_tta": float(preds_no_tta[idx]),
                "abs_error_tta": float(abs_errors_tta[idx]),
                "error_tta": float(preds_tta[idx] - actuals[idx]),
                "path": full_path,
            })

        with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "actual_mos", "pred_mos_tta", "pred_mos_no_tta", "abs_error_tta", "error_tta", "path"])
            for r in rows:
                writer.writerow([
                    r["image"],
                    f"{r['actual_mos']:.4f}",
                    f"{r['pred_mos_tta']:.4f}",
                    f"{r['pred_mos_no_tta']:.4f}",
                    f"{r['abs_error_tta']:.4f}",
                    f"{r['error_tta']:.4f}",
                    r["path"],
                ])

        print(f"  Metrics saved to:      {os.path.join(out_dir, f'{split_name.lower()}_metrics.json')}")
        print(f"  Per-image CSV saved to: {pred_csv_path}")

    if dataset in {"uwiqa", "saud"}:
        validation_result, validation_result_tta, validation_metrics = plot_results["validation"]
        test_result, test_result_tta, test_metrics = plot_results["test"]
        plot_paths = save_quality_plots(
            actual_test=test_result_tta["targets"],
            predicted_test_tta=test_result_tta["predictions"],
            test_metrics=test_metrics,
            actual_validation=validation_result_tta["targets"],
            predicted_validation_tta=validation_result_tta["predictions"],
            validation_metrics=validation_metrics,
            output_dir=os.path.join(out_dir, "plots"),
        )
        pareto_dataset = build_full_evaluation_dataset(config, project_root)
        pareto_result = test_result_tta
        if pareto_dataset is not None:
            pareto_loader = DataLoader(
                pareto_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            pareto_result = collect_predictions_tta(model, pareto_loader, device)
        error_plot_paths = save_error_analysis_plots(
            actual=test_result_tta["targets"],
            predicted_tta=test_result_tta["predictions"],
            predicted_no_tta=test_result["predictions"],
            image_paths=test_result_tta.get("paths", []),
            output_dir=os.path.join(out_dir, "plots"),
            pareto_actual=pareto_result["targets"],
            pareto_predicted=pareto_result["predictions"],
        )
        print("Correlation plots saved:")
        for plot_path in (*plot_paths, *error_plot_paths):
            print(f"  {plot_path}")


if __name__ == "__main__":
    main()
