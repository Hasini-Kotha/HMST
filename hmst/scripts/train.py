"""
scripts/train.py
----------------
Main entry point for HMST Architecture 1 training.

Usage:
    python scripts/train.py --mode test
    python scripts/train.py --mode full
    python scripts/train.py --mode full --resume checkpoints/full/latest.pt

The --mode flag selects configs/test.yaml or configs/full.yaml.
A --resume flag overrides config.resume and provides the checkpoint path.
"""

import sys
import os
import argparse
import random
import time

# Make the project root importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import (
    UWIQADataset,
    SAUDDataset,
    make_uwiqa_splits,
    make_saud_splits,
    get_train_transforms,
    get_val_transforms,
)
from models import HMST, HMSTLoss, build_correspondence_matrix
from training import HMSTTrainer, build_optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """Auto-detect best available device or use config override."""
    dev_str = config.get("device", "auto")
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev_str)
    return device


def load_config(mode: str) -> dict:
    """Load YAML config for the given mode."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", f"{mode}.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_absolute(path: str, base: str) -> str:
    """Make a path absolute relative to base if it isn't already, with fallback resolution."""
    if os.path.isabs(path):
        resolved = path
    else:
        resolved = os.path.normpath(os.path.join(base, path))

    if os.path.exists(resolved):
        return resolved

    # Fallback: check without redundant parent directory (e.g. UWIQA/UWIQA dataset -> UWIQA dataset)
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-2] == "UWIQA" and "UWIQA dataset" in parts[-1]:
        alt_path = path.replace("UWIQA/UWIQA dataset", "UWIQA dataset")
        alt_resolved = os.path.normpath(os.path.join(base, alt_path)) if not os.path.isabs(alt_path) else alt_path
        if os.path.exists(alt_resolved):
            return alt_resolved

    return resolved


def get_dataset_config(config: dict):
    """Return the selected dataset name and its validated configuration."""
    dataset_name = config.get("dataset", "uwiqa")
    required_fields = {
        "uwiqa": ("image_dir", "score_file"),
        "saud": ("benchmark_dir", "bt_scores_mat"),
        "uied": ("image_dir", "score_file"),
    }
    if dataset_name not in required_fields:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_config = config.get(dataset_name)
    if not isinstance(dataset_config, dict):
        raise ValueError(
            f"Dataset '{dataset_name}' is selected, but its '{dataset_name}:' "
            "configuration section is missing or invalid."
        )

    missing_fields = [
        field for field in required_fields[dataset_name]
        if not dataset_config.get(field)
    ]
    if missing_fields:
        raise ValueError(
            f"Dataset '{dataset_name}' configuration is missing required "
            f"field(s): {', '.join(missing_fields)}."
        )

    return dataset_name, dataset_config


def get_run_output_dir(project_root: str, dataset_name: str, mode: str,
                       iteration_suffix: str = "") -> str:
    """Keep results for each dataset in a separate output tree."""
    return make_absolute(
        os.path.join("outputs", dataset_name, f"{mode}{iteration_suffix}"),
        project_root,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(config: dict, project_root: str):
    """Build train/val/test datasets from config."""
    dataset_name, dataset_config = get_dataset_config(config)
    image_size   = config.get("image_size", 224)

    aug_cfg  = config.get("augmentation", {}).get("train", {})
    train_tf = get_train_transforms(
        image_size = image_size,
        hflip_p    = aug_cfg.get("horizontal_flip", 0.5),
        vflip_p    = aug_cfg.get("vertical_flip", 0.5),
        degrees    = aug_cfg.get("rotation", 10.0),
    )
    val_tf   = get_val_transforms(image_size)

    max_train = config.get("max_train_samples")
    max_val   = config.get("max_val_samples")
    max_test  = config.get("max_test_samples")
    seed      = config.get("seed", 42)

    if dataset_name == "uwiqa":
        image_dir  = make_absolute(dataset_config["image_dir"],  project_root)
        score_file = make_absolute(dataset_config["score_file"], project_root)

        train_idx, test_idx = make_uwiqa_splits(image_dir, score_file, seed=seed)

        # Val = last 20% of train indices
        val_size = max(1, len(train_idx) // 5)
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = UWIQADataset(image_dir, score_file, train_idx, train_tf, max_train)
        val_ds   = UWIQADataset(image_dir, score_file, val_idx,   val_tf,   max_val)
        test_ds  = UWIQADataset(image_dir, score_file, test_idx,  val_tf,   max_test)

    elif dataset_name == "saud":
        benchmark_dir = make_absolute(dataset_config["benchmark_dir"], project_root)
        bt_mat        = make_absolute(dataset_config["bt_scores_mat"], project_root)

        train_idx, test_idx = make_saud_splits(bt_mat, seed=seed)

        val_size  = max(1, len(train_idx) // 5)
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = SAUDDataset(benchmark_dir, bt_mat, train_idx, train_tf, max_train)
        val_ds   = SAUDDataset(benchmark_dir, bt_mat, val_idx,   val_tf,   max_val)
        test_ds  = SAUDDataset(benchmark_dir, bt_mat, test_idx,  val_tf,   max_test)

    elif dataset_name == "uied":
        image_dir  = make_absolute(dataset_config["image_dir"],  project_root)
        score_file = make_absolute(dataset_config["score_file"], project_root)

        # Import locally to avoid top-level issues if not present
        from data import UIEDDataset, make_uied_splits
        
        train_idx, test_idx = make_uied_splits(score_file, seed=seed)

        val_size  = max(1, len(train_idx) // 5)
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = UIEDDataset(image_dir, score_file, train_idx, train_tf, max_train)
        val_ds   = UIEDDataset(image_dir, score_file, val_idx,   val_tf,   max_val)
        test_ds  = UIEDDataset(image_dir, score_file, test_idx,  val_tf,   max_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HMST — Architecture 1")
    parser.add_argument(
        "--mode", type=str, choices=["test", "full"], default="test",
        help="Select configs/test.yaml or configs/full.yaml"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom config YAML file (e.g. configs/uwiqa_tuned.yaml). Overrides --mode"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume from (overrides config.resume)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, choices=["uwiqa", "saud", "uied"],
        help="Override dataset from config"
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of iterations to run with random splits"
    )
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    # ── Config ────────────────────────────────────────────────────────────
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(project_root, args.config)
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.mode)
    if args.dataset:
        config["dataset"] = args.dataset

    dataset_name, _ = get_dataset_config(config)
    mode = config["mode"]  # test or full

    # Override resume from CLI
    resume_path = args.resume
    if resume_path is None and config.get("resume", False):
        default_resume = os.path.join(
            project_root, config["checkpoint_dir"], "latest.pt"
        )
        if os.path.exists(default_resume):
            resume_path = default_resume

    # ── Iterations Loop ───────────────────────────────────────────────────
    num_iterations = getattr(args, "iterations", 1)
    base_seed = config.get("seed", 42)
    
    if num_iterations > 1:
        seed_print = f"Rolling (Base: {base_seed})"
    else:
        seed_print = str(base_seed) if base_seed is not None else "Random (None)"

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device(config)
    print(f"\n{'='*60}")
    print(f"HMST Architecture 1 — Training")
    print(f"{'='*60}")
    print(f"Mode:     {mode}")
    print(f"Dataset:  {config.get('dataset', 'uwiqa')}")
    print(f"Device:   {device}")
    if device.type == "cuda":
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Seed:     {seed_print}")

    all_metrics = []

    for it in range(num_iterations):
        iter_suffix = f"_iter{it+1}" if num_iterations > 1 else ""
        checkpoint_dir = make_absolute(config["checkpoint_dir"] + iter_suffix, project_root)
        output_dir = get_run_output_dir(project_root, dataset_name, mode, iter_suffix)
        final_path = os.path.join(output_dir, "test_metrics.json")

        if num_iterations > 1 and os.path.exists(final_path):
            import json
            with open(final_path, "r") as f:
                cached_metrics = json.load(f)
            if cached_metrics.get("dataset") == dataset_name:
                print(f"\n[Trainer] Iteration {it+1} already completed for {dataset_name}. Loading metrics and skipping...")
                all_metrics.append(cached_metrics)
                continue
            print(
                f"\n[Trainer] Ignoring iteration {it+1} metrics without matching "
                f"dataset metadata; rerunning."
            )

        if num_iterations > 1:
            print(f"\n{'#'*60}")
            print(f" ITERATION {it+1}/{num_iterations}")
            print(f"{'#'*60}")
            
        # Predictable seed for resuming dataset splits exactly
        iter_seed = base_seed + it if base_seed is not None else 42000 + it
        config["seed"] = iter_seed
        set_seed(iter_seed)
        
        # ── Datasets ──────────────────────────────────────────────────────────
        print(f"\nBuilding datasets...")
        train_ds, val_ds, test_ds = build_datasets(config, project_root)

        print(f"  Train:  {len(train_ds)} samples")
        print(f"  Val:    {len(val_ds)} samples")
        print(f"  Test:   {len(test_ds)} samples")

        num_workers = config.get("num_workers", 0)
        batch_size  = config.get("batch_size", 4)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            drop_last=(len(train_ds) > batch_size),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

        # ── Model ─────────────────────────────────────────────────────────────
        print(f"\nBuilding HMST model...")
        patch_sizes  = config.get("patch_sizes", [32, 64, 128])
        image_size   = config.get("image_size", 224)
        print_shapes = config.get("print_shapes", False)

        d_model   = int(config.get("d_model", 512))
        nhead     = int(config.get("nhead", 8))
        nlayers   = int(config.get("nlayers", 2))
        dropout   = float(config.get("dropout", 0.3))
        n_regions = int(config.get("n_regions", 4))

        model = HMST(
            image_size   = image_size,
            patch_sizes  = patch_sizes,
            d_model      = d_model,
            nhead        = nhead,
            nlayers      = nlayers,
            n_regions    = n_regions,
            dropout      = dropout,
            print_shapes = print_shapes if it == 0 else False,
        ).to(device)

        if it == 0:
            total_params     = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total parameters:     {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

        # ── Loss (§4.9) ──────────────────────────────────────────────────────
        print(f"\nBuilding loss function...")
        M = build_correspondence_matrix(
            num_patches_fine   = model.num_patches[0],   # 49 (p=32)
            num_patches_coarse = model.num_patches[2],   # 1  (p=128)
            grid_fine          = model.grid_sizes[0],    # 7
            grid_coarse        = model.grid_sizes[2],    # 1
        ).to(device)

        loss_cfg = config.get("loss", {})
        criterion = HMSTLoss(
            M            = M,
            lambda_mos   = loss_cfg.get("lambda_mos",  1.0),
            lambda_rank  = loss_cfg.get("lambda_rank", 1.0),
            lambda_cons  = loss_cfg.get("lambda_cons", 0.1),
            margin       = loss_cfg.get("margin", 0.1),
        )

        # ── Optimizer ───────────────────────────────────────────────────────────
        opt_cfg = config.get("optimizer", {})
        opt_name = opt_cfg.get("name", "adam").lower()
        lr        = float(opt_cfg.get("lr", 1e-4))
        wd        = float(opt_cfg.get("weight_decay", 1e-4))
        opt_extra = {k: v for k, v in opt_cfg.items() if k not in ("name", "lr", "weight_decay")}
        optimizer = build_optimizer(model, opt_name, lr=lr, weight_decay=wd, **opt_extra)

        # ── Scheduler ─────────────────────────────────────────────────────────
        sched_name = config.get("scheduler", "none")
        scheduler = None
        if sched_name == "cosine":
            total_epochs = config.get("epochs", 50)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=lr * 0.01
            )
        elif sched_name == "step":
            total_epochs = config.get("epochs", 50)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(5, total_epochs // 3), gamma=0.5
            )

        # ── Trainer ───────────────────────────────────────────────────────────
        trainer = HMSTTrainer(
            model          = model,
            criterion      = criterion,
            optimizer      = optimizer,
            config         = config,
            device         = device,
            output_dir     = output_dir,
            checkpoint_dir = checkpoint_dir,
            scheduler      = scheduler,
        )

        # ── Shape validation (test mode) ──────────────────────────────────────
        if mode == "test" and it == 0:
            print(f"\n{'─'*60}")
            print("Running architecture shape validation (test mode)...")
            model.print_shapes = False
            dummy = torch.zeros(2, 3, image_size, image_size).to(device)
            with torch.no_grad():
                out = model(dummy)
            print(f"Output Q_hat shape: {list(out['Q_hat'].shape)}")
            assert out["Q_hat"].shape == (2,), f"Expected [2], got {out['Q_hat'].shape}"
            print("Shape validation: PASS")
            print(f"{'─'*60}\n")

        model.print_shapes = False

        # ── Train ─────────────────────────────────────────────────────────────
        iter_resume_path = None
        if config.get("resume", False):
            default_iter_resume = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.exists(default_iter_resume):
                iter_resume_path = default_iter_resume
                
        if args.resume and it == 0:
            iter_resume_path = args.resume

        trainer.fit(train_loader, val_loader, resume_path=iter_resume_path)

        # ── Final test evaluation ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Final evaluation on test set for iteration {it+1}...")
        from training.metrics import compute_metrics, collect_predictions

        best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_ckpt_path):
            from training.checkpoint import load_checkpoint
            load_checkpoint(best_ckpt_path, model, device=device)
            print(f"Loaded best.pt for final evaluation")

        result = collect_predictions(model, test_loader, device)
        final_metrics = compute_metrics(result["predictions"], result["targets"])
        final_metrics["dataset"] = dataset_name
        all_metrics.append(final_metrics)

        import json
        final_path = os.path.join(output_dir, "test_metrics.json")
        with open(final_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        print(f"\nTest Set Results (Iter {it+1}):")
        print(f"  PLCC:  {final_metrics['plcc']:.4f}")
        print(f"  SROCC: {final_metrics['srocc']:.4f}")
        print(f"  KRCC:  {final_metrics['krcc']:.4f}")
        print(f"  RMSE:  {final_metrics['rmse']:.4f}")
        print(f"Results saved to: {final_path}")
        print(f"{'='*60}\n")

    if num_iterations > 1 and len(all_metrics) > 0:
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY ACROSS {len(all_metrics)} ITERATIONS")
        print(f"{'='*60}")

        plccs  = [m["plcc"] for m in all_metrics]
        sroccs = [m["srocc"] for m in all_metrics]
        krccs  = [m["krcc"] for m in all_metrics]
        rmses  = [m["rmse"] for m in all_metrics]

        med_plcc,  mean_plcc,  std_plcc  = float(np.median(plccs)),  float(np.mean(plccs)),  float(np.std(plccs))
        med_srocc, mean_srocc, std_srocc = float(np.median(sroccs)), float(np.mean(sroccs)), float(np.std(sroccs))
        med_krcc,  mean_krcc,  std_krcc  = float(np.median(krccs)),  float(np.mean(krccs)),  float(np.std(krccs))
        med_rmse,  mean_rmse,  std_rmse  = float(np.median(rmses)),  float(np.mean(rmses)),  float(np.std(rmses))

        print(f"  Metric       Median       Mean        Std")
        print(f"  ---------------------------------------------")
        print(f"  SROCC:       {med_srocc:.4f}       {mean_srocc:.4f}      (±{std_srocc:.4f})")
        print(f"  PLCC:        {med_plcc:.4f}       {mean_plcc:.4f}      (±{std_plcc:.4f})  [4-param logistic]")
        print(f"  KRCC:        {med_krcc:.4f}       {mean_krcc:.4f}      (±{std_krcc:.4f})")
        print(f"  RMSE:        {med_rmse:.4f}       {mean_rmse:.4f}      (±{std_rmse:.4f})  [4-param logistic]")
        print(f"{'='*60}\n")

        summary_data = {
            "num_iterations": len(all_metrics),
            "median": {"srocc": med_srocc, "plcc": med_plcc, "krcc": med_krcc, "rmse": med_rmse},
            "mean":   {"srocc": mean_srocc, "plcc": mean_plcc, "krcc": mean_krcc, "rmse": mean_rmse},
            "std":    {"srocc": std_srocc, "plcc": std_plcc, "krcc": std_krcc, "rmse": std_rmse},
            "all_iterations": all_metrics,
        }

        # ── Top-10 Summary (when running > 10 iterations, e.g. 15) ──────────
        if len(all_metrics) > 10:
            top_k = 10
            sorted_by_srocc = sorted(all_metrics, key=lambda m: m["srocc"], reverse=True)
            top_metrics = sorted_by_srocc[:top_k]

            t_plccs  = [m["plcc"] for m in top_metrics]
            t_sroccs = [m["srocc"] for m in top_metrics]
            t_krccs  = [m["krcc"] for m in top_metrics]
            t_rmses  = [m["rmse"] for m in top_metrics]

            t_med_plcc,  t_mean_plcc,  t_std_plcc  = float(np.median(t_plccs)),  float(np.mean(t_plccs)),  float(np.std(t_plccs))
            t_med_srocc, t_mean_srocc, t_std_srocc = float(np.median(t_sroccs)), float(np.mean(t_sroccs)), float(np.std(t_sroccs))
            t_med_krcc,  t_mean_krcc,  t_std_krcc  = float(np.median(t_krccs)),  float(np.mean(t_krccs)),  float(np.std(t_krccs))
            t_med_rmse,  t_mean_rmse,  t_std_rmse  = float(np.median(t_rmses)),  float(np.mean(t_rmses)),  float(np.std(t_rmses))

            print(f"{'='*60}")
            print(f"TOP {top_k} BEST ITERATIONS SUMMARY (Ranked by SROCC out of {len(all_metrics)})")
            print(f"{'='*60}")
            print(f"  Metric       Median       Mean        Std")
            print(f"  ---------------------------------------------")
            print(f"  SROCC:       {t_med_srocc:.4f}       {t_mean_srocc:.4f}      (±{t_std_srocc:.4f})")
            print(f"  PLCC:        {t_med_plcc:.4f}       {t_mean_plcc:.4f}      (±{t_std_plcc:.4f})  [4-param logistic]")
            print(f"  KRCC:        {t_med_krcc:.4f}       {t_mean_krcc:.4f}      (±{t_std_krcc:.4f})")
            print(f"  RMSE:        {t_med_rmse:.4f}       {t_mean_rmse:.4f}      (±{t_std_rmse:.4f})  [4-param logistic]")
            print(f"{'='*60}\n")

            summary_data["top_10"] = {
                "median": {"srocc": t_med_srocc, "plcc": t_med_plcc, "krcc": t_med_krcc, "rmse": t_med_rmse},
                "mean":   {"srocc": t_mean_srocc, "plcc": t_mean_plcc, "krcc": t_mean_krcc, "rmse": t_mean_rmse},
                "std":    {"srocc": t_std_srocc, "plcc": t_std_plcc, "krcc": t_std_krcc, "rmse": t_std_rmse},
                "top_iterations": top_metrics,
            }

        summary_dir = get_run_output_dir(project_root, dataset_name, mode)
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, f"{num_iterations}_iterations_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Full benchmark summary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
