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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
import yaml

from scripts.train import load_config, build_datasets, get_device, make_absolute, set_seed
from models import HMST, build_correspondence_matrix
from models.losses import HMSTLoss
from training.checkpoint import load_checkpoint
from training.metrics import compute_metrics, collect_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMST Architecture 1")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--mode", type=str, choices=["test", "full"], default="full",
                        help="Config mode to use for dataset settings")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["uwiqa", "saud"])
    args = parser.parse_args()

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    config = load_config(args.mode)
    if args.dataset:
        config["dataset"] = args.dataset

    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(config)

    print(f"\nEvaluating HMST Architecture 1")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")

    # Dataset
    _, val_ds, test_ds = build_datasets(config, project_root)
    batch_size = config.get("batch_size", 4)
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
        image_size=image_size,
        patch_sizes=patch_sizes,
        d_model=512, nhead=8, nlayers=2, n_regions=4,
    ).to(device)

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint, model, device=device)
    trained_epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {trained_epoch}")

    # Evaluate on val and test
    for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
        result = collect_predictions(model, loader, device)
        metrics = compute_metrics(result["predictions"], result["targets"])
        print(f"\n{split_name} Metrics (n={len(result['predictions'])}):")
        print(f"  PLCC:  {metrics['plcc']:.4f}")
        print(f"  SROCC: {metrics['srocc']:.4f}")
        print(f"  KRCC:  {metrics['krcc']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")

        out_dir = os.path.join(project_root, "outputs", args.mode)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{split_name.lower()}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
