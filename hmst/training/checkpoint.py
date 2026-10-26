"""
training/checkpoint.py
-----------------------
Robust checkpoint save/load for HMST training.

Checkpoint format:
{
    "epoch":              int,
    "model_state_dict":   OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "best_metric":        float,
    "training_history":   list of dicts,
    "config":             dict,
    "random_state":       dict,
}
"""

import os
import random
import torch
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Dict, Any


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    training_history: list,
    config: dict,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    trainer_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path:             Full path to save the .pt file
        epoch:            Current epoch (0-based)
        model:            HMST model
        optimizer:        Optimizer
        best_metric:      Best validation metric so far
        training_history: List of per-epoch metric dicts
        config:           Configuration dict
    """
    # Capture random states for reproducibility
    random_state = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state["cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric":          best_metric,
        "training_history":     training_history,
        "config":               config,
        "random_state":         random_state,
        "trainer_state":        trainer_state or {},
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    expected_dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model/optimizer state.

    Args:
        path:      Path to the .pt checkpoint file
        model:     Model to load state into
        optimizer: Optimizer to restore (optional)
        device:    Device to map tensors to

    Returns:
        The checkpoint dict (for epoch, history, etc.)
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Validate backbone compatibility
    ckpt_bb = checkpoint.get("config", {}).get("backbone", "resnet50")
    if ckpt_bb:
        ckpt_bb = ckpt_bb.lower().strip().replace("-", "_")
        if ckpt_bb == "convnext_tiny":
            ckpt_bb = "convnext"
    model_bb = getattr(model, "backbone_name", None)
    if model_bb is not None and ckpt_bb is not None and model_bb != ckpt_bb:
        raise RuntimeError(
            f"Backbone mismatch: Checkpoint '{path}' was trained with backbone '{ckpt_bb}', "
            f"but current model uses backbone '{model_bb}'. "
            f"Cannot cross-load checkpoints between different backbones."
        )

    if expected_dataset is not None:
        checkpoint_dataset = checkpoint.get("config", {}).get("dataset")
        if checkpoint_dataset != expected_dataset:
            raise RuntimeError(
                f"Dataset mismatch: Checkpoint '{path}' was trained with dataset "
                f"'{checkpoint_dataset}', but the current run uses '{expected_dataset}'. "
                "Refusing to resume across datasets."
            )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if optimizer is not None and scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            warnings.warn(
                f"Checkpoint '{path}' has no scheduler state; resuming with a fresh scheduler.",
                RuntimeWarning,
            )
        else:
            scheduler.load_state_dict(scheduler_state)
    if optimizer is not None and scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)

    # Restore random states for reproducibility.
    # Skip when loading for inference (optimizer=None) — only model weights matter.
    if optimizer is not None and "random_state" in checkpoint:
        rs = checkpoint["random_state"]
        try:
            random.setstate(rs["python"])
        except Exception:
            pass
        try:
            np.random.set_state(rs["numpy"])
        except Exception:
            pass
        try:
            # Cast to CPU ByteTensor — guards against dtype/device mismatch
            # that occurs when checkpoints are moved across machines or
            # PyTorch versions (e.g., saved on one CUDA version, loaded on another).
            torch_state = rs["torch"]
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch_state.cpu().to(torch.uint8)
            torch.set_rng_state(torch_state)
        except Exception:
            pass
        try:
            if torch.cuda.is_available() and "cuda" in rs:
                cuda_states = rs["cuda"]
                current_states = torch.cuda.get_rng_state_all()
                # Only restore if the number of GPUs matches
                if len(cuda_states) == len(current_states):
                    torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass

    return checkpoint
