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
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
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

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore random states for reproducibility
    if "random_state" in checkpoint:
        rs = checkpoint["random_state"]
        random.setstate(rs["python"])
        np.random.set_state(rs["numpy"])
        torch.set_rng_state(rs["torch"])
        if torch.cuda.is_available() and "cuda" in rs:
            torch.cuda.set_rng_state_all(rs["cuda"])

    return checkpoint
