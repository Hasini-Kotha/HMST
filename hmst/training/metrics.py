"""
training/metrics.py
-------------------
Evaluation metrics for UIQA:
  - PLCC  (Pearson Linear Correlation Coefficient)
  - SROCC (Spearman Rank-Order Correlation Coefficient)
  - KRCC  (Kendall Rank Correlation Coefficient)
  - RMSE  (Root Mean Square Error)

From §9.1–§9.3 of Underwater_ideas.pdf.
"""

import numpy as np
from scipy import stats
from typing import Dict
import torch


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all four UIQA evaluation metrics.

    Args:
        predictions: [N]  model predicted quality scores
        targets:     [N]  ground truth MOS scores

    Returns:
        dict with keys: plcc, srocc, krcc, rmse
    """
    # Handle edge cases
    if len(predictions) < 2:
        return {"plcc": 0.0, "srocc": 0.0, "krcc": 0.0, "rmse": float("inf")}

    # PLCC
    plcc, _ = stats.pearsonr(predictions, targets)

    # SROCC
    srocc, _ = stats.spearmanr(predictions, targets)

    # KRCC
    krcc, _ = stats.kendalltau(predictions, targets)

    # RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return {
        "plcc":  float(plcc),
        "srocc": float(srocc),
        "krcc":  float(krcc),
        "rmse":  float(rmse),
    }


def collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a dataloader and collect all predictions.

    Returns:
        dict with 'predictions' and 'targets' as numpy arrays
    """
    model.eval()
    preds_list = []
    tgts_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            scores = batch["score"].numpy()

            output = model(images)
            q_hat = output["Q_hat"].cpu().numpy()

            preds_list.append(q_hat)
            tgts_list.append(scores)

    return {
        "predictions": np.concatenate(preds_list),
        "targets":     np.concatenate(tgts_list),
    }
