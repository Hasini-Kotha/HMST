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
from scipy.optimize import curve_fit
from typing import Dict, Optional
import torch


class NonLinearFit:
    """
    Non-linear regression fitting for IQA (4-parameter logistic mapping),
    as recommended by VQEG and ITU-R BT.500 standards.
    """
    def __init__(self, x=None, y=None):
        self.maxfev = 10000
        self.nlinfunc = self.four_param_logistic_QA
        self.params = None
        if x is not None and (hasattr(x, "size") and x.size > 0 or hasattr(x, "__len__") and len(x) > 0):
            self.fit(x, y)

    def fit(self, x, y):
        """
        Fits 4-parameter logistic mapping from model predictions (y) to ground truth MOS (x).
        Args:
            x: Ground truth MOS (targets)
            y: Model predictions
        """
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        beta_init = [float(np.min(x)), float(np.max(x)), float(np.mean(y)), 1.0]
        try:
            self.params, self.params_covariance = curve_fit(
                self.nlinfunc, y, x, p0=beta_init, maxfev=self.maxfev
            )
        except Exception:
            # Fallback parameters if curve_fit fails to converge
            self.params = beta_init

    def transform(self, y):
        """
        Maps raw predictions y through the fitted 4-parameter logistic function.
        """
        if self.params is None:
            return y
        y = np.asarray(y, dtype=np.float64).flatten()
        try:
            fitted_predictions = self.nlinfunc(y, *self.params)
            if np.all(np.isfinite(fitted_predictions)):
                return fitted_predictions
            return y
        except Exception:
            return y

    @staticmethod
    def four_param_logistic_QA(x, beta1, beta2, beta3, beta4):
        return ((beta1 - beta2) / (1.0 + np.exp((beta3 - x) / (np.abs(beta4) + 1e-8)))) + beta2

    @staticmethod
    def exponential(x, beta1, beta2, beta3):
        return beta1 + beta2 * np.exp(beta3 * x)

    @staticmethod
    def affine(x, beta1, beta2):
        return beta1 + beta2 * x


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    use_nonlinear_fit: bool = True,
) -> Dict[str, float]:
    """
    Compute UIQA evaluation metrics with 4-parameter logistic regression mapping:
      - PLCC  (Pearson Linear Correlation Coefficient, after non-linear mapping)
      - SROCC (Spearman Rank-Order Correlation Coefficient)
      - KRCC  (Kendall Rank Correlation Coefficient)
      - RMSE  (Root Mean Square Error, after non-linear mapping)
      - raw_plcc & raw_rmse (before non-linear mapping)

    Args:
        predictions: [N]  model predicted quality scores
        targets:     [N]  ground truth MOS scores
        use_nonlinear_fit: whether to apply 4-parameter logistic fitting (default: True)

    Returns:
        dict with keys: plcc, srocc, krcc, rmse, raw_plcc, raw_rmse
    """
    # Handle edge cases
    if len(predictions) < 2:
        return {
            "plcc": 0.0,
            "srocc": 0.0,
            "krcc": 0.0,
            "rmse": float("inf"),
            "raw_plcc": 0.0,
            "raw_rmse": float("inf"),
        }

    predictions = np.asarray(predictions, dtype=np.float64).flatten()
    targets = np.asarray(targets, dtype=np.float64).flatten()

    # Raw metrics
    raw_plcc, _ = stats.pearsonr(predictions, targets)
    srocc, _ = stats.spearmanr(predictions, targets)
    krcc, _ = stats.kendalltau(predictions, targets)
    raw_rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))

    # 4-parameter logistic non-linear fitting (VQEG / ITU-R BT.500 protocol)
    if use_nonlinear_fit and len(predictions) >= 4:
        fitter = NonLinearFit()
        fitter.fit(x=targets, y=predictions)
        fitted_preds = fitter.transform(predictions)

        fitted_plcc, _ = stats.pearsonr(fitted_preds, targets)
        fitted_rmse = float(np.sqrt(np.mean((fitted_preds - targets) ** 2)))

        return {
            "plcc":     float(fitted_plcc),
            "srocc":    float(srocc),
            "krcc":     float(krcc),
            "rmse":     float(fitted_rmse),
            "raw_plcc": float(raw_plcc),
            "raw_rmse": float(raw_rmse),
        }

    return {
        "plcc":     float(raw_plcc),
        "srocc":    float(srocc),
        "krcc":     float(krcc),
        "rmse":     float(raw_rmse),
        "raw_plcc": float(raw_plcc),
        "raw_rmse": float(raw_rmse),
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
