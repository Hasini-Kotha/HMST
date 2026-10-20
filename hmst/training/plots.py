"""Correlation and error-analysis plots for image-quality predictions."""

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


def _finite_pairs(actual: Iterable[float], predicted: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have equal lengths.")
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[finite]
    predicted_values = predicted_values[finite]
    if actual_values.size == 0:
        raise ValueError("Cannot plot empty or non-finite prediction arrays.")
    return actual_values, predicted_values


def _four_parameter_logistic(x: np.ndarray, upper: float, lower: float,
                             midpoint: float, slope: float) -> np.ndarray:
    width = max(abs(slope), 1e-8)
    exponent = np.clip((midpoint - x) / width, -500.0, 500.0)
    return (upper - lower) / (1.0 + np.exp(exponent)) + lower


def _fit_logistic(actual: np.ndarray, predicted: np.ndarray):
    from scipy.optimize import OptimizeWarning, curve_fit
    import warnings

    if actual.size < 4 or np.unique(actual).size < 4:
        return None

    x_span = max(float(np.ptp(actual)), 1e-3)
    initial = [float(np.max(predicted)), float(np.min(predicted)),
               float(np.median(actual)), x_span / 4.0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            parameters, _ = curve_fit(
                _four_parameter_logistic,
                actual,
                predicted,
                p0=initial,
                maxfev=20000,
            )
        if not np.all(np.isfinite(parameters)):
            return None
        return parameters
    except (RuntimeError, ValueError, OptimizeWarning, FloatingPointError):
        return None


def _axis_limits(series: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    values = np.concatenate([np.concatenate(pair) for pair in series])
    low = float(np.min(values))
    high = float(np.max(values))
    padding = max((high - low) * 0.05, 0.02)
    return low - padding, high + padding


def _add_identity_line(axis, low: float, high: float) -> None:
    axis.plot([low, high], [low, high], linestyle="--", color="#555555",
              linewidth=1.2, label="Identity (y = x)")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Actual MOS")
    axis.set_ylabel("Predicted MOS (TTA)")
    axis.grid(True, alpha=0.2)


def _annotate_metrics(axis, metrics: Dict[str, float]) -> None:
    text = (
        f"PLCC  {metrics['plcc']:.4f}\n"
        f"SROCC {metrics['srocc']:.4f}\n"
        f"KRCC  {metrics['krcc']:.4f}\n"
        f"RMSE  {metrics['rmse']:.4f}"
    )
    axis.text(
        0.04, 0.96, text, transform=axis.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )


def _lowess(x: np.ndarray, y: np.ndarray, fraction: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Return a local-linear LOWESS curve without an extra dependency."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    count = len(x_sorted)
    neighbors = min(count, max(3, int(np.ceil(fraction * count))))
    fitted = np.empty(count, dtype=np.float64)

    for index, center in enumerate(x_sorted):
        distances = np.abs(x_sorted - center)
        radius = np.partition(distances, neighbors - 1)[neighbors - 1]
        if radius <= 0:
            weights = (distances == 0).astype(np.float64)
        else:
            scaled = np.minimum(distances / radius, 1.0)
            weights = (1.0 - scaled ** 3) ** 3
        design = np.column_stack((np.ones(count), x_sorted - center))
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_values = y_sorted * np.sqrt(weights)
        fitted[index] = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0][0]

    return x_sorted, fitted


def _finite_triplets(
    actual: Iterable[float],
    predicted_tta: Iterable[float],
    predicted_no_tta: Iterable[float],
    image_paths: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    tta_values = np.asarray(predicted_tta, dtype=np.float64).reshape(-1)
    no_tta_values = np.asarray(predicted_no_tta, dtype=np.float64).reshape(-1)
    paths = tuple(image_paths)
    if actual_values.shape != tta_values.shape or actual_values.shape != no_tta_values.shape:
        raise ValueError("Actual and both predicted arrays must have equal lengths.")
    if paths and len(paths) != len(actual_values):
        raise ValueError("Image paths must match the prediction array length.")
    finite = np.isfinite(actual_values) & np.isfinite(tta_values) & np.isfinite(no_tta_values)
    actual_values = actual_values[finite]
    tta_values = tta_values[finite]
    no_tta_values = no_tta_values[finite]
    if paths:
        paths = tuple(path for path, keep in zip(paths, finite) if keep)
    if actual_values.size == 0:
        raise ValueError("Cannot plot empty or non-finite prediction arrays.")
    return actual_values, tta_values, no_tta_values, paths


def save_error_analysis_plots(
    actual: Iterable[float],
    predicted_tta: Iterable[float],
    predicted_no_tta: Iterable[float],
    image_paths: Iterable[str],
    output_dir: str,
    pareto_actual: Iterable[float] = None,
    pareto_predicted: Iterable[float] = None,
    pareto_image_paths: Iterable[str] = None,
    top_n: int = 8,
) -> Tuple[Path, ...]:
    """Save B5-B9 error-analysis plots and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.stats import gaussian_kde

    actual, predicted_tta, predicted_no_tta, image_paths = _finite_triplets(
        actual, predicted_tta, predicted_no_tta, image_paths
    )
    if pareto_actual is None:
        pareto_actual = actual
        pareto_predicted = predicted_tta
        pareto_image_paths = image_paths
    pareto_actual, pareto_predicted = _finite_pairs(pareto_actual, pareto_predicted)
    if pareto_image_paths is not None and len(tuple(pareto_image_paths)) != len(pareto_actual):
        raise ValueError("Pareto image paths must match the Pareto prediction array length.")

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    residual = predicted_tta - actual

    # B5: Residuals against actual MOS with a local-linear LOWESS trend.
    fig, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    axis.scatter(actual, residual, s=25, alpha=0.65, color="#176B87",
                 edgecolors="white", linewidths=0.3)
    axis.axhline(0, color="#555555", linestyle="--", linewidth=1.2)
    lowess_x, lowess_y = _lowess(actual, residual)
    axis.plot(lowess_x, lowess_y, color="#D55E00", linewidth=2, label="LOWESS")
    axis.set_xlabel("Actual MOS")
    axis.set_ylabel("Residual (Predicted - Actual)")
    axis.set_title("Test Residuals vs Actual MOS (TTA)")
    axis.grid(True, alpha=0.2)
    axis.legend()
    path = plot_dir / "B5_residual_vs_actual_mos.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # B6: Compare TTA and no-TTA residual distributions.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, values, label, color in (
        (axes[0], predicted_tta - actual, "TTA", "#009E73"),
        (axes[1], predicted_no_tta - actual, "No-TTA", "#D55E00"),
    ):
        axis.hist(values, bins="auto", density=True, alpha=0.45, color=color,
                  edgecolor="white")
        if values.size > 1 and float(np.std(values)) > 0:
            density_x = np.linspace(float(np.min(values)), float(np.max(values)), 250)
            try:
                axis.plot(density_x, gaussian_kde(values)(density_x), color=color, linewidth=2)
            except (ValueError, np.linalg.LinAlgError):
                pass
        mean = float(np.mean(values))
        median = float(np.median(values))
        axis.axvline(mean, color="#0072B2", linestyle="--", linewidth=1.5, label="Mean")
        axis.axvline(median, color="#CC79A7", linestyle=":", linewidth=1.8, label="Median")
        axis.text(
            0.97, 0.96,
            f"Mean: {mean:.4f}\nMedian: {median:.4f}\nStd Dev: {np.std(values):.4f}",
            transform=axis.transAxes, va="top", ha="right", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )
        axis.set_title(label)
        axis.set_xlabel("Residual (Predicted - Actual)")
        axis.set_ylabel("Density")
        axis.grid(True, alpha=0.2)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    fig.suptitle("Residual Distribution: TTA vs No-TTA")
    path = plot_dir / "B6_error_histogram_kde_tta_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # B7: Absolute error distributions with box plots and deterministic jitter.
    absolute_errors = [np.abs(predicted_tta - actual), np.abs(predicted_no_tta - actual)]
    fig, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    violin = axis.violinplot(absolute_errors, positions=[1, 2], showextrema=False)
    for body, color in zip(violin["bodies"], ("#009E73", "#D55E00")):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    axis.boxplot(absolute_errors, positions=[1, 2], widths=0.18, showfliers=False,
                 medianprops={"color": "#222222", "linewidth": 1.5})
    jitter = np.random.default_rng(42)
    for position, values, color in zip((1, 2), absolute_errors, ("#009E73", "#D55E00")):
        axis.scatter(position + jitter.uniform(-0.08, 0.08, size=len(values)), values,
                     s=15, alpha=0.5, color=color, edgecolors="none")
    axis.set_xticks([1, 2], ["TTA", "No-TTA"])
    axis.set_ylabel("Absolute Error")
    axis.set_title("Absolute Error by Inference Method")
    axis.grid(axis="y", alpha=0.2)
    path = plot_dir / "B7_absolute_error_violin_box.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # B8: Best and worst predictions with source-image thumbnails.
    count = len(actual)
    shown = min(max(1, top_n), count)
    error_order = np.argsort(np.abs(residual))
    selected = (error_order[:shown], error_order[-shown:][::-1])
    fig, axes = plt.subplots(2, shown, figsize=(shown * 2.2, 5.6), squeeze=False,
                             constrained_layout=True)
    for row, (indices, color, title) in enumerate(zip(
        selected, ("#009E73", "#D55E00"), ("Best", "Worst")
    )):
        for column, index in enumerate(indices):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            if image_paths:
                try:
                    with Image.open(image_paths[index]) as image:
                        axis.imshow(image.convert("RGB"))
                except (OSError, ValueError):
                    axis.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            else:
                axis.text(0.5, 0.5, "Image path unavailable", ha="center", va="center")
            image_id = Path(image_paths[index]).name if image_paths else f"image_{index}"
            axis.set_xlabel(
                f"[{image_id} | actual_MOS={actual[index]:.3f} "
                f"| pred_MOS={predicted_tta[index]:.3f} | error={residual[index]:+.3f}]",
                fontsize=7,
            )
            if column == 0:
                axis.set_ylabel(title, color=color, fontsize=10)
    fig.suptitle(f"Best {shown} and Worst {shown} Test Predictions (TTA)")
    path = plot_dir / "B8_best_worst_prediction_images.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # B9: Sorted absolute errors with cumulative share of total absolute error.
    pareto_errors = np.abs(pareto_predicted - pareto_actual)
    order = np.argsort(pareto_errors)[::-1]
    sorted_errors = pareto_errors[order]
    cumulative = np.cumsum(sorted_errors)
    total_error = float(cumulative[-1])
    cumulative_percent = cumulative / total_error * 100 if total_error > 0 else np.zeros_like(cumulative)
    positions = np.arange(len(sorted_errors))
    colors = plt.get_cmap("RdYlGn_r")(np.linspace(1, 0, len(sorted_errors)))
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axis.barh(positions, sorted_errors, color=colors, height=1.0)
    axis.invert_yaxis()
    axis.set_yticks([])
    axis.set_xlabel("Absolute Error")
    axis.set_ylabel(f"Images sorted by absolute error (n={len(sorted_errors)})")
    axis.set_title("Per-Image Absolute Error Pareto Analysis")
    axis.grid(axis="x", alpha=0.2)
    cumulative_axis = axis.twiny()
    cumulative_axis.plot(cumulative_percent, positions, color="#222222", linewidth=1.5)
    cumulative_axis.set_xlim(0, 100)
    cumulative_axis.set_xlabel("Cumulative share of total absolute error (%)")
    path = plot_dir / "B9_per_image_error_pareto.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return tuple(saved_paths)


def save_quality_plots(
    actual_test: Iterable[float],
    predicted_test_tta: Iterable[float],
    test_metrics: Dict[str, float],
    actual_validation: Iterable[float],
    predicted_validation_tta: Iterable[float],
    validation_metrics: Dict[str, float],
    output_dir: str,
) -> Tuple[Path, ...]:
    """Save the four TTA correlation plots and return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    actual_test, predicted_test_tta = _finite_pairs(actual_test, predicted_test_tta)
    actual_validation, predicted_validation_tta = _finite_pairs(
        actual_validation, predicted_validation_tta
    )
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    limits = _axis_limits([
        (actual_test, predicted_test_tta),
        (actual_validation, predicted_validation_tta),
    ])
    fit_x = np.linspace(limits[0], limits[1], 300)
    test_fit = _fit_logistic(actual_test, predicted_test_tta)
    validation_fit = _fit_logistic(actual_validation, predicted_validation_tta)
    saved_paths = []

    # A1: Primary test-set correlation plot with fitted trend and metric box.
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter(actual_test, predicted_test_tta, s=26, alpha=0.72,
                 color="#176B87", edgecolors="white", linewidths=0.35,
                 label="Test images")
    _add_identity_line(axis, *limits)
    if test_fit is not None:
        axis.plot(fit_x, _four_parameter_logistic(fit_x, *test_fit),
                  color="#D95F02", linewidth=2, label="4-parameter logistic fit")
    _annotate_metrics(axis, test_metrics)
    axis.set_title("Test: Predicted vs Actual MOS (TTA)")
    axis.legend(loc="lower right", frameon=True)
    path = plot_dir / "A1_test_predicted_vs_actual_tta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # A2: Same test observations and fixed scales, with and without fitted trend.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for axis in axes:
        axis.scatter(actual_test, predicted_test_tta, s=24, alpha=0.72,
                     color="#176B87", edgecolors="white", linewidths=0.35)
        _add_identity_line(axis, *limits)
    if test_fit is not None:
        axes[1].plot(fit_x, _four_parameter_logistic(fit_x, *test_fit),
                     color="#D95F02", linewidth=2, label="4-parameter logistic fit")
        axes[1].legend(loc="lower right")
    axes[0].set_title("Raw predictions")
    axes[1].set_title("With 4-parameter logistic fit")
    axes[0].set_ylabel("Predicted MOS (TTA)")
    fig.suptitle("Test Set: Raw vs Fitted Predictions")
    path = plot_dir / "A2_test_raw_vs_fitted.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # A3: Compare validation and test distributions with independent fits.
    fig, axis = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    series = [
        (actual_validation, predicted_validation_tta, validation_fit,
         "#009E73", "Validation", validation_metrics),
        (actual_test, predicted_test_tta, test_fit,
         "#D55E00", "Test", test_metrics),
    ]
    _add_identity_line(axis, *limits)
    for actual, predicted, parameters, color, label, metrics in series:
        axis.scatter(actual, predicted, s=24, alpha=0.62, color=color,
                     edgecolors="white", linewidths=0.3,
                     label=f"{label} (SROCC={metrics['srocc']:.3f})")
        if parameters is not None:
            axis.plot(fit_x, _four_parameter_logistic(fit_x, *parameters),
                      color=color, linewidth=2, linestyle="-",
                      label=f"{label} logistic fit")
    axis.set_title("Validation vs Test (TTA)")
    axis.legend(loc="best", frameon=True)
    path = plot_dir / "A3_validation_vs_test_tta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # A4: Per-image absolute errors; cool blue indicates low error, red high.
    absolute_error = np.abs(predicted_test_tta - actual_test)
    maximum_error = max(float(np.max(absolute_error)), 1e-8)
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = axis.scatter(
        actual_test, predicted_test_tta, c=absolute_error, cmap="coolwarm",
        norm=Normalize(vmin=0.0, vmax=maximum_error), s=30, alpha=0.85,
        edgecolors="white", linewidths=0.35,
    )
    _add_identity_line(axis, *limits)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Absolute prediction error")
    axis.set_title("Test Predictions Colored by Absolute Error")
    path = plot_dir / "A4_test_error_magnitude.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return tuple(saved_paths)


def save_training_dynamics_plots(
    history: Iterable[Dict[str, float]],
    output_dir: str,
    monitor: str = "srocc",
    mode: str = "max",
) -> Tuple[Path, ...]:
    """Save training loss, metric, and learning-rate curves with convergence markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = list(history)
    if not records:
        return ()

    epochs = np.asarray([record.get("epoch", index + 1)
                         for index, record in enumerate(records)], dtype=np.float64)
    monitor_name = monitor.removeprefix("val_")
    monitor_key = "val_loss" if monitor_name == "loss" else f"val_{monitor_name}"
    monitor_values = np.asarray([record.get(monitor_key, np.nan) for record in records],
                                dtype=np.float64)
    finite_monitor = np.isfinite(monitor_values)
    if np.any(finite_monitor):
        candidate_indices = np.flatnonzero(finite_monitor)
        candidate_values = monitor_values[finite_monitor]
        best_index = candidate_indices[
            int(np.argmax(candidate_values) if mode == "max" else np.argmin(candidate_values))
        ]
        best_epoch = epochs[best_index]
        best_value = monitor_values[best_index]
    else:
        best_index = len(records) - 1
        best_epoch = epochs[best_index]
        best_value = float("nan")

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    final_epoch = float(epochs[-1])
    convergence_label = f"Best val {monitor_name.upper()}: {best_value:.4f} at epoch {best_epoch:g}"

    def mark_convergence(axis) -> None:
        axis.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2,
                     label="Best validation epoch")
        if final_epoch > best_epoch:
            axis.axvspan(best_epoch, final_epoch, color="#999999", alpha=0.12,
                         label="Training beyond best epoch")
        axis.text(
            0.99, 0.98, convergence_label,
            transform=axis.transAxes, va="top", ha="right", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    def plot_history_line(axis, key: str, label: str, style: str, color: str) -> None:
        values = np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64)
        finite = np.isfinite(values)
        if np.any(finite):
            axis.plot(epochs[finite], values[finite], linestyle=style, color=color,
                      marker="o", markersize=3, label=label)

    # C10: Total and component losses, with solid train and dashed validation lines.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    loss_panels = (
        ("Loss", "train_loss", "val_loss"),
        ("MOS Loss", "train_loss_mos", "val_loss_mos"),
        ("Ranking Loss", "train_loss_rank", "val_loss_rank"),
        ("Consistency Loss", "train_loss_cons", "val_loss_cons"),
    )
    for axis, (title, train_key, val_key) in zip(axes.flat, loss_panels):
        plot_history_line(axis, train_key, "Train", "-", "#176B87")
        plot_history_line(axis, val_key, "Validation", "--", "#D55E00")
        mark_convergence(axis)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle("Training and Validation Losses")
    path = plot_dir / "C10_loss_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # C11: Train and validation metrics, with a star at the selected best epoch.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    metric_names = ("PLCC", "SROCC", "KRCC", "RMSE")
    for axis, metric in zip(axes.flat, metric_names):
        key = metric.lower()
        plot_history_line(axis, f"train_{key}", "Train", "-", "#176B87")
        plot_history_line(axis, f"val_{key}", "Validation", "--", "#D55E00")
        val_values = np.asarray([record.get(f"val_{key}", np.nan) for record in records],
                                dtype=np.float64)
        if best_index < len(val_values) and np.isfinite(val_values[best_index]):
            axis.scatter([best_epoch], [val_values[best_index]], marker="*", s=110,
                         color="#E69F00", edgecolors="#333333", linewidths=0.5,
                         label="Selected best epoch", zorder=5)
        mark_convergence(axis)
        axis.set_title(metric)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric value")
        axis.grid(True, alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    fig.suptitle("Training and Validation Metrics")
    path = plot_dir / "C11_metric_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # C12: Learning rate used during each epoch, with scheduler changes marked.
    learning_rates = np.asarray([record.get("learning_rate", np.nan) for record in records],
                                dtype=np.float64)
    valid_lr = np.isfinite(learning_rates) & (learning_rates > 0)
    if np.any(valid_lr):
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(epochs[valid_lr], learning_rates[valid_lr], color="#009E73",
                  marker="o", linewidth=1.8, markersize=4, label="Learning rate")
        change_indices = np.flatnonzero(
            valid_lr[1:] & valid_lr[:-1]
            & ~np.isclose(learning_rates[1:], learning_rates[:-1], rtol=1e-8, atol=0.0)
        ) + 1
        for change_number, index in enumerate(change_indices):
            axis.axvline(epochs[index], color="#D55E00", linestyle=":", linewidth=1.2,
                         label="Scheduler change" if change_number == 0 else None)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Learning rate")
        axis.set_title("Learning Rate Schedule")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend(loc="best")
        path = plot_dir / "C12_learning_rate_schedule.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_paths.append(path)
    else:
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.text(0.5, 0.5, "Learning-rate history unavailable for this run",
                  ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Learning rate")
        axis.set_title("Learning Rate Schedule")
        axis.set_axis_off()
        path = plot_dir / "C12_learning_rate_schedule.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_paths.append(path)

    # C13: Dedicated view of the selected early-stopping/convergence epoch.
    convergence_values = np.asarray(
        [record.get(monitor_key, np.nan) for record in records], dtype=np.float64
    )
    convergence_finite = np.isfinite(convergence_values)
    if np.any(convergence_finite):
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(epochs[convergence_finite], convergence_values[convergence_finite],
                  color="#176B87", marker="o", linewidth=1.8, label=f"Validation {monitor_name.upper()}")
        axis.axvline(best_epoch, color="#D55E00", linestyle="--", linewidth=1.5,
                     label="Best validation epoch")
        if final_epoch > best_epoch:
            axis.axvspan(best_epoch, final_epoch, color="#999999", alpha=0.18,
                         label="Training beyond best epoch")
        axis.scatter([best_epoch], [best_value], marker="*", s=150,
                     color="#E69F00", edgecolors="#333333", linewidths=0.6,
                     label="Best validation value", zorder=5)
        axis.text(
            0.98, 0.96,
            f"Early stop epoch: {best_epoch:g}\nBest val {monitor_name.upper()}: {best_value:.4f}",
            transform=axis.transAxes, va="top", ha="right", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(f"Validation {monitor_name.upper()}")
        axis.set_title("Early Stopping and Convergence")
        axis.grid(True, alpha=0.2)
        axis.legend(loc="best", fontsize=8)
        path = plot_dir / "C13_early_stopping_convergence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_paths.append(path)

    return tuple(saved_paths)


def save_split_distribution_plots(
    train_dataset,
    validation_dataset,
    test_dataset,
    output_dir: str,
    mos_range: Tuple[float, float] = (0.2, 1.0),
) -> Tuple[Path, ...]:
    """Save MOS distribution, split-count, and KS comparison plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, ks_2samp

    datasets = (
        ("Train", train_dataset, "#0072B2"),
        ("Validation", validation_dataset, "#D55E00"),
        ("Test", test_dataset, "#009E73"),
    )
    split_scores = []
    for label, dataset, color in datasets:
        samples = getattr(dataset, "samples", None)
        if samples is None:
            raise ValueError(f"{label} dataset does not expose split samples.")
        scores = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        split_scores.append((label, scores, color))

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    bin_edges = np.linspace(mos_range[0], mos_range[1], 41)

    # F22: Overlaid MOS histograms and per-split distribution statistics.
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    stats_lines = []
    for label, scores, color in split_scores:
        axis.hist(scores, bins=bin_edges, alpha=0.42, color=color,
                  edgecolor=color, label=f"{label} (n={len(scores)})")
        if scores.size:
            stats_lines.append(
                f"{label}: n={len(scores)}, mean={np.mean(scores):.3f}, std={np.std(scores):.3f}"
            )
        else:
            stats_lines.append(f"{label}: n=0, mean=n/a, std=n/a")
    axis.text(
        0.98, 0.97, "\n".join(stats_lines), transform=axis.transAxes,
        ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    axis.set_xlim(*mos_range)
    axis.set_xlabel("MOS value")
    axis.set_ylabel("Count of images")
    axis.set_title("MOS Distribution by Dataset Split")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper left")
    path = plot_dir / "F22_mos_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # F23: Counts and share of the total dataset for each split.
    labels = [label for label, _, _ in split_scores]
    counts = np.asarray([len(scores) for _, scores, _ in split_scores])
    total = int(np.sum(counts))
    percentages = counts / total * 100 if total else np.zeros_like(counts, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    bars = axis.bar(labels, counts, color=[color for _, _, color in split_scores], width=0.62)
    for bar, count, percentage in zip(bars, counts, percentages):
        axis.annotate(
            f"{count} ({percentage:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
        )
    axis.set_ylabel("Count of images")
    axis.set_title("Sample Counts by Dataset Split")
    axis.set_ylim(0, max(int(np.max(counts)) * 1.15, 1))
    axis.grid(axis="y", alpha=0.2)
    path = plot_dir / "F23_split_sample_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # F24: Smooth split densities and pairwise Kolmogorov-Smirnov tests.
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x_values = np.linspace(mos_range[0], mos_range[1], 400)
    for label, scores, color in split_scores:
        if not scores.size:
            continue
        if scores.size > 1 and float(np.std(scores)) > 0:
            try:
                density = gaussian_kde(scores)(x_values)
            except (ValueError, np.linalg.LinAlgError):
                density = None
        else:
            density = None
        if density is None:
            bandwidth = max((mos_range[1] - mos_range[0]) * 0.025, 1e-3)
            scaled = (x_values[:, None] - scores[None, :]) / bandwidth
            density = np.exp(-0.5 * scaled ** 2).mean(axis=1) / (
                bandwidth * np.sqrt(2 * np.pi)
            )
        axis.plot(x_values, density, color=color, linewidth=2, label=label)

    split_map = {label: scores for label, scores, _ in split_scores}
    comparisons = (("Train", "Validation"), ("Validation", "Test"), ("Train", "Test"))
    test_lines = []
    for left, right in comparisons:
        left_scores = split_map[left]
        right_scores = split_map[right]
        if left_scores.size and right_scores.size:
            p_value = ks_2samp(left_scores, right_scores).pvalue
            test_lines.append(f"{left} vs {right}: p={p_value:.4g}")
        else:
            test_lines.append(f"{left} vs {right}: p=n/a")
    test_lines.append("p > 0.05: distributions not significantly different")
    axis.text(
        0.98, 0.97, "\n".join(test_lines), transform=axis.transAxes,
        ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    axis.set_xlim(*mos_range)
    axis.set_xlabel("MOS value")
    axis.set_ylabel("Density")
    axis.set_title("MOS Split Distributions and KS Tests")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper left")
    path = plot_dir / "F24_split_distribution_ks.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return tuple(saved_paths)


def save_model_interpretability_plots(
    data: Dict[str, Any],
    model,
    correspondence_matrix,
    output_dir: str,
) -> Tuple[Path, ...]:
    """Save scale, patch, region, and consistency visualizations from HMST outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    scale_labels = [
        f"{size}px ({grid}x{grid})"
        for size, grid in zip(model.patch_sizes, model.grid_sizes)
    ]
    scale_order = list(range(len(scale_labels) - 1, -1, -1))
    lambda_weights = np.asarray(data["lambda_weights"], dtype=np.float64)
    scale_qualities = [np.asarray(values, dtype=np.float64) for values in data["scale_qualities"]]
    scale_means = np.mean(lambda_weights, axis=0)
    scale_stds = np.std(lambda_weights, axis=0)
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # E17: Mean scale-fusion weights across the evaluated test samples.
    ordered_means = scale_means[scale_order]
    colors = plt.get_cmap("coolwarm")(
        Normalize(vmin=float(np.min(scale_means)), vmax=max(float(np.max(scale_means)), 1e-8))(
            ordered_means
        )
    )
    fig, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    bars = axis.bar([scale_labels[index] for index in scale_order], ordered_means, color=colors)
    for bar, value in zip(bars, ordered_means):
        axis.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
    axis.set_ylim(0, max(float(np.max(ordered_means)) * 1.18, 1e-3))
    axis.set_xlabel("Scale (coarse to fine)")
    axis.set_ylabel("Mean scale weight (lambda)")
    axis.set_title("HMST Scale Importance")
    axis.grid(axis="y", alpha=0.2)
    path = plot_dir / "E17_lambda_scale_importance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # E18: Per-patch alpha maps overlaid on the same de-normalized test image.
    image = np.asarray(data["reference_image"], dtype=np.float64).transpose(1, 2, 0)
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    image = np.clip(image * std + mean, 0.0, 1.0)
    height, width = image.shape[:2]
    fig, axes = plt.subplots(1, len(scale_labels), figsize=(15, 5), constrained_layout=True)
    for scale_index, axis in enumerate(axes):
        grid = model.grid_sizes[scale_index]
        patch_size = model.patch_sizes[scale_index]
        alpha_map = np.asarray(data["reference_alphas"][scale_index]).reshape(grid, grid)
        axis.imshow(image)
        axis.imshow(
            alpha_map,
            cmap="coolwarm",
            norm=Normalize(vmin=0.0, vmax=max(float(np.max(alpha_map)), 1e-8)),
            interpolation="nearest",
            extent=(0, grid * patch_size, grid * patch_size, 0),
            alpha=0.55,
        )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_title(f"{scale_labels[scale_index]} patch importance")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Patch Importance (Alpha) Overlays")
    path = plot_dir / "E18_alpha_patch_importance_overlays.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # E19: Per-scale Q_s, computed as sum(alpha * q) for each test image.
    ordered_quality = [scale_qualities[index] for index in scale_order]
    quality_means = np.asarray([np.mean(values) for values in ordered_quality])
    quality_stds = np.asarray([np.std(values) for values in ordered_quality])
    quality_colors = ["#D55E00", "#E69F00", "#0072B2"][:len(scale_order)]
    fig, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    bars = axis.bar(
        [scale_labels[index] for index in scale_order], quality_means,
        yerr=quality_stds, capsize=5, color=quality_colors,
    )
    for bar, value in zip(bars, quality_means):
        axis.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 5), textcoords="offset points", ha="center", va="bottom")
    axis.set_ylim(0, min(1.0, float(np.max(quality_means + quality_stds)) * 1.18 + 0.02))
    axis.set_xlabel("Scale (coarse to fine)")
    axis.set_ylabel("Mean Q_s (error bars: +/- 1 std)")
    axis.set_title("Per-Scale Quality on Test Set")
    axis.grid(axis="y", alpha=0.2)
    path = plot_dir / "E19_per_scale_quality.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # E20: Map actual fine-scale pooling groups onto the reference image.
    fine_index = 0
    fine_grid = model.grid_sizes[fine_index]
    patch_size = model.patch_sizes[fine_index]
    groups = model.hierarchical_pools[fine_index]._make_groups(
        model.num_patches[fine_index], fine_grid
    )
    region_map = np.full((fine_grid, fine_grid), np.nan, dtype=np.float64)
    for region_index, patch_indices in enumerate(groups):
        for patch_index in patch_indices:
            row, column = divmod(patch_index, fine_grid)
            region_map[row, column] = data["reference_regions"][fine_index][region_index]
    region_cmap = plt.get_cmap("RdYlGn").copy()
    region_cmap.set_bad(alpha=0)
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.imshow(image)
    overlay = axis.imshow(
        np.ma.masked_invalid(region_map), cmap=region_cmap,
        norm=Normalize(vmin=0.0, vmax=1.0), interpolation="nearest",
        extent=(0, fine_grid * patch_size, fine_grid * patch_size, 0), alpha=0.55,
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_title("Fine-Scale Region Quality (Q_regions)")
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar = fig.colorbar(overlay, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Region quality (red=low, green=high)")
    path = plot_dir / "E20_region_quality_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    # E21: Show the exact binary correspondence matrix used by L_cons.
    matrix = np.asarray(correspondence_matrix.detach().cpu(), dtype=np.float64)
    fig_height = max(5.0, min(12.0, matrix.shape[0] * 0.18))
    fig, axis = plt.subplots(figsize=(6, fig_height), constrained_layout=True)
    heatmap = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0,
                          interpolation="nearest", aspect="auto")
    axis.set_xlabel("Coarse-scale patch index")
    axis.set_ylabel("Fine-scale patch index")
    axis.set_title(f"Consistency Correspondence Matrix ({matrix.shape[0]} x {matrix.shape[1]})")
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_yticks(np.arange(matrix.shape[0]))
    if matrix.size <= 100:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.0f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if matrix[row, column] > 0.5 else "#222222")
    fig.colorbar(heatmap, ax=axis, label="Correspondence strength")
    path = plot_dir / "E21_consistency_correspondence_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_paths.append(path)

    return tuple(saved_paths)