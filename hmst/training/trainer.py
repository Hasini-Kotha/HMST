"""
training/trainer.py
--------------------
Clean training loop for HMST — Architecture 1.

Flow:
    Load configuration
          ↓
    Set seed → device → dataset → dataloaders
          ↓
    Create model → loss → optimizer
          ↓
    Load checkpoint if requested
          ↓
    For each epoch:
        Training (forward → loss → backward → optimizer step)
        Validation
        Metrics (PLCC, SROCC, KRCC, RMSE)
        Save latest checkpoint
        Save best checkpoint
          ↓
    Final evaluation → Save results
"""

import os
import time
import random
import csv
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from training.metrics import compute_metrics, collect_predictions
from training.checkpoint import save_checkpoint, load_checkpoint
from training.plots import save_training_dynamics_plots
from training.optimizers import SAM


class HMSTTrainer:
    """
    Trainer for Architecture 1 (HMST).

    Args:
        model:        HMST model instance
        criterion:    HMSTLoss instance
        optimizer:    Optimizer (Adam)
        config:       Configuration dict (from YAML)
        device:       torch.device
        output_dir:   Where to save history.csv and metrics.json
        checkpoint_dir: Where to save checkpoints
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict,
        device: torch.device,
        output_dir: str,
        checkpoint_dir: str,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.training_history = []
        
        self.save_monitor = config.get("checkpoint", {}).get("save_best_on", "srocc")
        self.save_mode = config.get("checkpoint", {}).get("mode", "max")
        self.best_metric = -float("inf") if self.save_mode == "max" else float("inf")
        
        es_cfg = config.get("early_stopping", {})
        self.es_monitor = es_cfg.get("monitor", "srocc")
        self.es_mode = es_cfg.get("mode", "max")
        self.es_patience = es_cfg.get("patience", 10)
        self.es_min_delta = es_cfg.get("min_delta", 0.0)
        self.es_counter = 0
        self.best_es_metric = -float("inf") if self.es_mode == "max" else float("inf")
        self.min_epochs = config.get("min_epochs", 0)

        self.start_epoch = 0
        self.epochs = config.get("epochs", 50)

        # Optimizer type inspection (e.g. SAM requires dual-pass forward/backward)
        self.is_sam = isinstance(optimizer, SAM)

        # AMP scaler — only if mixed_precision and CUDA available (and not SAM)
        self.use_amp = config.get("mixed_precision", False) and device.type == "cuda" and not self.is_sam
        self.scaler = GradScaler("cuda") if self.use_amp else None

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        running = {"loss": 0.0, "loss_mos": 0.0, "loss_rank": 0.0, "loss_cons": 0.0}
        predictions = []
        targets = []
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            scores = batch["score"].to(self.device)

            if self.is_sam:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss_dict = self.criterion(output, scores)
                loss_dict["loss"].backward()
                self.optimizer.first_step(zero_grad=True)

                output2 = self.model(images)
                loss_dict2 = self.criterion(output2, scores)
                loss_dict2["loss"].backward()
                self.optimizer.second_step(zero_grad=True)
            elif self.use_amp:
                self.optimizer.zero_grad()
                with autocast("cuda"):
                    output = self.model(images)
                    loss_dict = self.criterion(output, scores)
                self.scaler.scale(loss_dict["loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss_dict = self.criterion(output, scores)
                loss_dict["loss"].backward()
                self.optimizer.step()

            for k in running:
                running[k] += loss_dict[k].item()
            predictions.append(output["Q_hat"].detach().cpu().numpy())
            targets.append(scores.detach().cpu().numpy())
            n_batches += 1

        epoch_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
        if predictions:
            epoch_metrics.update(compute_metrics(np.concatenate(predictions), np.concatenate(targets)))
        return epoch_metrics

    def _val_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation and compute all metrics."""
        self.model.eval()
        preds_list = []
        tgts_list = []
        running = {"loss": 0.0, "loss_mos": 0.0, "loss_rank": 0.0, "loss_cons": 0.0}
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                scores = batch["score"].to(self.device)

                output = self.model(images)
                loss_dict = self.criterion(output, scores)

                for key in running:
                    running[key] += loss_dict[key].item()
                preds_list.append(output["Q_hat"].cpu().numpy())
                tgts_list.append(scores.cpu().numpy())
                n_batches += 1

        predictions = np.concatenate(preds_list)
        targets     = np.concatenate(tgts_list)
        metrics = compute_metrics(predictions, targets)
        for key, value in running.items():
            metrics[key] = value / max(n_batches, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        resume_path:  Optional[str] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training DataLoader
            val_loader:   Validation DataLoader
            resume_path:  Optional path to a checkpoint to resume from
        """
        # ── Resume from checkpoint ─────────────────────────────────────────
        if resume_path and Path(resume_path).exists():
            print(f"\n[Trainer] Resuming from: {resume_path}")
            ckpt = load_checkpoint(resume_path, self.model, self.optimizer, self.device)
            self.start_epoch = ckpt["epoch"] + 1
            default_best = -float("inf") if self.save_mode == "max" else float("inf")
            self.best_metric = ckpt.get("best_metric", default_best)
            self.best_es_metric = self.best_metric  # Approximate sync on resume
            self.training_history = ckpt.get("training_history", [])
            print(f"[Trainer] Continuing from epoch {self.start_epoch}")
        else:
            print(f"\n[Trainer] Starting training from epoch 0")

        # ── Training loop ──────────────────────────────────────────────────
        save_every = self.config.get("save_every_epoch", False)

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            learning_rate = self.optimizer.param_groups[0]["lr"]

            # Training
            train_metrics = self._train_epoch(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            val_metrics = self._val_epoch(val_loader)

            epoch_time = time.time() - epoch_start

            # Metrics for checkpointing and early stopping
            ckpt_val = val_metrics.get(self.save_monitor, 0.0)
            es_val = val_metrics.get(self.es_monitor, 0.0)

            # ── Logging ────────────────────────────────────────────────────
            epoch_record = {
                "epoch":          epoch + 1,
                "train_loss":     train_metrics["loss"],
                "train_loss_mos": train_metrics["loss_mos"],
                "train_loss_rank":train_metrics["loss_rank"],
                "train_loss_cons":train_metrics["loss_cons"],
                "train_plcc":      train_metrics.get("plcc", float("nan")),
                "train_srocc":     train_metrics.get("srocc", float("nan")),
                "train_krcc":      train_metrics.get("krcc", float("nan")),
                "train_rmse":      train_metrics.get("rmse", float("nan")),
                "val_loss":       val_metrics["loss"],
                "val_loss_mos":   val_metrics["loss_mos"],
                "val_loss_rank":  val_metrics["loss_rank"],
                "val_loss_cons":  val_metrics["loss_cons"],
                "val_plcc":       val_metrics["plcc"],
                "val_srocc":      val_metrics["srocc"],
                "val_krcc":       val_metrics["krcc"],
                "val_rmse":       val_metrics["rmse"],
                "learning_rate":  learning_rate,
                "time_s":         epoch_time,
            }
            self.training_history.append(epoch_record)
            self._write_history()

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(MOS={train_metrics['loss_mos']:.4f} "
                f"Rank={train_metrics['loss_rank']:.4f} "
                f"Cons={train_metrics['loss_cons']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"PLCC: {val_metrics['plcc']:.4f} | "
                f"SROCC: {val_metrics['srocc']:.4f} | "
                f"KRCC: {val_metrics['krcc']:.4f} | "
                f"RMSE: {val_metrics['rmse']:.4f} | "
                f"Best {self.save_monitor.upper()}: {self.best_metric:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # ── Save latest checkpoint ─────────────────────────────────────
            self._save_ckpt("latest.pt", epoch)

            # ── Save epoch checkpoint (test mode) ─────────────────────────
            if save_every:
                self._save_ckpt(f"epoch_{epoch+1:03d}.pt", epoch)

            # ── Save best checkpoint ───────────────────────────────────────
            is_best = False
            if self.save_mode == "max":
                if ckpt_val > self.best_metric:
                    is_best = True
            else:
                if ckpt_val < self.best_metric:
                    is_best = True
                    
            if is_best:
                self.best_metric = ckpt_val
                self._save_ckpt("best.pt", epoch)
                print(f"  → New best {self.save_monitor.upper()}: {self.best_metric:.4f} — saved best.pt")

            # ── Early Stopping ─────────────────────────────────────────────
            es_improved = False
            if self.es_mode == "max":
                if es_val > self.best_es_metric + self.es_min_delta:
                    es_improved = True
            else:
                if es_val < self.best_es_metric - self.es_min_delta:
                    es_improved = True
                    
            if es_improved:
                self.best_es_metric = es_val
                self.es_counter = 0
                print(f"  → {self.es_monitor.upper()} improved to {es_val:.4f}")
            else:
                self.es_counter += 1
                print(f"  → {self.es_monitor.upper()} did not improve. Patience: {self.es_counter}/{self.es_patience}")
                
            if self.es_counter >= self.es_patience and (epoch + 1) >= self.min_epochs:
                print(f"\n[Trainer] Early stopping triggered after {epoch + 1} epochs!")
                break

        print("\n[Trainer] Training complete.")
        dynamics_paths = save_training_dynamics_plots(
            self.training_history,
            output_dir=str(self.output_dir / "plots"),
            monitor=self.es_monitor,
            mode=self.es_mode,
        )
        for plot_path in dynamics_paths:
            print(f"[Trainer] Training dynamics plot saved to {plot_path}")
        self._save_final_metrics(val_loader)

    def _save_ckpt(self, filename: str, epoch: int) -> None:
        """Save a checkpoint to checkpoint_dir/filename."""
        save_checkpoint(
            path=str(self.checkpoint_dir / filename),
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            best_metric=self.best_metric,
            training_history=self.training_history,
            config=self.config,
        )

    def _write_history(self) -> None:
        """Write training history to CSV."""
        if not self.training_history:
            return
        csv_path = self.output_dir / "history.csv"
        keys = list(dict.fromkeys(
            key for record in self.training_history for key in record
        ))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.training_history)

    def _save_final_metrics(self, val_loader: DataLoader) -> None:
        """Collect final predictions and save metrics.json."""
        result = collect_predictions(self.model, val_loader, self.device)
        metrics = compute_metrics(result["predictions"], result["targets"])
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Trainer] Final metrics saved to {metrics_path}")
        print(f"  PLCC:  {metrics['plcc']:.4f}")
        print(f"  SROCC: {metrics['srocc']:.4f}")
        print(f"  KRCC:  {metrics['krcc']:.4f}")
        print(f"  RMSE:  {metrics['rmse']:.4f}")
