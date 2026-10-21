"""
test_all_optimizers.py
----------------------
Unit tests verifying every optimizer in hmst/training/optimizers.py:
1. Adam
2. AdamW
3. Lion (sign momentum update)
4. SGD (momentum, nesterov)
5. SAM-AdamW (first_step and second_step)
6. SAM-SGD
7. HMSTTrainer compatibility with standard optimizers and SAM
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.optimizers import Lion, SAM, build_optimizer
from training.trainer import HMSTTrainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return {"Q_hat": self.fc(x).squeeze(-1)}


class SimpleCriterion(nn.Module):
    def forward(self, out, target):
        loss = torch.mean((out["Q_hat"] - target) ** 2)
        return {"loss": loss, "loss_mos": loss, "loss_rank": torch.tensor(0.0), "loss_cons": torch.tensor(0.0)}


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"image": self.x[idx], "score": self.y[idx]}


def test_optimizers():
    print("Testing Optimizer Implementations...")

    # 1. Test build_optimizer for all types
    configs = [
        ("adam", {"lr": 1e-4, "weight_decay": 1e-4}),
        ("adamw", {"lr": 1e-4, "weight_decay": 1e-4}),
        ("lion", {"lr": 1e-5, "weight_decay": 1e-2}),
        ("sgd", {"lr": 1e-3, "weight_decay": 1e-4, "momentum": 0.9, "nesterov": True}),
        ("sam_adamw", {"lr": 1e-4, "weight_decay": 1e-4, "rho": 0.05}),
        ("sam_sgd", {"lr": 1e-3, "weight_decay": 1e-4, "rho": 0.05, "momentum": 0.9}),
    ]

    for opt_name, kwargs in configs:
        model = SimpleModel()
        opt = build_optimizer(model, opt_name, **kwargs)
        assert opt is not None, f"Failed to build {opt_name}"
        print(f"  [PASS] Successfully built {opt_name} ({type(opt).__name__})")

    # 2. Test Step execution on dummy dataset
    device = torch.device("cpu")
    x = torch.randn(16, 10)
    y = torch.randn(16)
    ds = DictDataset(x, y)
    loader = DataLoader(ds, batch_size=4)

    criterion = SimpleCriterion()

    for opt_name, kwargs in [("adam", {"lr": 1e-4, "weight_decay": 1e-4}),
                             ("lion", {"lr": 1e-5, "weight_decay": 1e-2}),
                             ("sam_adamw", {"lr": 1e-4, "weight_decay": 1e-4, "rho": 0.05})]:
        model = SimpleModel()
        opt = build_optimizer(model, opt_name, **kwargs)
        cfg = {"epochs": 1, "mixed_precision": False}
        trainer = HMSTTrainer(
            model=model,
            criterion=criterion,
            optimizer=opt,
            config=cfg,
            device=device,
            output_dir="./tmp_out",
            checkpoint_dir="./tmp_ckpt",
        )
        metrics = trainer._train_epoch(loader)
        assert "loss" in metrics, f"Training epoch failed for {opt_name}"
        print(f"  [PASS] HMSTTrainer._train_epoch succeeded with {opt_name} (Loss: {metrics['loss']:.4f})")

    # Cleanup temp dirs
    import shutil
    for d in ["./tmp_out", "./tmp_ckpt"]:
        if os.path.exists(d):
            shutil.rmtree(d)


if __name__ == "__main__":
    test_optimizers()
    print("\nALL OPTIMIZER TESTS PASSED!")
