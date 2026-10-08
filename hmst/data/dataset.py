"""
data/dataset.py
---------------
Dataset loaders for UWIQA and SAUD datasets.

UWIQA
  890 images (0001.png – 0890.png)
  MOS scores in IQA Value.xlsx (range 0.2 – 1.0)
  Split: 80/20 → 712 train / 178 test

SAUD
  100 raw images × 10 UIE methods = 1000 enhanced images
  BT scores in sub_bt_scores.mat (Bradley-Terry, unbounded)
  Split: 800 train / 200 test (per EDANet paper)
  For ranking loss: pairs of enhanced images from the same raw

Both datasets use the same Dataset class interface.
"""

import os
import math
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io
import openpyxl


# ─────────────────────────────────────────────────────────────────────────────
# UWIQA Dataset
# ─────────────────────────────────────────────────────────────────────────────

class UWIQADataset(Dataset):
    """
    UWIQA dataset loader.

    Args:
        image_dir:   Path to folder containing 0001.png – 0890.png
        score_file:  Path to IQA Value.xlsx
        indices:     List of 0-based indices to include (for train/val/test split)
        transform:   torchvision transform to apply to each image
        max_samples: Optional cap on number of samples (for local test mode)
    """

    def __init__(
        self,
        image_dir: str,
        score_file: str,
        indices: List[int],
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Load scores from Excel
        wb = openpyxl.load_workbook(score_file)
        ws = wb.active
        rows = list(ws.iter_rows(min_row=2, values_only=True))
        # rows[i] = (image_number, score)
        all_images = []
        for row in rows:
            img_num, score = row
            if img_num is not None and score is not None:
                fname = self.image_dir / f"{int(img_num):04d}.png"
                all_images.append((str(fname), float(score)))

        # Apply split
        self.samples = [all_images[i] for i in indices if i < len(all_images)]

        # Apply max_samples cap AFTER the split (test mode only)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, score = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "score": torch.tensor(score, dtype=torch.float32),
            "path": img_path,
        }


def make_uwiqa_splits(
    image_dir: str,
    score_file: str,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> Tuple[List[int], List[int]]:
    """
    Returns (train_indices, test_indices) for UWIQA.
    80/20 split, seed-controlled and reproducible.
    """
    wb = openpyxl.load_workbook(score_file)
    ws = wb.active
    n = sum(1 for row in ws.iter_rows(min_row=2, values_only=True)
            if row[0] is not None)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(n * train_ratio)
    return indices[:split], indices[split:]


# ─────────────────────────────────────────────────────────────────────────────
# SAUD Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SAUDDataset(Dataset):
    """
    SAUD dataset loader.

    1000 enhanced images (100 raw × 10 methods).
    BT scores are used as the quality label (normalized to [0,1]).
    Ranking pairs are returned for L_rank training.

    Args:
        benchmark_dir: Path to Benchmark/ (contains Enhanced/ and Raw/)
        bt_scores_mat: Path to sub_bt_scores.mat
        indices:       0-based indices to include
        transform:     torchvision transform
        max_samples:   Optional cap (test mode)
        return_pairs:  If True, __getitem__ returns a pair for ranking loss
    """

    def __init__(
        self,
        benchmark_dir: str,
        bt_scores_mat: str,
        indices: List[int],
        transform=None,
        max_samples: Optional[int] = None,
        return_pairs: bool = False,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.transform = transform
        self.return_pairs = return_pairs

        # Load MAT file
        mat = scipy.io.loadmat(bt_scores_mat)
        bt = mat["sub_bt_scores"]

        raw_samples = []
        for i in range(len(bt)):
            rel_path = str(bt[i, 0].flat[0])
            score = float(bt[i, 1].flat[0])
            img_path = self.benchmark_dir / "Enhanced" / rel_path
            raw_samples.append((str(img_path), score))

        # Normalize BT scores to [0, 1] for MOS-like training
        scores_arr = np.array([s for _, s in raw_samples])
        s_min, s_max = scores_arr.min(), scores_arr.max()
        self.samples = [
            (p, float((s - s_min) / (s_max - s_min + 1e-8)))
            for p, s in raw_samples
        ]

        # Apply split
        self.samples = [self.samples[i] for i in indices if i < len(self.samples)]

        # Apply cap
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Build groups for ranking pairs
        # Images from the same raw image form a group
        # Group key = relative path up to the image number subfolder
        if return_pairs:
            self._build_pair_index()

    def _build_pair_index(self):
        """Pre-build an index of groups for pair sampling."""
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (path, score) in enumerate(self.samples):
            p = Path(path)
            # key = group_X/image_number e.g. group_1/1
            key = f"{p.parts[-3]}/{p.parts[-2]}"
            groups[key].append(idx)
        self.groups = [v for v in groups.values() if len(v) >= 2]

    def __len__(self) -> int:
        if self.return_pairs:
            return len(self.groups)
        return len(self.samples)

    def _load_image(self, idx: int) -> torch.Tensor:
        img_path, _ = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int) -> dict:
        if self.return_pairs:
            group = self.groups[idx]
            i, j = random.sample(group, 2)
            img_a = self._load_image(i)
            img_b = self._load_image(j)
            score_a = self.samples[i][1]
            score_b = self.samples[j][1]
            # label=1 if a is better, 0 otherwise
            label = 1.0 if score_a > score_b else 0.0
            return {
                "image_a": img_a,
                "image_b": img_b,
                "score_a": torch.tensor(score_a, dtype=torch.float32),
                "score_b": torch.tensor(score_b, dtype=torch.float32),
                "label":   torch.tensor(label, dtype=torch.float32),
            }
        else:
            img_path, score = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return {
                "image": image,
                "score": torch.tensor(score, dtype=torch.float32),
                "path": img_path,
            }


def make_saud_splits(
    bt_scores_mat: str,
    seed: int = 42,
    n_train: int = 800,
    n_test: int = 200,
) -> Tuple[List[int], List[int]]:
    """
    Returns (train_indices, test_indices) for SAUD.
    800/200 split (per EDANet paper), seed-controlled.
    """
    mat = scipy.io.loadmat(bt_scores_mat)
    n = len(mat["sub_bt_scores"])
    assert n == 1000, f"Expected 1000 SAUD entries, got {n}"
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:n_train], indices[n_train:n_train + n_test]
