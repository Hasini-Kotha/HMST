"""
dataset.py
---------------
Dataset loaders for UWIQA, SAUD and UIED datasets.

UWIQA
  890 images (0001.png – 0890.png)
  MOS scores in IQA Value.xlsx (range 0.2 – 1.0)
  Split: 80/20 train-candidate/test, then 80/20 train/validation

SAUD
  100 raw images × 10 UIE methods = 1000 enhanced images
  BT scores in sub_bt_scores.mat (Bradley-Terry, unbounded)
  Split: 80/20 by raw scene; validation is grouped within the training scenes
  Optional pair mode groups enhancements from the same raw image.

UIED


Both datasets use the same Dataset class interface.
"""

import os
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            if len(row) < 2:
                continue
            img_num, score = row[:2]
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
    image_numbers = [
        int(row[0])
        for row in ws.iter_rows(min_row=2, values_only=True)
        if len(row) >= 2 and row[0] is not None and row[1] is not None
    ]
    if len(image_numbers) != len(set(image_numbers)):
        raise ValueError(
            f"Duplicate image numbers in UWIQA score file '{score_file}'; "
            "a record-level split could put the same image in multiple splits."
        )
    n = len(image_numbers)
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
    BT scores are scaled with the training-split extrema when supplied.
    Pair sampling is opt-in; the standard trainer uses batchwise score ranking.

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
        score_range_indices: Optional[List[int]] = None,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.transform = transform
        self.return_pairs = return_pairs

        # Load MAT file
        records = _load_saud_records(bt_scores_mat)

        raw_samples = []
        for rel_path, score in records:
            img_path = self.benchmark_dir / "Enhanced" / rel_path
            raw_samples.append((str(img_path), score))

        # Fit label scaling on training labels only when the split is supplied.
        scores_arr = np.array([s for _, s in raw_samples])
        if score_range_indices is None:
            scaling_scores = scores_arr
        else:
            if not score_range_indices:
                raise ValueError("score_range_indices must contain training indices.")
            if any(i < 0 or i >= len(scores_arr) for i in score_range_indices):
                raise IndexError("score_range_indices contains an out-of-range SAUD index.")
            scaling_scores = scores_arr[score_range_indices]
        s_min, s_max = float(scaling_scores.min()), float(scaling_scores.max())
        if s_max <= s_min:
            raise ValueError("SAUD training scores must contain at least two distinct values.")
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


def _load_saud_records(bt_scores_mat: str) -> List[Tuple[str, float]]:
    """Load SAUD image paths and Bradley-Terry scores in MAT-file order."""
    mat = scipy.io.loadmat(bt_scores_mat)
    if "sub_bt_scores" not in mat:
        raise ValueError(f"'{bt_scores_mat}' does not contain 'sub_bt_scores'.")
    bt = mat["sub_bt_scores"]
    return [
        (str(bt[i, 0].flat[0]).replace("\\", "/"), float(bt[i, 1].flat[0]))
        for i in range(len(bt))
    ]


def _saud_scene_key(rel_path: str) -> str:
    """Return the raw-scene directory key used by SAUD's grouped image paths."""
    parts = Path(rel_path.replace("\\", "/")).parts
    if len(parts) < 3:
        raise ValueError(
            "Expected SAUD enhanced-image paths shaped like "
            "'group/<scene>/<method-image>', got "
            f"'{rel_path}'."
        )
    return "/".join(parts[-3:-1])


def make_saud_splits(
    bt_scores_mat: str,
    seed: int = 42,
    n_train: int = 800,
    n_test: int = 200,
) -> Tuple[List[int], List[int]]:
    """
    Returns raw-scene-disjoint (train_indices, test_indices) for SAUD.

    n_train and n_test define the desired ratio; all enhanced images from a
    source scene remain in the same split.
    """
    records = _load_saud_records(bt_scores_mat)
    if len(records) != 1000:
        raise ValueError(f"Expected 1000 SAUD entries, got {len(records)}.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must both be positive.")

    groups: Dict[str, List[int]] = {}
    for index, (rel_path, _) in enumerate(records):
        groups.setdefault(_saud_scene_key(rel_path), []).append(index)
    if len(groups) < 2:
        raise ValueError("SAUD must contain at least two distinct raw-scene groups.")

    train_ratio = n_train / (n_train + n_test)
    train_group_count = round(len(groups) * train_ratio)
    train_group_count = min(max(train_group_count, 1), len(groups) - 1)
    group_keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    train_keys = group_keys[:train_group_count]
    test_keys = group_keys[train_group_count:]
    train_indices = [index for key in train_keys for index in groups[key]]
    test_indices = [index for key in test_keys for index in groups[key]]
    return train_indices, test_indices


def split_saud_train_validation(
    bt_scores_mat: str,
    train_indices: List[int],
    seed: int = 42,
    val_ratio: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """Split SAUD training indices into raw-scene-disjoint train/validation sets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}.")
    records = _load_saud_records(bt_scores_mat)
    groups: Dict[str, List[int]] = {}
    for index in train_indices:
        if index < 0 or index >= len(records):
            raise IndexError(f"SAUD training index {index} is out of range.")
        key = _saud_scene_key(records[index][0])
        groups.setdefault(key, []).append(index)
    if len(groups) < 2:
        raise ValueError("SAUD training split must contain at least two raw-scene groups.")

    group_keys = list(groups)
    random.Random(seed).shuffle(group_keys)
    val_group_count = min(max(round(len(group_keys) * val_ratio), 1), len(group_keys) - 1)
    val_keys = set(group_keys[:val_group_count])
    train_part = [index for key in group_keys[val_group_count:] for index in groups[key]]
    val_part = [index for key in group_keys if key in val_keys for index in groups[key]]
    return train_part, val_part
