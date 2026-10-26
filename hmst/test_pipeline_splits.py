"""Regression tests for leakage-safe SAUD split construction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook
import scipy.io

from data.dataset import (
    SAUDDataset,
    UWIQADataset,
    make_saud_splits,
    make_uwiqa_splits,
    split_saud_train_validation,
)
from models.modules import HierarchicalPooling
from training.metrics import compute_metrics, compute_test_metrics


class SaudSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mat_path = Path(self.temp_dir.name) / "saud.mat"
        records = np.empty((1000, 2), dtype=object)
        for index in range(1000):
            scene, method = divmod(index, 10)
            records[index, 0] = (
                f"group_{scene // 20}/scene_{scene}/method_{method}.png"
            )
            records[index, 1] = float(index)
        scipy.io.savemat(self.mat_path, {"sub_bt_scores": records})

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_validation_and_test_are_raw_scene_disjoint(self):
        train_candidate, test_indices = make_saud_splits(
            str(self.mat_path), seed=23
        )
        train_indices, val_indices = split_saud_train_validation(
            str(self.mat_path), train_candidate, seed=24
        )

        self.assertEqual(
            (len(train_indices), len(val_indices), len(test_indices)),
            (640, 160, 200),
        )
        self.assertEqual(
            set(train_indices) | set(val_indices) | set(test_indices),
            set(range(1000)),
        )
        self.assertFalse(set(train_indices) & set(val_indices))
        self.assertFalse(set(train_indices) & set(test_indices))
        self.assertFalse(set(val_indices) & set(test_indices))

        def scenes(indices):
            return {index // 10 for index in indices}

        self.assertFalse(scenes(train_indices) & scenes(val_indices))
        self.assertFalse(scenes(train_indices) & scenes(test_indices))
        self.assertFalse(scenes(val_indices) & scenes(test_indices))

    def test_saud_scaling_uses_training_indices(self):
        train_candidate, test_indices = make_saud_splits(
            str(self.mat_path), seed=23
        )
        train_indices, _ = split_saud_train_validation(
            str(self.mat_path), train_candidate, seed=24
        )
        dataset = SAUDDataset(
            self.temp_dir.name,
            str(self.mat_path),
            test_indices,
            score_range_indices=train_indices,
        )

        train_min, train_max = min(train_indices), max(train_indices)
        expected_score = (test_indices[0] - train_min) / (train_max - train_min + 1e-8)
        self.assertAlmostEqual(dataset.samples[0][1], expected_score)


class UwiqaSplitTests(unittest.TestCase):
    def test_split_indices_match_only_complete_score_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            score_path = root / "scores.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["image_number", "score"])
            for image_number in range(1, 11):
                sheet.append([image_number, float(image_number) / 10])
                if image_number == 5:
                    sheet.append([11, None])
            workbook.save(score_path)

            train_indices, test_indices = make_uwiqa_splits(
                str(root), str(score_path), seed=4
            )
            dataset = UWIQADataset(
                str(root), str(score_path), train_indices + test_indices
            )

            self.assertEqual(len(train_indices) + len(test_indices), 10)
            self.assertEqual(len(dataset), 10)
            self.assertEqual(
                {Path(path).name for path, _ in dataset.samples},
                {f"{number:04d}.png" for number in range(1, 11)},
            )

    def test_duplicate_image_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            score_path = Path(temp_dir) / "scores.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["image_number", "score"])
            sheet.append([1, 0.3])
            sheet.append([1, 0.7])
            workbook.save(score_path)

            with self.assertRaisesRegex(ValueError, "Duplicate image numbers"):
                make_uwiqa_splits(temp_dir, str(score_path))


class HierarchicalRegionTests(unittest.TestCase):
    def test_all_patches_are_assigned_at_each_scale(self):
        pooling = HierarchicalPooling(d_model=8)
        for grid_size in (7, 3, 1):
            groups = pooling._make_groups(grid_size * grid_size, grid_size)
            indices = [index for group in groups for index in group]
            self.assertEqual(sorted(indices), list(range(grid_size * grid_size)))
            self.assertEqual(len(indices), len(set(indices)))


class TestMetricTests(unittest.TestCase):
    def test_test_metrics_keep_raw_values_primary_and_label_same_split_fit(self):
        predictions = np.array([0.1, 0.4, 0.6, 0.9, 0.7, 0.2])
        targets = np.array([0.2, 0.3, 0.8, 0.7, 0.9, 0.1])

        metrics = compute_test_metrics(predictions, targets)
        raw_metrics = compute_metrics(predictions, targets, use_nonlinear_fit=False)
        fitted_metrics = compute_metrics(predictions, targets, use_nonlinear_fit=True)

        self.assertEqual(metrics["plcc"], raw_metrics["plcc"])
        self.assertEqual(metrics["rmse"], raw_metrics["rmse"])
        self.assertEqual(
            metrics["same_split_fitted_plcc"], fitted_metrics["plcc"]
        )
        self.assertEqual(
            metrics["same_split_fitted_rmse"], fitted_metrics["rmse"]
        )


if __name__ == "__main__":
    unittest.main()
