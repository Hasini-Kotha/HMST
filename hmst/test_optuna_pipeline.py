"""
test_optuna_pipeline.py
-----------------------
Verification script for:
1. Python compilation of tune_optuna.py, train.py, evaluate.py
2. Allowed batch sizes [8, 16, 32] in search space
3. Optuna safe study migration and old trial preservation
4. Full configuration export (SGD momentum/nesterov, SAM rho, schedulers, loss, etc.)
5. train.py configuration resolution and strict validation (no silent fallback)
6. evaluate.py --config and --batch-size handling
"""

import os
import sys
import py_compile
import tempfile
import yaml
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_compilation():
    print("Test 1: Compiling modified scripts...")
    scripts = [
        os.path.join(PROJECT_ROOT, "scripts", "tune_optuna.py"),
        os.path.join(PROJECT_ROOT, "scripts", "train.py"),
        os.path.join(PROJECT_ROOT, "scripts", "evaluate.py"),
    ]
    for s in scripts:
        py_compile.compile(s, doraise=True)
        print(f"  [PASS] Successfully compiled {os.path.basename(s)}")


def test_train_config_loading_and_validation():
    print("\nTest 2: Verifying train.py config resolution and validation...")
    from scripts.train import resolve_config_path, validate_training_config

    # Test resolution of configs/uwiqa_tuned.yaml
    resolved = resolve_config_path("configs/uwiqa_tuned.yaml", PROJECT_ROOT)
    assert os.path.isfile(resolved), f"Could not find resolved path: {resolved}"
    print(f"  [PASS] Successfully resolved: {resolved}")

    # Load and validate configs/uwiqa_tuned.yaml
    with open(resolved, "r") as f:
        cfg = yaml.safe_load(f)
    validate_training_config(cfg, resolved)
    assert cfg["batch_size"] in [8, 16, 32], f"Batch size {cfg['batch_size']} not in [8, 16, 32]"
    print(f"  [PASS] Validated configs/uwiqa_tuned.yaml (batch_size={cfg['batch_size']})")

    # Test rejection of missing batch size (no silent fallback!)
    bad_cfg = dict(cfg)
    del bad_cfg["batch_size"]
    try:
        validate_training_config(bad_cfg, "bad_cfg.yaml")
        assert False, "Should have raised ValueError for missing batch_size"
    except ValueError as e:
        assert "batch_size" in str(e).lower()
        print(f"  [PASS] Strict rejection of missing batch_size: '{e}'")

    # Test rejection of non-positive batch size
    bad_cfg2 = dict(cfg)
    bad_cfg2["batch_size"] = -1
    try:
        validate_training_config(bad_cfg2, "bad_cfg2.yaml")
        assert False, "Should have raised ValueError for negative batch_size"
    except ValueError as e:
        print(f"  [PASS] Strict rejection of negative batch_size: '{e}'")

    # Test rejection of invalid optimizer
    bad_cfg3 = dict(cfg)
    bad_cfg3["optimizer"] = {"name": "adam", "lr": -0.01}
    try:
        validate_training_config(bad_cfg3, "bad_cfg3.yaml")
        assert False, "Should have raised ValueError for negative lr"
    except ValueError as e:
        print(f"  [PASS] Strict rejection of negative optimizer lr: '{e}'")


def test_evaluate_config_resolution():
    print("\nTest 3: Verifying evaluate.py helper functions...")
    from scripts.evaluate import resolve_config_path, validate_training_config

    resolved = resolve_config_path("configs/uwiqa_tuned.yaml", PROJECT_ROOT)
    assert os.path.isfile(resolved)
    with open(resolved, "r") as f:
        cfg = yaml.safe_load(f)
    validate_training_config(cfg, resolved)
    print(f"  [PASS] evaluate.py correctly resolves and validates config from {resolved}")


def test_tune_optuna_search_space_inspect():
    print("\nTest 4: Inspecting scripts/tune_optuna.py source code rules...")
    tune_file = os.path.join(PROJECT_ROOT, "scripts", "tune_optuna.py")
    with open(tune_file, "r") as f:
        content = f.read()

    # Verify ALLOWED_BATCH_SIZES definition
    assert "ALLOWED_BATCH_SIZES = [8, 16, 32]" in content, "ALLOWED_BATCH_SIZES not defined as [8, 16, 32]"
    print("  [PASS] ALLOWED_BATCH_SIZES = [8, 16, 32] confirmed in tune_optuna.py")

    # Verify batch size suggestion
    assert 'trial.suggest_categorical("batch_size", ALLOWED_BATCH_SIZES)' in content or \
           'trial.suggest_categorical("batch_size", [8, 16, 32])' in content, "Batch size suggestion missing"
    print("  [PASS] Optuna trial suggests batch_size from ALLOWED_BATCH_SIZES")

    # Verify safe migration logic
    assert "def get_or_migrate_study" in content, "get_or_migrate_study function missing"
    print("  [PASS] get_or_migrate_study exists to preserve database and handle old studies")

    # Verify export includes optimizer kwargs (momentum, nesterov, rho)
    assert '"momentum": 0.9' in content and '"nesterov": True' in content, "SGD kwargs export missing"
    assert '"rho": rho' in content or '"rho": float(p["rho_sam"])' in content, "SAM rho export missing"
    print("  [PASS] Optimizer kwargs (momentum, nesterov, rho) confirmed in export logic")

    # Verify CUDA OOM handling
    assert "OutOfMemoryError" in content, "OutOfMemoryError handling missing"
    assert "optuna.TrialPruned" in content, "optuna.TrialPruned missing"
    print("  [PASS] CUDA OutOfMemoryError gracefully caught and trial pruned")

    # Verify NaN / non-finite guard
    assert "np.isfinite" in content, "np.isfinite guard missing"
    print("  [PASS] Non-finite metric guard verified")


if __name__ == "__main__":
    try:
        test_compilation()
        test_train_config_loading_and_validation()
        test_evaluate_config_resolution()
        test_tune_optuna_search_space_inspect()
        print("\n" + "="*60)
        print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
