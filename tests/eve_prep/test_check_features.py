"""Validation Group 4 -- the preflight guard, whose exit code gates the GPU run."""

import builtins
import json
import os

import pytest
import torch

from conftest import write_fixations, write_store
from eve_prep import EvePrepError
from eve_prep.check_features import check_features, main


def test_complete_cache_passes(bridge, feature_dir):
    summary = check_features(bridge["fixations"], bridge["heatmaps"], feature_dir)
    assert summary["n_checked"] == 8
    assert summary["n_test"] == 6
    assert summary["n_train"] == 2
    assert summary["n_test"] + summary["n_train"] == summary["n_checked"]
    assert summary["missing"] == [] and summary["bad_shape"] == []


def test_missing_file_raises_and_names_it(bridge, feature_dir, exp_keys):
    victim = exp_keys[("shared_a.jpg", 1)]
    os.remove(os.path.join(feature_dir, victim + ".pth"))
    with pytest.raises(EvePrepError) as exc:
        check_features(bridge["fixations"], bridge["heatmaps"], feature_dir)
    msg = str(exc.value)
    assert victim + ".pth" in msg
    assert "1 total" in msg


def test_bad_shape_is_a_distinct_failure(bridge, feature_dir, exp_keys):
    victim = exp_keys[("shared_b.jpg", 2)]
    torch.save(torch.zeros((700, 2048), dtype=torch.float32),
               os.path.join(feature_dir, victim + ".pth"))
    with pytest.raises(EvePrepError) as exc:
        check_features(bridge["fixations"], bridge["heatmaps"], feature_dir)
    msg = str(exc.value)
    assert "wrong shape or dtype" in msg
    assert "missing" not in msg
    assert "(700, 2048)" in msg


def test_wrong_dtype_is_caught(bridge, feature_dir, exp_keys):
    """FR7.2 -- a right-shaped float64 tensor is still wrong."""
    victim = exp_keys[("shared_b.jpg", 0)]
    torch.save(torch.zeros((768, 2048), dtype=torch.float64),
               os.path.join(feature_dir, victim + ".pth"))
    with pytest.raises(EvePrepError) as exc:
        check_features(bridge["fixations"], bridge["heatmaps"], feature_dir)
    assert "torch.float64" in str(exc.value)


def test_split_selection_ignores_the_other_split(bridge, feature_dir, exp_keys):
    for name, subject in [("support_x.jpg", 0), ("support_y.jpg", 1)]:
        os.remove(os.path.join(feature_dir, exp_keys[(name, subject)] + ".pth"))
    summary = check_features(bridge["fixations"], bridge["heatmaps"], feature_dir,
                             split="test")
    assert summary["n_checked"] == 6 and summary["n_train"] == 0
    with pytest.raises(EvePrepError):
        check_features(bridge["fixations"], bridge["heatmaps"], feature_dir,
                       split="train")


def test_train_split_passes_while_test_is_absent(bridge, feature_dir, exp_keys):
    for (name, subject), key in exp_keys.items():
        if name.startswith("shared"):
            os.remove(os.path.join(feature_dir, key + ".pth"))
    summary = check_features(bridge["fixations"], bridge["heatmaps"], feature_dir,
                             split="train")
    assert summary["n_checked"] == 2 and summary["n_test"] == 0


def test_phantom_trial_raises(tmp_path, fixations, exp_keys, feature_dir):
    """FR2.1 -- a trial in fixations.json that the store has never heard of."""
    store_records = fixations[:-1]
    fix = write_fixations(tmp_path / "fixations.json", fixations)
    # The store must carry the MATCHING hash, or FR2.5 fires first and masks this.
    store = write_store(tmp_path / "gt.h5", store_records, exp_keys,
                        fixations_path=fix)
    with pytest.raises(EvePrepError) as exc:
        check_features(fix, store, feature_dir)
    assert "not in the heatmap store" in str(exc.value)
    assert "support_y.jpg" in str(exc.value)


def test_imports_without_evedataset(monkeypatch):
    """Group 4 -- the guard must run before the bundle is staged on the cluster."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "evedataset" or name.startswith("evedataset."):
            raise ImportError("evedataset is not installed in this env")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(__import__("sys").modules, "evedataset", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    import importlib

    import eve_prep.check_features as module
    importlib.reload(module)
    assert hasattr(module, "check_features")


def test_main_exit_codes(bridge, feature_dir, exp_keys, capsys):
    argv = ["--fix", bridge["fixations"], "--heatmaps", bridge["heatmaps"],
            "--feat-dir", feature_dir]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_checked"] == 8

    os.remove(os.path.join(feature_dir, exp_keys[("shared_a.jpg", 0)] + ".pth"))
    assert main(argv) == 1
    assert "FATAL preflight failure" in capsys.readouterr().err
