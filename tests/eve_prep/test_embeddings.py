"""Validation Group 5 -- task embeddings, the report, and the atomic write."""

import json
import os

import numpy as np
import pytest
import torch

from eve_prep import EvePrepError, FEATURE_SHAPE
from eve_prep.check_features import check_features
from eve_prep.extract_features import (COUNTER_NAMES, copy_task_embeddings,
                                       extract_all)


def _write_embeddings(path, value=None, key="free-viewing"):
    if value is None:
        value = np.zeros(768, dtype=np.float32)
    with open(path, "wb") as fh:
        np.save(fh, {key: value}, allow_pickle=True)
    return str(path)


def test_copy_round_trips(tmp_path):
    src = _write_embeddings(tmp_path / "src.npy")
    dst = str(tmp_path / "dst.npy")
    info = copy_task_embeddings(src, dst)

    loaded = np.load(dst, allow_pickle=True).item()
    assert set(loaded) == {"free-viewing"}
    assert loaded["free-viewing"].dtype == np.float32
    assert info["sha256"] == info["sha256"]
    assert info["key"] == "free-viewing" and info["shape"] == [768]

    import hashlib
    with open(src, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == info["sha256"]


def test_missing_key_raises(tmp_path):
    src = _write_embeddings(tmp_path / "src.npy", key="search")
    with pytest.raises(EvePrepError) as exc:
        copy_task_embeddings(src, str(tmp_path / "dst.npy"))
    assert "free-viewing" in str(exc.value)


def test_wrong_shape_raises(tmp_path):
    src = _write_embeddings(tmp_path / "src.npy", np.zeros(384, dtype=np.float32))
    with pytest.raises(EvePrepError) as exc:
        copy_task_embeddings(src, str(tmp_path / "dst.npy"))
    assert "(384,)" in str(exc.value)


def test_wrong_dtype_raises(tmp_path):
    src = _write_embeddings(tmp_path / "src.npy", np.zeros(768, dtype=np.float64))
    with pytest.raises(EvePrepError) as exc:
        copy_task_embeddings(src, str(tmp_path / "dst.npy"))
    assert "float64" in str(exc.value)


def test_verification_reads_the_destination(tmp_path, monkeypatch):
    """FR6 -- verifying the source would not catch a truncated write."""
    import eve_prep.extract_features as module

    src = _write_embeddings(tmp_path / "src.npy")
    dst = str(tmp_path / "dst.npy")
    # Capture the real function first: module.shutil IS the shutil module, so
    # patching its attribute and then calling through it would recurse.
    real_copyfile = module.shutil.copyfile

    def truncating_copy(a, b):
        real_copyfile(a, b)
        with open(b, "r+b") as fh:
            fh.truncate(16)

    monkeypatch.setattr(module.shutil, "copyfile", truncating_copy)
    with pytest.raises(EvePrepError):
        copy_task_embeddings(src, dst)


# --------------------------------------------------------------------------
# extract_all -- counters, resumption, and the atomic write
# --------------------------------------------------------------------------

class _FakeBundle(object):
    """Returns a deterministic, per-key stimulus without touching a real bundle."""

    #: main() reads this before the derivation is reached; the derivation itself is
    #: monkeypatched, so the value is never used.
    samples_df = None

    def __init__(self, keys):
        self.keys = list(keys)
        self.calls = []

    def get_stimulus(self, exp_key):
        self.calls.append(exp_key)
        seed = self.keys.index(exp_key)
        return (np.random.RandomState(seed).rand(1080, 1920, 3) * 255).astype(np.uint8)


class _FakeBackbone(object):
    """Cheap stand-in: the real backbone is exercised by Group 1."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.full((1,) + FEATURE_SHAPE, float(self.calls),
                          dtype=torch.float32)


def test_extract_all_writes_and_counts(tmp_path, exp_keys):
    keys = sorted(exp_keys.values())
    result = extract_all(_FakeBundle(keys), keys, str(tmp_path),
                         torch.device("cpu"), backbone=_FakeBackbone())
    assert result["n_extracted"] == len(keys)
    assert result["n_skipped_existing"] == 0
    assert set(result["counters"]) == set(COUNTER_NAMES)
    assert all(v == 0 for k, v in result["counters"].items()
               if k != "skipped_existing")
    assert len(result["feature_sha256"]) == len(keys)
    for sha in result["feature_sha256"].values():
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
    for key in keys:
        assert os.path.isfile(os.path.join(str(tmp_path), "image_features",
                                           key + ".pth"))


def test_extract_all_is_resumable(tmp_path, exp_keys):
    keys = sorted(exp_keys.values())
    bundle = _FakeBundle(keys)
    first = extract_all(bundle, keys, str(tmp_path), torch.device("cpu"),
                        backbone=_FakeBackbone())
    second = extract_all(bundle, keys, str(tmp_path), torch.device("cpu"),
                         backbone=_FakeBackbone())
    assert second["n_extracted"] == 0
    assert second["n_skipped_existing"] == len(keys)
    assert second["counters"]["skipped_existing"] == len(keys)
    # The sha map is reported for skipped files too, so a resumed run still
    # describes the whole cache (FR8.2).
    assert second["feature_sha256"] == first["feature_sha256"]


def test_interrupted_write_leaves_nothing_loadable(tmp_path, bridge, exp_keys,
                                                   monkeypatch):
    """Group 5 -- a kill between torch.save and os.replace reads as MISSING."""
    import eve_prep.extract_features as module

    keys = sorted(exp_keys.values())
    victim = keys[0]

    real_replace = os.replace

    def dying_replace(a, b):
        if victim + ".pth" in str(b):
            raise KeyboardInterrupt("killed between save and replace")
        return real_replace(a, b)

    monkeypatch.setattr(module.os, "replace", dying_replace)
    out_dir = str(tmp_path / "out")
    with pytest.raises(KeyboardInterrupt):
        extract_all(_FakeBundle(keys), keys, out_dir, torch.device("cpu"),
                    backbone=_FakeBackbone())

    feat_dir = os.path.join(out_dir, "image_features")
    assert os.path.isfile(os.path.join(feat_dir, victim + ".pth.tmp"))
    assert not os.path.isfile(os.path.join(feat_dir, victim + ".pth"))

    with pytest.raises(EvePrepError) as exc:
        check_features(bridge["fixations"], bridge["heatmaps"], feat_dir)
    assert "missing" in str(exc.value)
    assert "wrong shape" not in str(exc.value)


def test_report_written_when_no_work_is_done(tmp_path, bridge, exp_keys,
                                             monkeypatch, capsys):
    """FR8.3 -- a fully-skipped run must still leave a record (D5)."""
    import eve_prep.extract_features as module

    out_dir = str(tmp_path / "out")
    keys = sorted(exp_keys.values())
    extract_all(_FakeBundle(keys), keys, out_dir, torch.device("cpu"),
                backbone=_FakeBackbone())

    osie_src = _write_embeddings(tmp_path / "embeddings.npy")
    bridge_report = str(tmp_path / "bridge_report.json")
    with open(bridge_report, "w") as fh:
        json.dump({"num_trials": 8, "num_trials_test": 6, "num_trials_train": 2}, fh)
    subject_map = str(tmp_path / "subject_id_map.json")
    with open(subject_map, "w") as fh:
        json.dump({"to_dense": {}, "to_eve": {}}, fh)

    monkeypatch.setattr(module, "_open_bundle", lambda d: _FakeBundle(keys))
    monkeypatch.setattr(module, "derive_trial_exp_keys",
                        lambda df, fx, sm: dict(exp_keys))
    monkeypatch.setattr(module, "versions",
                        lambda device=None: {"python": "3.12", "torch": "x",
                                             "cuda_available": True,
                                             "device": str(device)})

    bridge_dir = os.path.dirname(bridge["fixations"])
    for src, dst in [(bridge_report, os.path.join(bridge_dir, "bridge_report.json")),
                     (subject_map, os.path.join(bridge_dir, "subject_id_map.json"))]:
        if not os.path.exists(dst):
            os.replace(src, dst)

    assert module.main([
        "--bundle-dir", str(tmp_path), "--bridge-dir", bridge_dir,
        "--out-dir", out_dir, "--osie-embeddings", osie_src, "--allow-cpu"]) == 0
    capsys.readouterr()

    with open(os.path.join(out_dir, "feature_report.json")) as fh:
        report = json.load(fh)

    assert report["n_extracted"] == 0
    assert report["n_skipped_existing"] == report["n_trials"] == 8
    expected_keys = {
        "args", "versions", "created_utc", "bundle_dir", "fixations_path",
        "heatmaps_path", "fixations_sha256", "n_trials", "n_trials_test",
        "n_trials_train", "n_extracted", "n_skipped_existing", "feature_shape",
        "resize_input", "squash", "keying", "embeddings", "exp_key_crosscheck",
        "counters", "feature_sha256"}
    assert set(report) == expected_keys
    assert set(report["counters"]) == set(COUNTER_NAMES)
    assert report["keying"] == "exp_key"
    assert report["feature_shape"] == [768, 2048]
    assert report["resize_input"] == [768, 1024]
    assert report["squash"]["uniform"] is False
    assert abs(report["squash"]["x"] - 1024 / 1920.0) < 1e-12
    assert abs(report["squash"]["y"] - 768 / 1080.0) < 1e-12
    assert report["exp_key_crosscheck"]["agree"] is True
    assert len(report["feature_sha256"]) == 8
