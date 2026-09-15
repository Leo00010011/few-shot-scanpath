"""The post-run validator: it must pass on a good cache and FAIL on a broken one.

A checker that cannot fail is not a check, so every case here breaks the cache in one
specific way and asserts that the matching check -- and only sensible ones -- fails.
"""

import json
import os

import numpy as np
import pytest
import torch

from eve_prep import FEATURE_SHAPE
from eve_prep.validate_features import MIN_FEATURE_BYTES, main, validate


@pytest.fixture
def cache(tmp_path, bridge, exp_keys, fixations):
    """A complete, plausible split=test feature cache plus its feature_report.json."""
    features = tmp_path / "features"
    feat_dir = features / "image_features"
    feat_dir.mkdir(parents=True)

    test_trials = [(r["name"], int(r["subject"])) for r in fixations
                   if r["split"] == "test"]
    keys = sorted(exp_keys[t] for t in test_trials)

    rng = np.random.RandomState(0)
    sha = {}
    for i, key in enumerate(keys):
        # Post-ReLU-looking: non-negative, sparse, non-constant, and per-stimulus
        # correlated so the cosine check has something real to measure.
        stimulus = key_to_stimulus(key, test_trials, exp_keys)
        base = rng.RandomState if False else None  # noqa: F841  (clarity only)
        t = _synthetic(stimulus, i)
        path = feat_dir / (key + ".pth")
        torch.save(t, str(path))
        sha[key] = _sha(str(path))

    bridge_report = json.load(open(os.path.join(
        os.path.dirname(bridge["fixations"]), "bridge_report.json")))

    report = {
        "args": {"split": "test"},
        "versions": {"python": "3.12"},
        "created_utc": "2026-09-15T00:00:00Z",
        "bundle_dir": "/scratch/bundle",
        "fixations_path": bridge["fixations"],
        "heatmaps_path": bridge["heatmaps"],
        "fixations_sha256": _sha(bridge["fixations"]),
        "n_trials": len(keys),
        "n_trials_test": len(keys),
        "n_trials_train": 0,
        "n_extracted": len(keys),
        "n_skipped_existing": 0,
        "feature_shape": list(FEATURE_SHAPE),
        "resize_input": [768, 1024],
        "squash": {"x": 1024 / 1920.0, "y": 768 / 1080.0, "uniform": False},
        "keying": "exp_key",
        "embeddings": {"src": str(tmp_path / "osie_embeddings.npy"),
                       "dst": str(features / "embeddings.npy"),
                       "sha256": "", "key": "free-viewing",
                       "shape": [768], "dtype": "float32"},
        "exp_key_crosscheck": {"agree": True, "n": len(exp_keys)},
        "counters": {},
        "feature_sha256": sha,
    }

    emb = {"free-viewing": np.zeros(768, dtype=np.float32)}
    for path in (tmp_path / "osie_embeddings.npy", features / "embeddings.npy"):
        with open(path, "wb") as fh:
            np.save(fh, emb, allow_pickle=True)
    report["embeddings"]["sha256"] = _sha(str(features / "embeddings.npy"))

    with open(features / "feature_report.json", "w") as fh:
        json.dump(report, fh)

    return {"features": str(features), "feat_dir": str(feat_dir),
            "bridge_dir": os.path.dirname(bridge["fixations"]),
            "keys": keys, "report": report,
            "report_path": str(features / "feature_report.json")}


def key_to_stimulus(key, trials, exp_keys):
    for (name, subject) in trials:
        if exp_keys[(name, subject)] == key:
            return name
    return key


def _synthetic(stimulus, i):
    """Same stimulus -> correlated tensors; different stimulus -> not."""
    shared = np.random.RandomState(abs(hash(stimulus)) % 2 ** 31).rand(*FEATURE_SHAPE)
    jitter = np.random.RandomState(1000 + i).rand(*FEATURE_SHAPE)
    arr = (0.9 * shared + 0.1 * jitter).astype(np.float32)
    arr[arr < 0.45] = 0.0                       # post-ReLU sparsity, ~45 %
    return torch.from_numpy(arr)


def _sha(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _status(summary, name):
    return next(r["status"] for r in summary["checks"] if r["check"] == name)


def _write_bridge_report(bridge_dir, n_test, n_train, n_subjects=3):
    path = os.path.join(bridge_dir, "bridge_report.json")
    with open(path, "w") as fh:
        json.dump({"num_trials": n_test + n_train, "num_trials_test": n_test,
                   "num_trials_train": n_train, "num_subjects": n_subjects,
                   "fixations_sha256": _sha(os.path.join(bridge_dir,
                                                         "fixations.json"))}, fh)


@pytest.fixture(autouse=True)
def _bridge_report(bridge, fixations):
    d = os.path.dirname(bridge["fixations"])
    n_test = sum(1 for r in fixations if r["split"] == "test")
    n_train = sum(1 for r in fixations if r["split"] == "train")
    _write_bridge_report(d, n_test, n_train)
    with open(os.path.join(d, "subject_id_map.json"), "w") as fh:
        json.dump({"to_dense": {"train00": 0, "train01": 1, "train02": 2},
                   "to_eve": {"0": "train00", "1": "train01", "2": "train02"}}, fh)


def test_good_cache_passes(cache):
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    failed = [r for r in summary["checks"] if r["status"] == "FAIL"]
    assert failed == [], failed
    assert ok
    # The bundle-dependent checks skip, and a skip is never reported as a pass.
    assert _status(summary, "fixations_land_on_the_stimulus") == "skip"
    assert summary["n_skipped"] >= 2
    assert summary["cosine"]["within_name"]["median"] > \
        summary["cosine"]["cross_name"]["median"]


def test_missing_tensor_is_caught(cache):
    os.remove(os.path.join(cache["feat_dir"], cache["keys"][0] + ".pth"))
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "coverage_is_exact") == "FAIL"
    assert _status(summary, "no_phantom_keys") == "FAIL"


def test_truncated_tensor_is_caught(cache):
    victim = os.path.join(cache["feat_dir"], cache["keys"][0] + ".pth")
    with open(victim, "r+b") as fh:
        fh.truncate(MIN_FEATURE_BYTES - 1)
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "every_file_is_non_trivial") == "FAIL"


def test_stale_extra_file_is_caught(cache):
    torch.save(torch.zeros(FEATURE_SHAPE),
               os.path.join(cache["feat_dir"], "train99_step999.pth"))
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "no_phantom_keys") == "FAIL"


def test_dead_tensor_is_caught(cache):
    torch.save(torch.zeros(FEATURE_SHAPE, dtype=torch.float32),
               os.path.join(cache["feat_dir"], cache["keys"][0] + ".pth"))
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "no_dead_tensors") == "FAIL"


def test_negative_values_are_caught(cache):
    t = torch.full(FEATURE_SHAPE, -1.0, dtype=torch.float32)
    t[0, 0] = 1.0
    torch.save(t, os.path.join(cache["feat_dir"], cache["keys"][0] + ".pth"))
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "post_relu_non_negative") == "FAIL"


def test_collapsed_per_trial_keying_is_caught(cache, exp_keys):
    """The OPEN-6 regression: two participants' tensors identical again."""
    a = exp_keys[("shared_a.jpg", 0)]
    b = exp_keys[("shared_a.jpg", 1)]
    import shutil
    shutil.copyfile(os.path.join(cache["feat_dir"], a + ".pth"),
                    os.path.join(cache["feat_dir"], b + ".pth"))
    summary, ok = validate(cache["features"], cache["bridge_dir"], verify_sha=False)
    assert not ok
    detail = next(r["detail"] for r in summary["checks"]
                  if r["check"] == "per_trial_keying_did_something")
    assert "OPEN-6 is NOT resolved" in detail


def test_sha_mismatch_is_caught(cache):
    """FR8.2 -- the cache must be the one that was checked."""
    victim = os.path.join(cache["feat_dir"], cache["keys"][0] + ".pth")
    t = torch.load(victim, map_location="cpu")
    t[0, 0] += 1.0
    torch.save(t, victim)
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "feature_sha256_matches_disk") == "FAIL"

    summary2, _ = validate(cache["features"], cache["bridge_dir"], verify_sha=False)
    assert _status(summary2, "feature_sha256_matches_disk") == "skip"


def test_edited_fixations_are_caught(cache, bridge, fixations):
    """F4 is read-only with respect to F2; an edited fixations.json must show."""
    with open(bridge["fixations"], "w") as fh:
        json.dump(fixations, fh, indent=2)
    summary, ok = validate(cache["features"], cache["bridge_dir"])
    assert not ok
    assert _status(summary, "f2_artefacts_untouched") == "FAIL"


def test_main_writes_a_report_and_returns_exit_codes(cache, capsys):
    argv = ["--features", cache["features"], "--bridge-dir", cache["bridge_dir"]]
    assert main(argv) == 0
    capsys.readouterr()
    out = os.path.join(cache["features"], "validation_report.json")
    assert os.path.isfile(out)
    with open(out) as fh:
        written = json.load(fh)
    assert written["n_failed"] == 0 and written["n_tensors"] == len(cache["keys"])

    os.remove(os.path.join(cache["feat_dir"], cache["keys"][0] + ".pth"))
    assert main(argv) == 1
    capsys.readouterr()


def test_uninstalled_evedataset_skips_rather_than_fails(cache, monkeypatch, tmp_path):
    """A check that CANNOT RUN is not a check that failed.

    Observed for real on F4's first validation run: invoked from the `senet` env,
    where evedataset is not installed, two sound checks reported FAIL and the run
    read as broken when the cache was perfect.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "evedataset" or name.startswith("evedataset."):
            raise ImportError("No module named 'evedataset'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    staged = tmp_path / "bundle"
    staged.mkdir()
    summary, ok = validate(cache["features"], cache["bridge_dir"],
                           bundle_dir=str(staged), verify_sha=False)
    assert ok, [r for r in summary["checks"] if r["status"] == "FAIL"]
    for name in ("fixations_land_on_the_stimulus", "subject_identity_survives"):
        assert _status(summary, name) == "skip"
        detail = next(r["detail"] for r in summary["checks"] if r["check"] == name)
        assert "scanpath" in detail, "the skip must name the env that has it"


def test_absent_bundle_dir_skips(cache):
    summary, ok = validate(cache["features"], cache["bridge_dir"],
                           bundle_dir="/nonexistent/bundle", verify_sha=False)
    assert ok
    assert _status(summary, "subject_identity_survives") == "skip"


def test_f3_embedding_is_found_by_embedding_file(cache, tmp_path):
    """The report names the tensor in `embedding_file` -- not `embedding_path`."""
    import hashlib

    seed_dir = tmp_path / "eve_senet" / "seed0"
    seed_dir.mkdir(parents=True)
    emb = seed_dir / "eve_fewshot_user_embedding_10_seed0.pt"
    torch.save(torch.ones((38, 384), dtype=torch.float32), str(emb))
    with open(emb, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = seed_dir / "senet_report.json"
    with open(report, "w") as fh:
        json.dump({"embedding_file": emb.name, "embedding_sha256": sha}, fh)

    summary, ok = validate(cache["features"], cache["bridge_dir"],
                           senet_report=str(report), verify_sha=False)
    assert _status(summary, "f3_embedding_untouched") == "ok"

    # A changed tensor must FAIL, not skip -- this is F5's handshake with F3.
    torch.save(torch.zeros((38, 384), dtype=torch.float32), str(emb))
    summary2, ok2 = validate(cache["features"], cache["bridge_dir"],
                             senet_report=str(report), verify_sha=False)
    assert _status(summary2, "f3_embedding_untouched") == "FAIL"
    assert not ok2


def test_f3_embedding_found_by_glob_when_key_is_missing(cache, tmp_path):
    """A renamed key must not silently skip the handshake."""
    import hashlib

    seed_dir = tmp_path / "eve_senet" / "seed1"
    seed_dir.mkdir(parents=True)
    emb = seed_dir / "eve_fewshot_user_embedding_10_seed1.pt"
    torch.save(torch.ones((38, 384), dtype=torch.float32), str(emb))
    # Control arms must not confuse the fallback.
    torch.save(torch.ones(2), str(seed_dir / "eve_x_raw.pt"))
    torch.save(torch.ones(2), str(seed_dir / "eve_x_absurd.pt"))
    with open(emb, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = seed_dir / "senet_report.json"
    with open(report, "w") as fh:
        json.dump({"embedding_sha256": sha}, fh)      # no embedding_file key

    summary, _ = validate(cache["features"], cache["bridge_dir"],
                          senet_report=str(report), verify_sha=False)
    assert _status(summary, "f3_embedding_untouched") == "ok"
