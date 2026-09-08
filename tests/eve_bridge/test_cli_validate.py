"""Validation Group 6 -- CLI and validator (build.py, validate.py)."""

import json
import os
import re
import subprocess

import h5py
import numpy as np
import pytest

from conftest import make_fake_bundle
from eve_bridge.build import parse_args, run
from eve_bridge.validate import BridgeValidationError, validate

EXPECTED_FILES = {"fixations.json", "subject_id_map.json", "bridge_report.json",
                  "gt_heatmaps.h5", "stimuli"}


def _args(out_dir, extra=()):
    return parse_args(["--bundle-dir", "fake-bundle", "--out-dir", str(out_dir),
                       "--support-pool-size", "2"] + list(extra))


@pytest.fixture
def built(tmp_path):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=21)
    out = tmp_path / "out"
    report = run(_args(out), bundle=bundle)
    return bundle, str(out), report


def test_cli_writes_exactly_the_expected_artefacts(built):
    _, out, _ = built
    assert set(os.listdir(out)) == EXPECTED_FILES
    stimuli = os.listdir(os.path.join(out, "stimuli"))
    assert len(stimuli) == 5
    assert all(f.endswith(".jpg") for f in stimuli)


def test_skip_stimuli_fails_validation(tmp_path):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=22)
    out = tmp_path / "out"
    with pytest.raises(BridgeValidationError) as exc:
        run(_args(out, ["--skip-stimuli"]), bundle=bundle)
    assert "FR9.1.8" in str(exc.value)
    assert not os.path.exists(os.path.join(str(out), "stimuli"))


def test_skip_stimuli_exits_non_zero(tmp_path, monkeypatch):
    from eve_bridge import build as build_mod
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=23)
    out = tmp_path / "out"
    original_run = build_mod.run
    monkeypatch.setattr(build_mod, "run", lambda args: original_run(args, bundle=bundle))
    rc = build_mod.main(["--bundle-dir", "fake", "--out-dir", str(out),
                         "--support-pool-size", "2", "--skip-stimuli"])
    assert rc == 1


def test_skip_heatmaps_omits_store_and_still_passes(tmp_path):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=24)
    out = tmp_path / "out"
    report = run(_args(out, ["--skip-heatmaps"]), bundle=bundle)
    assert not os.path.exists(os.path.join(str(out), "gt_heatmaps.h5"))
    assert report["validation"]["heatmap_store"].startswith("skipped")
    assert report["validation"]["heatmap_sampling"].startswith("skipped")


def test_report_carries_every_counter_at_zero(built):
    _, _, report = built
    for name in ("empty_scanpath", "non_finite", "no_stimulus", "incomplete_stimulus",
                 "clamped_coords", "zero_duration", "over_max_length",
                 "short_scanpath", "stimulus_image_conflict"):
        assert report["counters"][name] == 0, name
    assert report["origin_size"] == [1080, 1920]
    assert report["support_pool_size"] == 2
    assert report["num_subjects"] == 3
    assert report["num_stimuli_train"] == 2
    assert report["num_stimuli_test"] == 3
    assert report["num_trials"] == 15
    assert len(report["fixations_sha256"]) == 64


def test_short_scanpath_emits_a_d7_warning(tmp_path, capsys):
    lengths = {(s, k): (2 if (s, k) == (0, 0) else 5)
               for s in range(3) for k in range(5)}
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=25, lengths=lengths)
    out = tmp_path / "out"
    report = run(_args(out), bundle=bundle)
    assert report["counters"]["short_scanpath"] == 1
    assert "D7" in capsys.readouterr().err


def test_two_runs_are_byte_identical(tmp_path):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=26)
    a, b = tmp_path / "a", tmp_path / "b"
    run(_args(a), bundle=bundle)
    run(_args(b), bundle=bundle)
    with open(os.path.join(str(a), "fixations.json"), "rb") as fh:
        first = fh.read()
    with open(os.path.join(str(b), "fixations.json"), "rb") as fh:
        second = fh.read()
    assert first == second
    with h5py.File(os.path.join(str(a), "gt_heatmaps.h5"), "r") as fa, \
            h5py.File(os.path.join(str(b), "gt_heatmaps.h5"), "r") as fb:
        assert np.array_equal(fa["trials"]["heatmaps"][:], fb["trials"]["heatmaps"][:])


def test_no_absolute_paths_in_sources(repo_root):
    src_dir = os.path.join(repo_root, "tools", "eve_bridge")
    pattern = re.compile(r"[A-Za-z]:\\\\|/mnt/|/home/")
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".py"):
            continue
        with open(os.path.join(src_dir, fname)) as fh:
            for lineno, line in enumerate(fh, 1):
                assert not pattern.search(line), "{}:{}: {}".format(fname, lineno, line)


# ----------------------------------------------------------------------
# One negative test per FR9.1 check
# ----------------------------------------------------------------------

def _corrupt(out_dir, mutate):
    path = os.path.join(out_dir, "fixations.json")
    with open(path) as fh:
        fixations = json.load(fh)
    mutate(fixations)
    with open(path, "w") as fh:
        json.dump(fixations, fh, indent=4)
    return path


@pytest.mark.parametrize("marker,mutate", [
    ("FR9.1.1", lambda fx: fx[0].__setitem__("length", fx[0]["length"] + 1)),
    ("FR9.1.2", lambda fx: fx[0]["X"].__setitem__(0, 0.5)),
    ("FR9.1.3", lambda fx: fx[0]["T"].__setitem__(0, 0)),
    ("FR9.1.4", lambda fx: fx[0].__setitem__("subject", 99)),
    ("FR9.1.5", lambda fx: fx.pop(0)),
    ("FR9.1.6", lambda fx: fx[0].__setitem__("split", "validation")),
    ("FR9.1.7", lambda fx: fx[0].__setitem__("condition", "search")),
])
def test_validator_negative_cases(built, marker, mutate):
    _, out, _ = built
    _corrupt(out, mutate)
    with pytest.raises(BridgeValidationError) as exc:
        validate(out)
    assert marker in str(exc.value)


def test_validator_rejects_split_overlap(built):
    _, out, _ = built

    def mutate(fx):
        train_name = next(r["name"] for r in fx if r["split"] == "train")
        for r in fx:
            if r["name"] == train_name and r["subject"] == 0:
                r["split"] = "test"

    _corrupt(out, mutate)
    with pytest.raises(BridgeValidationError) as exc:
        validate(out)
    assert "FR3.3" in str(exc.value)


def test_validator_rejects_missing_stimulus_file(built):
    _, out, _ = built
    name = sorted(os.listdir(os.path.join(out, "stimuli")))[0]
    os.remove(os.path.join(out, "stimuli", name))
    with pytest.raises(BridgeValidationError) as exc:
        validate(out)
    assert "FR9.1.8" in str(exc.value)


def test_validator_rejects_jpg_in_stem(built):
    _, out, _ = built

    def mutate(fx):
        for r in fx:
            r["name"] = r["name"].replace("stim00", "jpgstim")

    _corrupt(out, mutate)
    for f in os.listdir(os.path.join(out, "stimuli")):
        if "stim00" in f:
            os.rename(os.path.join(out, "stimuli", f),
                      os.path.join(out, "stimuli", f.replace("stim00", "jpgstim")))
    with pytest.raises(BridgeValidationError) as exc:
        validate(out)
    assert "FR9.1.9" in str(exc.value)


def test_validator_rejects_a_stale_store(built):
    _, out, _ = built
    _corrupt(out, lambda fx: fx[0]["X"].__setitem__(0, fx[0]["X"][0] + 0.0001))
    with pytest.raises(BridgeValidationError) as exc:
        validate(out)
    assert "FR9.1.10" in str(exc.value)


def test_validator_rejects_a_tampered_heatmap(built):
    _, out, _ = built
    h5_path = os.path.join(out, "gt_heatmaps.h5")
    with h5py.File(h5_path, "r+") as f:
        hm = f["trials"]["heatmaps"]
        block = hm[0]
        block[0] = 0.0
        block[0, 5, 7] = 1.0
        hm[0] = block
    with pytest.raises(BridgeValidationError) as exc:
        validate(out)
    assert "FR9.1.11" in str(exc.value)


def test_validator_cross_checks_against_the_bundle(built):
    bundle, out, _ = built
    assert validate(out, bundle=bundle)["heatmap_sampling"] == "ok"

    _corrupt(out, lambda fx: None)   # rewrite unchanged -> hash still matches
    assert validate(out, bundle=bundle)["heatmap_sampling"] == "ok"
