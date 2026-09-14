"""Fixtures for the F4 per-trial feature tests.

Self-contained: synthetic ``fixations.json`` records, a synthetic ``gt_heatmaps.h5``
carrying only the two string datasets ``load_trial_exp_keys`` reads, a fake
``samples_df`` and a ``subject_id_map.json``. No cluster data and no GPU, mirroring
``tests/eve_senet/`` and ``tests/osie_prep/``.

Markers (FR14.2):

* ``bundle`` -- needs the real EVE bundle; skipped unless ``--bundle-dir`` is passed.
* ``bridge`` -- reads the real ``data/eve_bridge/`` artefacts; skipped when absent
  (that directory is git-ignored, so a fresh checkout has none).

The FR4.4 bit-identity test is deliberately **unmarked**: it needs only a synthetic
image and a CPU backbone, and it is the single most load-bearing assertion in F4.
"""

import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(REPO_ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

BRIDGE_DIR = os.path.join(REPO_ROOT, "data", "eve_bridge")
VLEN_STR = h5py.special_dtype(vlen=str)


def pytest_addoption(parser):
    parser.addoption("--bundle-dir", action="store", default=None,
                     help="path to the EVE bundle directory (enables 'bundle' tests)")


def pytest_configure(config):
    config.addinivalue_line("markers", "bundle: needs the real EVE bundle")
    config.addinivalue_line("markers", "bridge: needs the real data/eve_bridge/ artefacts")


def pytest_collection_modifyitems(config, items):
    no_bundle = config.getoption("--bundle-dir") is None
    no_bridge = not os.path.isfile(os.path.join(BRIDGE_DIR, "fixations.json"))
    skip_bundle = pytest.mark.skip(reason="needs --bundle-dir")
    skip_bridge = pytest.mark.skip(reason="needs data/eve_bridge/ (git-ignored)")
    for item in items:
        if no_bundle and "bundle" in item.keywords:
            item.add_marker(skip_bundle)
        if no_bridge and "bridge" in item.keywords:
            item.add_marker(skip_bridge)


@pytest.fixture
def bundle_dir(request):
    path = request.config.getoption("--bundle-dir")
    if path is None:
        pytest.skip("needs --bundle-dir")
    return path


@pytest.fixture
def bridge_dir():
    if not os.path.isfile(os.path.join(BRIDGE_DIR, "fixations.json")):
        pytest.skip("needs data/eve_bridge/")
    return BRIDGE_DIR


def make_record(name, subject, split="test", n=4):
    """One ``fixations.json`` record (TechStack section 3.1) in EVE's 1920x1080 space."""
    return {"name": name, "subject": subject,
            "X": [100.0 + 37 * i for i in range(n)],
            "Y": [80.0 + 29 * i for i in range(n)],
            "T": [150 + 11 * i for i in range(n)],
            "length": n, "split": split,
            "condition": "freeview", "task": "none"}


#: Six synthetic trials: two stimuli seen by three subjects on the scored split (the
#: uniform-count shape F2 built), plus two subject-private support stimuli.
TRIALS = [
    ("shared_a.jpg", 0, "test", "train00_step001"),
    ("shared_a.jpg", 1, "test", "train01_step002"),
    ("shared_a.jpg", 2, "test", "train02_step003"),
    ("shared_b.jpg", 0, "test", "train00_step004"),
    ("shared_b.jpg", 1, "test", "train01_step005"),
    ("shared_b.jpg", 2, "test", "train02_step006"),
    ("support_x.jpg", 0, "train", "train00_step007"),
    ("support_y.jpg", 1, "train", "train01_step008"),
]

EVE_IDS = {0: "train00", 1: "train01", 2: "train02"}


@pytest.fixture
def fixations():
    return [make_record(name, subject, split)
            for name, subject, split, _ in TRIALS]


@pytest.fixture
def exp_keys():
    """The expected authoritative mapping, as a plain dict."""
    return {(name, subject): key for name, subject, _, key in TRIALS}


@pytest.fixture
def subject_id_map():
    return {"to_dense": {v: k for k, v in EVE_IDS.items()},
            "to_eve": {str(k): v for k, v in EVE_IDS.items()}}


@pytest.fixture
def samples_df():
    """The six ``samples_df`` columns the derivation reads (FR2.4)."""
    rows = []
    for name, subject, split, key in TRIALS:
        rows.append({"exp_key": key, "subject": EVE_IDS[subject],
                     "stimulus_name": name[:-4], "split": split,
                     "valid": True, "stimulus_path": "stimuli/{}.png".format(key)})
    # A row for a participant outside the cohort: the derivation must not match it.
    rows.append({"exp_key": "test99_step000", "subject": "test99",
                 "stimulus_name": "shared_a", "split": "test",
                 "valid": False, "stimulus_path": "stimuli/test99_step000.png"})
    return pd.DataFrame(rows)


def write_fixations(path, fixations):
    with open(path, "w") as fh:
        json.dump(fixations, fh)
    return str(path)


def write_store(path, fixations, exp_keys, fixations_path=None, overrides=None):
    """A minimal ``gt_heatmaps.h5`` -- the two string datasets plus the sha attr.

    ``overrides`` replaces the exp_key list wholesale, so a test can inject a
    duplicate without going through the bridge.
    """
    import hashlib

    trial_keys = ["{}|{}".format(r["name"], r["subject"]) for r in fixations]
    keys = overrides if overrides is not None else [
        exp_keys[(r["name"], int(r["subject"]))] for r in fixations]

    sha = ""
    if fixations_path is not None:
        with open(fixations_path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()

    with h5py.File(path, "w") as f:
        f.attrs["fixations_sha256"] = sha
        grp = f.create_group("trials")
        grp.create_dataset("trial_key", data=np.array(trial_keys, dtype=object),
                           dtype=VLEN_STR)
        grp.create_dataset("exp_key", data=np.array(keys, dtype=object),
                           dtype=VLEN_STR)
        grp.create_dataset("subject",
                           data=np.array([int(r["subject"]) for r in fixations],
                                         dtype=np.int32))
        # Present so the test store is shaped like the real one; never read by F4.
        grp.create_dataset("heatmaps",
                           data=np.zeros((len(fixations), 2, 2, 2), dtype=np.float32))
    return str(path)


@pytest.fixture
def bridge(tmp_path, fixations, exp_keys):
    """A synthetic bridge directory: fixations.json + gt_heatmaps.h5 + the map."""
    fix_path = write_fixations(tmp_path / "fixations.json", fixations)
    store_path = write_store(tmp_path / "gt_heatmaps.h5", fixations, exp_keys,
                             fixations_path=fix_path)
    return {"dir": str(tmp_path), "fixations": fix_path, "heatmaps": store_path}


@pytest.fixture
def feature_dir(tmp_path, exp_keys):
    """A complete synthetic feature cache: one (768, 2048) float32 tensor per trial."""
    import torch

    from eve_prep import FEATURE_SHAPE

    d = tmp_path / "features" / "image_features"
    d.mkdir(parents=True)
    for key in sorted(exp_keys.values()):
        torch.save(torch.zeros(FEATURE_SHAPE, dtype=torch.float32),
                   str(d / (key + ".pth")))
    return str(d)
