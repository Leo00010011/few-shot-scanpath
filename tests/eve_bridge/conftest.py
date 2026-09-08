"""Fixtures for the EVE bridge tests.

The fake bundle exposes exactly the three members the bridge is allowed to use --
``samples_df``, ``get_scanpath()``, ``get_stimulus()`` -- so the whole feature is
testable without the real dataset. Tests marked ``bundle`` need ``--bundle-dir``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))


def pytest_addoption(parser):
    parser.addoption("--bundle-dir", action="store", default=None,
                     help="directory containing a real EVE bundle.h5")
    parser.addoption("--bridge-subjects", action="store", default=None,
                     help="comma-separated EVE participant ids for the [bundle] runs; "
                          "default is the bridge's own default (every 'test*' id)")
    parser.addoption("--bridge-support-pool-size", action="store", type=int, default=20,
                     help="support_pool_size for the [bundle] runs")


def pytest_configure(config):
    config.addinivalue_line("markers", "bundle: needs a real EVE bundle (--bundle-dir)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--bundle-dir"):
        return
    skip = pytest.mark.skip(reason="needs --bundle-dir")
    for item in items:
        if "bundle" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def real_bundle(request):
    bundle_dir = request.config.getoption("--bundle-dir")
    if not bundle_dir:
        pytest.skip("needs --bundle-dir")
    from evedataset import EveBundle
    return EveBundle.load(bundle_dir)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


class FakeBundle(object):
    """Minimal stand-in for ``evedataset.EveBundle``."""

    def __init__(self, rows, scanpaths, stimuli=None):
        self._df = pd.DataFrame(rows, columns=[
            "exp_key", "subject", "stimulus_name", "split", "valid", "stimulus_path"])
        self._scanpaths = scanpaths
        self._stimuli = stimuli or {}

    @property
    def samples_df(self):
        return self._df.copy()

    def get_scanpath(self, exp_key):
        return np.asarray(self._scanpaths[exp_key], dtype=np.float32)

    def get_stimulus(self, exp_key):
        if exp_key in self._stimuli:
            return self._stimuli[exp_key]
        name = self._df.set_index("exp_key").loc[exp_key, "stimulus_name"]
        seed = abs(hash(name)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        return rng.randint(0, 256, size=(12, 16, 3)).astype(np.uint8)


def make_scanpath(n, rng, duration=(150.0, 800.0)):
    """Return a (4, n) float32 EVE scanpath: [t_sec, duration_ms, x_px, y_px]."""
    return np.stack([
        np.cumsum(rng.uniform(0.1, 0.4, size=n)),
        rng.uniform(duration[0], duration[1], size=n),
        rng.uniform(0.0, 1919.0, size=n),
        rng.uniform(0.0, 1079.0, size=n),
    ]).astype(np.float32)


def make_fake_bundle(n_subjects=3, n_stimuli=5, seed=0, lengths=None):
    """Fully-crossed fake bundle: every subject saw every stimulus, all valid."""
    rng = np.random.RandomState(seed)
    rows, scanpaths = [], {}
    for si in range(n_subjects):
        subject = "test{:02d}".format(si + 1)
        for k in range(n_stimuli):
            stimulus = "stim{:02d}".format(k)
            exp_key = "{}_{}".format(subject, k)
            rows.append((exp_key, subject, stimulus, "test", True,
                         "stimuli/{}.png".format(stimulus)))
            n = 5 if lengths is None else lengths[(si, k)]
            scanpaths[exp_key] = make_scanpath(n, rng)
    return FakeBundle(rows, scanpaths)


@pytest.fixture
def fake_bundle():
    return make_fake_bundle()
