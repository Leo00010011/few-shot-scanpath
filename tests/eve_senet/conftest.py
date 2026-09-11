"""Fixtures for the F3 subject-embedding tests.

Self-contained: synthetic ``fixations.json`` records and a stub stimulus directory, no
cluster data and no GPU, mirroring ``tests/osie_prep/``. Validation Groups 1, 2, 3 and
6 live here; Groups 4, 5 and Data Validity need the ``senet`` env and run on the
cluster.

Tests marked ``bridge`` read the real ``data/eve_bridge/`` artefacts and skip when
they are absent (they are git-ignored, so a fresh checkout has none).
"""

import os
import sys

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(REPO_ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

BRIDGE_DIR = os.path.join(REPO_ROOT, "data", "eve_bridge")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "bridge: needs the real data/eve_bridge/ artefacts")


def pytest_collection_modifyitems(config, items):
    if os.path.isfile(os.path.join(BRIDGE_DIR, "fixations.json")):
        return
    skip = pytest.mark.skip(reason="needs data/eve_bridge/ (git-ignored)")
    for item in items:
        if "bridge" in item.keywords:
            item.add_marker(skip)


def make_record(name, subject, xs, ys, ts, split="train"):
    """One ``fixations.json`` record (TechStack section 3.1) in EVE's 1920x1080 space."""
    return {"name": name, "subject": subject,
            "X": list(xs), "Y": list(ys), "T": list(ts),
            "length": len(xs), "split": split,
            "condition": "freeview", "task": "none"}


@pytest.fixture
def records():
    """38 subjects x 20 support scanpaths, stimulus-disjoint per subject (as F2's are)."""
    out = []
    for s in range(38):
        for k in range(20):
            n = 3 + (k % 5)
            out.append(make_record(
                "s{:02d}_img{:02d}.jpg".format(s, k), s,
                [100.0 + 37 * i + s for i in range(n)],
                [80.0 + 29 * i + k for i in range(n)],
                [150 + 11 * i for i in range(n)]))
    return out


@pytest.fixture
def stub_images(tmp_path, records):
    """A 4x4 RGB .jpg per stimulus name -- enough for the existence check (FR11.3)."""
    d = tmp_path / "stimuli"
    d.mkdir()
    for name in sorted({r["name"] for r in records}):
        Image.new("RGB", (4, 4), (10, 20, 30)).save(d / name)
    return str(d)
