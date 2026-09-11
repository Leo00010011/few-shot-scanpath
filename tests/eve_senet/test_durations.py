"""Validation Group 1 -- duration binning (FR3)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from eve_senet import EveSenetError
from eve_senet.durations import bin_occupancy, bin_scanpath_durations, decile_bins

from conftest import BRIDGE_DIR, make_record

REAL_EDGES = [100, 131, 151, 169, 188, 210, 234, 263, 308, 390, 1144]


def test_edges_shape_dtype_and_monotone():
    edges, _ = decile_bins(range(100))
    assert edges.shape == (11,)
    assert edges.dtype == np.float64
    # a decreasing edge means the quantile call was mis-parameterised
    assert np.all(np.diff(edges) >= 0)


def test_uniform_input_fills_ten_bins_evenly():
    edges, bin_of = decile_bins(range(1000))
    bins = [bin_of(t) for t in range(1000)]
    # a value outside {0..9} means searchsorted got the full edges rather than
    # edges[1:-1] -- an off-by-one that silently creates an 11th bin
    assert set(bins) == set(range(10))
    counts = [bins.count(b) for b in range(10)]
    assert all(abs(c - 100) <= 1 for c in counts), counts


def test_boundary_values_land_in_the_end_bins():
    edges, bin_of = decile_bins(range(1000))
    assert bin_of(edges[0]) == 0
    assert bin_of(edges[-1]) == 9  # the maximum must not overflow into an 11th bin


def test_tied_input_collapses_into_one_bin_without_raising():
    edges, bin_of = decile_bins([5] * 100)
    # equal counts are deliberately NOT forced: forcing them would have to break ties
    # arbitrarily and would assign identical durations to different bins
    assert len({bin_of(5) for _ in range(100)}) == 1
    assert np.all(edges == 5)


def test_empty_input_raises_our_error_not_numpys():
    with pytest.raises(EveSenetError):
        decile_bins([])


def test_non_finite_input_raises():
    with pytest.raises(EveSenetError):
        decile_bins([1.0, float("nan"), 3.0])


def test_bin_scanpath_durations_returns_copies():
    recs = [make_record("a.jpg", 0, [1.0, 2.0], [3.0, 4.0], [100, 200])]
    original = list(recs[0]["T"])
    _, bin_of = decile_bins([100, 200, 300])
    out = bin_scanpath_durations(recs, bin_of)
    assert out[0] is not recs[0]
    assert out[0]["T"] is not recs[0]["T"]
    assert recs[0]["T"] == original  # the caller still needs the milliseconds
    assert all(0 <= b <= 9 for b in out[0]["T"])


def test_bin_occupancy_counts_every_fixation():
    recs = [make_record("a.jpg", 0, [1.0] * 3, [1.0] * 3, [100, 200, 300])]
    _, bin_of = decile_bins([100, 200, 300])
    occ = bin_occupancy(bin_scanpath_durations(recs, bin_of))
    assert sorted(occ) == list(range(10))
    assert sum(occ.values()) == 3


@pytest.mark.bridge
def test_real_support_split_edges_match_the_recorded_run():
    """FR3.3 -- computed from fixations.json, never hardcoded in the implementation."""
    with open(os.path.join(BRIDGE_DIR, "fixations.json")) as fh:
        recs = json.load(fh)
    train = [r for r in recs if r["split"] == "train"]
    edges, _ = decile_bins([t for r in train for t in r["T"]])
    assert np.allclose(edges, REAL_EDGES, atol=1e-9)


def test_edges_are_not_hardcoded_in_the_source():
    """The realised edges are a *result*; finding them in the code means they are an
    input, and a future bridge run would be quantised against a stale distribution."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "tools", "eve_senet", "durations.py")).read()
    assert "1144" not in src and "263" not in src


def test_draw_is_independent_of_pythonhashseed():
    """Both the bins and the selection must survive a different interpreter hash seed."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from eve_senet.durations import decile_bins;"
        "e, f = decile_bins(range(1000));"
        "import json; print(json.dumps([float(x) for x in e]))"
    ) % os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "tools")
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    assert json.loads(out) == [float(x) for x in decile_bins(range(1000))[0]]
