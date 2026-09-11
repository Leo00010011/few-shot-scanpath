"""EVE's own decile duration bins (FR3).

The released SE-Net OSIE checkpoint was trained with ``Data.fix_path =
"osie_fixations_update_duration.json"``, whose ``T`` is a **decile bin index 0-9**,
not milliseconds (TechStack section 3.7). EVE's ``fixations.json`` carries
milliseconds, so the bins are re-derived here over EVE's *own* support-split
distribution rather than borrowed from OSIE -- the two distributions differ (EVE's
minimum fixation is 100 ms against OSIE's 20 ms), so borrowing OSIE's edges would
quantise EVE durations against the wrong scale.

The duration channel is in fact **dead** in the released SE-Net -- ``src/models.py``
adds ``duration_encoding`` into ``ventral_pos`` and calls ``ventral_pos.fill_(0)`` on
the next line, after ``ventral_embs += ventral_pos`` has already happened (FR3.4). We
bin anyway: it is the input contract the checkpoint was trained under, it costs one
numpy call, and it is what a corrected SE-Net would need. What we do *not* do is
assume the deadness -- validation Group 5 pins it with a bitwise-identity check.

numpy only, so this module is unit-testable on the Windows dev machine (FR1.6).
"""

import os
import sys

import numpy as np

try:
    from . import EveSenetError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_senet import EveSenetError

N_BINS = 10


def decile_bins(durations):
    """Ten equal-count buckets over ``durations``. Returns ``(edges, bin_of)``.

    ``edges`` is ``(11,) float64`` -- ``np.quantile`` at ``q = i/10, i = 0..10``, so
    ``edges[0]`` is the minimum and ``edges[-1]`` the maximum. ``bin_of(t)`` returns
    ``int`` in ``{0..9}``.

    Ties at an edge collapse into one bucket, so bin counts need not be exactly
    equal; equality is deliberately **not** forced -- forcing it would have to break
    ties arbitrarily and would misassign identical durations to different bins.
    """
    d = np.asarray(list(durations), dtype=np.float64)
    if d.size == 0:
        raise EveSenetError("decile_bins: no durations to bin (FR3.2)")
    if not np.isfinite(d).all():
        raise EveSenetError("decile_bins: non-finite duration in the input (FR3.2)")
    edges = np.quantile(d, [i / N_BINS for i in range(N_BINS + 1)])
    if not np.all(np.diff(edges) >= 0):
        raise EveSenetError("decile_bins: edges are not monotone non-decreasing: {}".format(edges))

    inner = edges[1:-1]  # the 9 interior edges -- NOT the full 11, which would
    # give searchsorted an 11th bin and push the maximum duration to index 10.

    def bin_of(t):
        return int(np.searchsorted(inner, float(t), side="right"))

    return edges, bin_of


def bin_scanpath_durations(records, bin_of):
    """Copy each record with ``T`` (ms) replaced by its decile index ``0..9``.

    Never in place: the caller still needs the milliseconds for the report (FR3.3)
    and for the raw-ms arm of validation Group 5.
    """
    out = []
    for r in records:
        c = dict(r)
        c["T"] = [bin_of(t) for t in r["T"]]
        out.append(c)
    return out


def bin_occupancy(records_binned):
    """``{bin: count}`` over every fixation, for the report's occupancy check."""
    counts = {i: 0 for i in range(N_BINS)}
    for r in records_binned:
        for b in r["T"]:
            counts[int(b)] += 1
    return counts
