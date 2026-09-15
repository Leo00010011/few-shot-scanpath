"""Re-run F2's heatmap bitwise-parity check in THIS env's numpy (FR3.10).

``tools/eve_bridge/heatmaps.py`` is a transcription of the construction inside
``ISP/OSIE/GazeformerISP/src/dataset/dataset.py::OSIE.__getitem__``, and F2 pinned the
two against each other **bitwise** -- but it did so under numpy 2.1.2 on the Windows
dev machine, which is not the stack the cluster run resolves to (TechStack section 1's
dev-machine reality check). Float32 promotion and structured-array behaviour are
precisely what the pin list exists for, so the check is re-run in the run's own
interpreter before ``gt_heatmaps.h5`` is trusted.

numpy + scipy only -- no torch, no h5py, no ISP import. The local transcription below
is deliberately a *third* copy rather than an import of either side: the point is to
compare two independent derivations, and importing one of them would compare a thing
to itself.
"""

import numpy as np
import scipy
import scipy.ndimage as filters

try:
    from . import EveEvalError, ACTION_MAP, MAX_LENGTH, ORIGIN_SIZE
except ImportError:  # executed as a script, not as a package member
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_eval import EveEvalError, ACTION_MAP, MAX_LENGTH, ORIGIN_SIZE


def _upstream_step_heatmaps(X, Y, length, origin_size=ORIGIN_SIZE,
                            action_map=ACTION_MAP, max_length=MAX_LENGTH,
                            blur_sigma=1):
    """A line-for-line transcription of ``OSIE.__getitem__``'s construction.

    The ``.astype(np.int32)`` **truncates**; it is not rounding, and it is load
    bearing -- a fixation at x = 119.9 lands in column 1, not column 2. Transcribed
    exactly, including the float32 cast of the coordinate arrays and the
    normalise-after-blur order.
    """
    downscale_x = origin_size[1] / action_map[1]
    downscale_y = origin_size[0] / action_map[0]

    scanpath = np.zeros((max_length, action_map[0], action_map[1]), dtype=np.float32)

    pos_x = np.array(X).astype(np.float32)
    pos_y = np.array(Y).astype(np.float32)

    pos_x_discrete = np.zeros(max_length, dtype=np.int32) - 1
    pos_y_discrete = np.zeros(max_length, dtype=np.int32) - 1
    for index in range(len(pos_x)):
        if index == max_length:
            break
        # since pixel is start from 1 ~ max based on matlab
        pos_x_discrete[index] = ((pos_x[index] - 1) / downscale_x).astype(np.int32)
        pos_y_discrete[index] = ((pos_y[index] - 1) / downscale_y).astype(np.int32)

    n = min(int(length), max_length)
    for index in range(max_length):
        if index >= n:
            continue
        if pos_x_discrete[index] == -1 or pos_y_discrete[index] == -1:
            continue
        scanpath[index, pos_y_discrete[index], pos_x_discrete[index]] = 1
        if blur_sigma:
            scanpath[index] = filters.gaussian_filter(scanpath[index], blur_sigma)
            scanpath[index] /= scanpath[index].sum()
    return scanpath


def _cases(n_cases, seed, origin_size, max_length):
    """Seeded (length, X, Y) cases, all strictly inside the stimulus (1-indexed)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(n_cases)):
        length = int(rng.randint(1, max_length + 4))   # includes over-length cases
        X = (rng.uniform(1.0, float(origin_size[1]), size=length)).tolist()
        Y = (rng.uniform(1.0, float(origin_size[0]), size=length)).tolist()
        out.append((length, X, Y))
    return out


def check_heatmap_parity(n_cases=200, seed=0, origin_size=ORIGIN_SIZE,
                         action_map=ACTION_MAP, max_length=MAX_LENGTH,
                         blur_sigma=1):
    """Bitwise-compare the bridge's construction against the upstream transcription.

    Returns a JSON-able dict on success. Raises :class:`EveEvalError` naming the first
    differing case, its length, its coordinates and ``np.abs(a - b).max()`` otherwise
    -- ``np.array_equal``, never ``allclose``: the whole point of the check is that a
    numpy version change must not move a single bit of the ground truth.
    """
    try:
        from eve_bridge.heatmaps import build_step_heatmaps
    except ImportError:  # pragma: no cover - exercised only outside the tools/ path
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eve_bridge.heatmaps import build_step_heatmaps

    for i, (length, X, Y) in enumerate(_cases(n_cases, seed, origin_size, max_length)):
        bridge, _mask = build_step_heatmaps(
            X, Y, length, origin_size=origin_size, action_map=action_map,
            max_length=max_length, blur_sigma=blur_sigma)
        upstream = _upstream_step_heatmaps(
            X, Y, length, origin_size=origin_size, action_map=action_map,
            max_length=max_length, blur_sigma=blur_sigma)
        if not np.array_equal(bridge, upstream):
            raise EveEvalError(
                "heatmap parity FAILED at case {} (FR3.10): length={}, "
                "X={}, Y={}, max|diff|={!r}. tools/eve_bridge/heatmaps.py and the "
                "upstream OSIE.__getitem__ construction do not agree bitwise under "
                "numpy {} / scipy {}; gt_heatmaps.h5 was built elsewhere and must "
                "not be trusted on this stack (TechStack section 1).".format(
                    i, length, X, Y, float(np.abs(bridge - upstream).max()),
                    np.__version__, scipy.__version__))

    return {
        "n_cases": int(n_cases),
        "seed": int(seed),
        "bitwise_equal": True,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
