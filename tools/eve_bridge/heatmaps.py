"""Ground-truth per-timestep fixation heatmaps (FR7).

Numerically identical to the construction inside
``ISP/OSIE/GazeformerISP/src/dataset/dataset.py::OSIE.__getitem__``. It is
re-derived here rather than imported because importing ``dataset.py`` drags in
torch, torchvision, skimage and matplotlib — none of which the bridge needs.
The parity test in ``tests/eve_bridge/test_heatmaps_parity.py`` pins the two
implementations against each other bitwise so this copy cannot drift silently.

numpy + scipy only.
"""

import numpy as np
import scipy.ndimage as filters


def build_step_heatmaps(X, Y, length, origin_size=(1080, 1920),
                        action_map=(24, 32), max_length=16, blur_sigma=1):
    """Build the (max_length, *action_map) ground-truth heatmap stack for one trial.

    Parameters
    ----------
    X, Y : sequence of float
        Fixation coordinates in original stimulus space, 1-indexed (FR4.2).
    length : int
        Ground-truth scanpath length; only ``min(length, max_length)`` steps are used.
    origin_size : (H, W)
    action_map : (rows, cols)
    max_length : int
    blur_sigma : float
        ``0`` / ``None`` disables the Gaussian blur and leaves a one-hot map.

    Returns
    -------
    (heatmaps, action_mask) : ((max_length, *action_map) f32, (max_length,) f32)
    """
    downscale_x = origin_size[1] / action_map[1]      # 60.0
    downscale_y = origin_size[0] / action_map[0]      # 45.0

    heatmaps = np.zeros((max_length, action_map[0], action_map[1]), dtype=np.float32)
    action_mask = np.zeros(max_length, dtype=np.float32)

    pos_x = np.array(X).astype(np.float32)
    pos_y = np.array(Y).astype(np.float32)
    n = min(int(length), max_length)

    for i in range(n):
        # since pixel is start from 1 ~ max based on matlab (truncation, as upstream)
        col = ((pos_x[i] - 1) / downscale_x).astype(np.int32)
        row = ((pos_y[i] - 1) / downscale_y).astype(np.int32)
        if not (0 <= row < action_map[0] and 0 <= col < action_map[1]):
            raise ValueError(
                "discretized fixation {} out of range: (row, col) = ({}, {}) "
                "for action_map {} from (X, Y) = ({}, {}) with origin_size {}".format(
                    i, int(row), int(col), action_map,
                    float(pos_x[i]), float(pos_y[i]), origin_size))
        heatmaps[i, row, col] = 1
        if blur_sigma:
            heatmaps[i] = filters.gaussian_filter(heatmaps[i], blur_sigma)
            heatmaps[i] /= heatmaps[i].sum()
        action_mask[i] = 1

    if action_mask.sum() <= max_length - 1:
        action_mask[int(action_mask.sum())] = 1

    return heatmaps, action_mask


def to_target_scanpath(heatmaps, action_mask):
    """Reconstruct the upstream ``target_scanpath`` layout (FR7.3).

    Column 0 is the termination flag (set where the step carries no fixation,
    matching upstream's ``pos_x_discrete[index] == -1`` branch); columns ``1:``
    are the flattened heatmap.
    """
    heatmaps = np.asarray(heatmaps)
    action_mask = np.asarray(action_mask)
    max_length = heatmaps.shape[0]
    if action_mask.shape != (max_length,):
        raise ValueError(
            "action_mask shape {} does not match heatmaps shape {}".format(
                action_mask.shape, heatmaps.shape))

    n_bins = heatmaps.shape[1] * heatmaps.shape[2]
    target_scanpath = np.zeros((max_length, n_bins + 1), dtype=np.float32)
    for index in range(max_length):
        if heatmaps[index].sum() == 0:
            target_scanpath[index, 0] = 1
        else:
            target_scanpath[index, 1:] = heatmaps[index].reshape(-1)
    return target_scanpath
