"""Validation Group 2 -- bitwise parity with upstream ``OSIE.__getitem__``.

The reference arithmetic is transcribed verbatim from
``ISP/OSIE/GazeformerISP/src/dataset/dataset.py::OSIE.__getitem__`` rather than
imported, because importing ``dataset.py`` drags in torch, torchvision, skimage
and matplotlib. Any drift -- rounding instead of truncation, normalising before
blurring, a different filter mode -- fails here.
"""

import subprocess

import numpy as np
import pytest
import scipy.ndimage as filters

from eve_bridge.heatmaps import build_step_heatmaps

ACTION_MAP = (24, 32)
ORIGIN_SIZE = (1080, 1920)
MAX_LENGTH = 16
BLUR_SIGMA = 1


def upstream_reference(X, Y, max_length=MAX_LENGTH, action_map=ACTION_MAP,
                       origin_size=ORIGIN_SIZE, blur_sigma=BLUR_SIGMA):
    """Verbatim transcription of OSIE.__getitem__'s heatmap construction."""
    downscale_x = origin_size[1] / action_map[1]
    downscale_y = origin_size[0] / action_map[0]

    scanpath = np.zeros((max_length, action_map[0], action_map[1]), dtype=np.float32)
    action_mask = np.zeros(max_length, dtype=np.float32)

    pos_x = np.array(X).astype(np.float32)
    pos_y = np.array(Y).astype(np.float32)

    pos_x_discrete = np.zeros(max_length, dtype=np.int32) - 1
    pos_y_discrete = np.zeros(max_length, dtype=np.int32) - 1
    for index in range(len(pos_x)):
        if index == max_length:
            break
        pos_x_discrete[index] = ((pos_x[index] - 1) / downscale_x).astype(np.int32)
        pos_y_discrete[index] = ((pos_y[index] - 1) / downscale_y).astype(np.int32)
        action_mask[index] = 1
    if action_mask.sum() <= max_length - 1:
        action_mask[int(action_mask.sum())] = 1

    for index in range(max_length):
        if pos_x_discrete[index] == -1 or pos_y_discrete[index] == -1:
            continue
        scanpath[index, pos_y_discrete[index], pos_x_discrete[index]] = 1
        if blur_sigma:
            scanpath[index] = filters.gaussian_filter(scanpath[index], blur_sigma)
            scanpath[index] /= scanpath[index].sum()

    return scanpath, action_mask


def _cases(n=200, seed=1234):
    rng = np.random.RandomState(seed)
    for _ in range(n):
        length = int(rng.randint(1, 21))
        X = rng.uniform(1.0, 1920.0, size=length)
        Y = rng.uniform(1.0, 1080.0, size=length)
        yield length, X, Y


def test_heatmaps_are_bitwise_equal_to_upstream():
    for length, X, Y in _cases():
        ours, _ = build_step_heatmaps(X, Y, length, origin_size=ORIGIN_SIZE,
                                      action_map=ACTION_MAP, max_length=MAX_LENGTH,
                                      blur_sigma=BLUR_SIGMA)
        theirs, _ = upstream_reference(X, Y)
        assert np.array_equal(ours, theirs), "drift at length {}".format(length)


def test_action_masks_are_bitwise_equal_to_upstream():
    for length, X, Y in _cases():
        _, ours = build_step_heatmaps(X, Y, length, origin_size=ORIGIN_SIZE,
                                      action_map=ACTION_MAP, max_length=MAX_LENGTH,
                                      blur_sigma=BLUR_SIGMA)
        _, theirs = upstream_reference(X, Y)
        assert np.array_equal(ours, theirs), "mask drift at length {}".format(length)


def test_frozen_files_are_untouched(repo_root):
    paths = ["ISP/OSIE/GazeformerISP/src/utils/",
             "ISP/OSIE/GazeformerISP/src/models/loss.py"]
    proc = subprocess.run(["git", "diff", "--exit-code", "--"] + paths,
                          cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail("frozen files modified (D1, FR10.3):\n{}".format(proc.stdout))
