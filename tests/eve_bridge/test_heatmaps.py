"""Validation Group 1 -- heatmap construction (heatmaps.py)."""

import numpy as np
import pytest

from eve_bridge.heatmaps import build_step_heatmaps, to_target_scanpath


def test_shapes_and_dtypes():
    hm, mask = build_step_heatmaps([61.0], [46.0], 1)
    assert hm.shape == (16, 24, 32)
    assert hm.dtype == np.float32
    assert mask.shape == (16,)
    assert mask.dtype == np.float32


def test_argmax_bin_respects_the_one_indexing():
    # col = int(60 / 60) = 1, row = int(45 / 45) = 1
    hm, _ = build_step_heatmaps([61.0], [46.0], 1)
    assert np.unravel_index(int(hm[0].argmax()), hm[0].shape) == (1, 1)


def test_valid_steps_sum_to_one_and_padding_is_zero():
    rng = np.random.RandomState(0)
    X = list(rng.uniform(1.0, 1920.0, size=5))
    Y = list(rng.uniform(1.0, 1080.0, size=5))
    hm, _ = build_step_heatmaps(X, Y, 5)
    for t in range(5):
        assert hm[t].sum() == pytest.approx(1.0, abs=1e-5)
    assert hm[5:].sum() == 0.0


def test_action_mask_termination_slot():
    _, mask = build_step_heatmaps([61.0] * 5, [46.0] * 5, 5)
    assert list(mask) == [1.0] * 5 + [1.0] + [0.0] * 10


def test_action_mask_full_length_has_no_extra_slot():
    _, mask = build_step_heatmaps([61.0] * 16, [46.0] * 16, 16)
    assert list(mask) == [1.0] * 16


def test_blur_sigma_zero_is_one_hot():
    hm, _ = build_step_heatmaps([61.0], [46.0], 1, blur_sigma=0)
    assert hm[0].sum() == 1.0
    assert int((hm[0] == 1.0).sum()) == 1
    assert int((hm[0] != 0.0).sum()) == 1


def test_boundary_coordinates():
    hm, _ = build_step_heatmaps([1.0, 1920.0], [1.0, 1080.0], 2, blur_sigma=0)
    assert np.unravel_index(int(hm[0].argmax()), hm[0].shape) == (0, 0)
    assert np.unravel_index(int(hm[1].argmax()), hm[1].shape) == (23, 31)


def test_out_of_range_raises_rather_than_clipping():
    with pytest.raises(ValueError):
        build_step_heatmaps([1921.0], [1.0], 1)


def test_length_beyond_max_length_is_truncated():
    rng = np.random.RandomState(1)
    X = list(rng.uniform(1.0, 1920.0, size=20))
    Y = list(rng.uniform(1.0, 1080.0, size=20))
    hm, mask = build_step_heatmaps(X, Y, 20)
    assert int((hm.sum(axis=(1, 2)) > 0).sum()) == 16
    assert list(mask) == [1.0] * 16


def test_to_target_scanpath_layout():
    rng = np.random.RandomState(2)
    X = list(rng.uniform(1.0, 1920.0, size=5))
    Y = list(rng.uniform(1.0, 1080.0, size=5))
    hm, mask = build_step_heatmaps(X, Y, 5)
    out = to_target_scanpath(hm, mask)
    assert out.shape == (16, 769)
    assert out.dtype == np.float32
    assert np.all(out[:5, 0] == 0.0)
    assert np.all(out[5:, 0] == 1.0)
    assert np.allclose(out[2, 1:], hm[2].reshape(-1))
