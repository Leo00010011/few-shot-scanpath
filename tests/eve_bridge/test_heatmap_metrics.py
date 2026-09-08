"""Validation Group 5 -- heatmap metrics (heatmap_metrics.py)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eve_bridge.heatmap_metrics import _loss_module, score_step_heatmaps
from eve_bridge.heatmaps import build_step_heatmaps

MAX_LENGTH = 16


def _gt_batch(lengths, seed=0):
    rng = np.random.RandomState(seed)
    maps = []
    for n in lengths:
        X = rng.uniform(1.0, 1920.0, size=max(n, 1))
        Y = rng.uniform(1.0, 1080.0, size=max(n, 1))
        hm, _ = build_step_heatmaps(X, Y, n)
        maps.append(hm)
    return torch.from_numpy(np.stack(maps))


def _pred_from_gt(gt):
    B, T = gt.shape[0], gt.shape[1]
    pred = torch.zeros(B, T, 24 * 32 + 1)
    pred[:, :, 1:] = gt.reshape(B, T, -1)
    return pred


def test_returns_plain_floats():
    lengths = [3, 5]
    gt = _gt_batch(lengths)
    out = score_step_heatmaps(_pred_from_gt(gt), gt, lengths)
    assert set(out) == {"NSS", "CC", "KLD"}
    assert all(type(v) is float for v in out.values())


def test_perfect_prediction():
    lengths = [4, 6]
    gt = _gt_batch(lengths, seed=1)
    perfect = score_step_heatmaps(_pred_from_gt(gt), gt, lengths)
    assert perfect["CC"] == pytest.approx(1.0, abs=1e-4)
    assert perfect["KLD"] == pytest.approx(0.0, abs=1e-4)

    rng = np.random.RandomState(11)
    random_pred = torch.zeros(gt.shape[0], MAX_LENGTH, 24 * 32 + 1)
    random_pred[:, :, 1:] = torch.from_numpy(
        rng.uniform(0.01, 1.0, size=(gt.shape[0], MAX_LENGTH, 768)).astype(np.float32))
    baseline = score_step_heatmaps(random_pred, gt, lengths)
    assert perfect["NSS"] > 0.0
    assert perfect["NSS"] > baseline["NSS"] + 1.0


def test_uniform_prediction_is_the_floor():
    """A perfectly flat prediction carries no information: CC must be 0.

    NSS is *not* well conditioned here. Upstream ``NSS`` standardises the
    prediction by ``(x - mean) / (std + 1e-7)``; on a constant float32 map the
    numerator is pure rounding residual (~1e-8) and the denominator is the bare
    epsilon, so the ratio is amplified to O(1) noise whose sign and magnitude
    depend on the constant. That is a property of the upstream metric, which
    FR10.3 forbids us to reimplement, so the assertion records the bound rather
    than pretending the value is 0. The meaningful NSS floor is the
    random-prediction case below.
    """
    lengths = [4, 6]
    gt = _gt_batch(lengths, seed=2)
    pred = torch.zeros(gt.shape[0], MAX_LENGTH, 24 * 32 + 1)
    pred[:, :, 1:] = 0.37
    out = score_step_heatmaps(pred, gt, lengths)
    assert out["CC"] == pytest.approx(0.0, abs=1e-4)
    assert abs(out["NSS"]) < 2.0


def test_random_prediction_is_the_nss_floor():
    lengths = [MAX_LENGTH] * 40
    gt = _gt_batch(lengths, seed=9)
    rng = np.random.RandomState(10)
    pred = torch.zeros(len(lengths), MAX_LENGTH, 24 * 32 + 1)
    pred[:, :, 1:] = torch.from_numpy(
        rng.uniform(0.01, 1.0, size=(len(lengths), MAX_LENGTH, 768)).astype(np.float32))
    out = score_step_heatmaps(pred, gt, lengths)
    assert out["NSS"] == pytest.approx(0.0, abs=0.05)
    assert out["CC"] == pytest.approx(0.0, abs=0.05)


def test_argument_order_is_prediction_first():
    lengths = [3]
    gt = _gt_batch(lengths, seed=3)
    rng = np.random.RandomState(4)
    noise = torch.from_numpy(
        rng.uniform(0.01, 1.0, size=(1, MAX_LENGTH, 24, 32)).astype(np.float32))
    pred = torch.zeros(1, MAX_LENGTH, 24 * 32 + 1)
    pred[:, :, 1:] = noise.reshape(1, MAX_LENGTH, -1)

    L = _loss_module()
    P = noise[0, :3]
    G = gt[0, :3]
    forward = float(L.KLD(P, G))
    backward = float(L.KLD(G, P))
    assert forward != pytest.approx(backward, abs=1e-6)
    assert score_step_heatmaps(pred, gt, lengths)["KLD"] == pytest.approx(forward,
                                                                         abs=1e-6)


def test_padding_steps_are_masked_out():
    lengths = [2, 16]
    gt = _gt_batch(lengths, seed=5)
    pred = _pred_from_gt(gt).clone()
    before = score_step_heatmaps(pred, gt, lengths)

    assert sum(min(v, MAX_LENGTH) for v in lengths) == 18

    pred2 = pred.clone()
    pred2[0, 2:, 1:] = torch.rand(MAX_LENGTH - 2, 24 * 32)
    gt2 = gt.clone()
    gt2[0, 2:] = 0.0
    after = score_step_heatmaps(pred2, gt2, lengths)
    for k in before:
        assert before[k] == pytest.approx(after[k], abs=1e-6)


def test_zero_valid_steps_raises():
    gt = _gt_batch([1, 1], seed=6)
    pred = _pred_from_gt(gt)
    with pytest.raises(ValueError) as exc:
        score_step_heatmaps(pred, gt, [0, 0])
    msg = str(exc.value)
    assert "M = 0" in msg
    assert "-1" not in msg


def test_shape_mismatch_names_all_three_shapes():
    gt = _gt_batch([3, 3, 3], seed=7)
    pred = torch.zeros(4, MAX_LENGTH, 24 * 32 + 1)
    with pytest.raises(ValueError) as exc:
        score_step_heatmaps(pred, gt, [3, 3, 3, 3])
    msg = str(exc.value)
    assert "(4, 16, 769)" in msg
    assert "(3, 16, 24, 32)" in msg
    assert "(4,)" in msg


def test_pred_without_termination_column_raises():
    gt = _gt_batch([3], seed=8)
    pred = torch.zeros(1, MAX_LENGTH, 24 * 32)
    with pytest.raises(ValueError):
        score_step_heatmaps(pred, gt, [3])


def test_metrics_come_from_the_on_disk_loss_module():
    path = _loss_module().__file__.replace("\\", "/")
    assert path.endswith("ISP/OSIE/GazeformerISP/src/models/loss.py")
