"""NSS / CC / KLD over the model's per-step action-probability maps (FR10).

The three metrics come from ``ISP/OSIE/GazeformerISP/src/models/loss.py``, loaded
by path via ``importlib`` -- not copied, not reimplemented -- so the numbers are the
authors' and not ours. The only thing this wrapper adds is the masking of padding
steps (FR10.2), which is a call-site concern rather than a metric change.

This is the only module in the bridge that imports torch.
"""

import importlib.util
from pathlib import Path

import torch

_LOSS_MODULE = None

LOSS_REL_PATH = ("ISP", "OSIE", "GazeformerISP", "src", "models", "loss.py")


def _loss_module():
    """Load ``models/loss.py`` by path, without importing the ISP branch."""
    global _LOSS_MODULE
    if _LOSS_MODULE is None:
        root = Path(__file__).resolve().parents[2]
        path = root.joinpath(*LOSS_REL_PATH)
        if not path.is_file():
            raise FileNotFoundError(
                "cannot locate the upstream loss module at {}".format(path))
        spec = importlib.util.spec_from_file_location("isp_osie_loss", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOSS_MODULE = module
    return _LOSS_MODULE


def score_step_heatmaps(pred_action_prob, gt_heatmaps, lengths, max_length=16):
    """Score the model's action maps against the ground-truth step heatmaps.

    Parameters
    ----------
    pred_action_prob : torch.Tensor ``(B, max_length, rows*cols + 1)``
        ``all_actions_prob`` exactly as the model returns it; column 0 is the
        termination flag and is dropped here.
    gt_heatmaps : torch.Tensor ``(B, max_length, rows, cols)``
        From :meth:`GtHeatmapStore.get_batch`.
    lengths : sequence of int ``(B,)``
        Ground-truth scanpath lengths.

    Returns
    -------
    dict with keys ``NSS``, ``CC``, ``KLD`` -- each a mean over the ``M`` valid
    *timesteps*, not over trials. That is a different denominator from the
    scanpath metrics, which average over (image, subject) cells.
    """
    lengths = [int(v) for v in lengths]

    if pred_action_prob.dim() != 3 or gt_heatmaps.dim() != 4:
        raise ValueError(
            "expected pred_action_prob (B, T, bins+1) and gt_heatmaps (B, T, rows, "
            "cols); got pred {}, gt {}, lengths ({},)".format(
                tuple(pred_action_prob.shape), tuple(gt_heatmaps.shape), len(lengths)))

    B, T, C = pred_action_prob.shape
    rows, cols = gt_heatmaps.shape[2], gt_heatmaps.shape[3]

    if (gt_heatmaps.shape[0] != B or gt_heatmaps.shape[1] != T
            or len(lengths) != B or C != rows * cols + 1 or T != max_length):
        raise ValueError(
            "shape mismatch: pred_action_prob {}, gt_heatmaps {}, lengths ({},); "
            "expected pred (B, {}, {}) and gt (B, {}, {}, {}) with len(lengths) == "
            "B".format(tuple(pred_action_prob.shape), tuple(gt_heatmaps.shape),
                       len(lengths), max_length, rows * cols + 1, max_length,
                       rows, cols))

    pred = pred_action_prob[:, :, 1:].reshape(B, T, rows, cols)

    valid = [(b, t) for b in range(B) for t in range(min(lengths[b], max_length))]
    if not valid:
        raise ValueError(
            "no valid timesteps to score: lengths {} yield M = 0 (FR10.4)".format(
                lengths))

    index_b = torch.tensor([b for b, _ in valid], dtype=torch.long,
                           device=pred.device)
    index_t = torch.tensor([t for _, t in valid], dtype=torch.long,
                           device=pred.device)
    P = pred[index_b, index_t]                                  # (M, rows, cols)
    G = gt_heatmaps[index_b.to(gt_heatmaps.device),
                    index_t.to(gt_heatmaps.device)].to(P.device)

    L = _loss_module()
    # Argument order is prediction first, ground truth second in all three.
    return {
        "NSS": float(L.NSS(P, G)),
        "CC": float(L.CC(P, G)),
        "KLD": float(L.KLD(P, G)),
    }
