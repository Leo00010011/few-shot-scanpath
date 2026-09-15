"""Group 5 -- wiring of the supplementary NSS / CC / KLD block.

The metrics themselves are the authors', loaded by path from ``models/loss.py``; what
F5 adds is the alignment between ``all_actions_prob`` row *k* and ground-truth row
*k*, and the weighting that makes the reported value a mean over valid timesteps
rather than a mean of batch means. Both are silent when wrong.
"""

import ast
import json
import os
import shutil

import numpy as np
import pytest
import torch

from conftest import EVE_SRC, REPO_ROOT, TEST_PY, TOOLS

MAX_LENGTH = 16
ROWS, COLS = 24, 32


def _pred(n, seed=0):
    """A plausible ``all_actions_prob``: (N, 16, rows*cols + 1), rows summing to 1."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, MAX_LENGTH, ROWS * COLS + 1), generator=g)
    return x / x.sum(-1, keepdim=True)


def test_metrics_are_not_reimplemented_anywhere():
    """FR8.3 -- the three functions must stay the authors' own."""
    targets = [os.path.join(EVE_SRC, "test.py"),
               os.path.join(EVE_SRC, "dataset", "dataset.py")]
    for base, dirs, files in os.walk(os.path.join(TOOLS, "eve_eval")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        targets.extend(os.path.join(base, f) for f in files if f.endswith(".py"))

    for path in targets:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        defined = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
        assert not defined & {"NSS", "CC", "KLD"}, (path, defined)

    with open(TEST_PY, encoding="utf-8") as fh:
        source = fh.read()
    assert "from eve_bridge.heatmap_metrics import score_step_heatmaps" in source


def test_get_batch_is_positional_and_ordered(artefacts, exp_key_map, fixations):
    """FR8.2 -- row k of the returned array is the store row for flat_keys[k]."""
    from eve_bridge.store import GtHeatmapStore

    store = GtHeatmapStore.load(artefacts["heatmaps"],
                                fixations_path=artefacts["fixations"])
    keys = ["{}|{}".format(r["name"], r["subject"]) for r in fixations[:6]]
    batch = store.get_batch(keys)
    assert batch.shape == (6, MAX_LENGTH, ROWS, COLS)
    assert batch.dtype == np.float32
    for k, key in enumerate(keys):
        name, subject = key.rsplit("|", 1)
        assert np.array_equal(batch[k], store.get(name, int(subject)))


def test_permuted_keys_change_the_scores(artefacts, fixations):
    """A silent misalignment must not be able to pass unnoticed."""
    from eve_bridge.heatmap_metrics import score_step_heatmaps
    from eve_bridge.store import GtHeatmapStore

    store = GtHeatmapStore.load(artefacts["heatmaps"],
                                fixations_path=artefacts["fixations"])
    keys = ["{}|{}".format(r["name"], r["subject"]) for r in fixations[:6]]
    lengths = [min(r["length"], MAX_LENGTH) for r in fixations[:6]]
    pred = _pred(6)

    ordered = score_step_heatmaps(pred, torch.from_numpy(store.get_batch(keys)),
                                  lengths, max_length=MAX_LENGTH)
    permuted_keys = keys[::-1]
    shuffled = score_step_heatmaps(
        pred, torch.from_numpy(store.get_batch(permuted_keys)), lengths,
        max_length=MAX_LENGTH)
    assert ordered != shuffled


def test_row_count_mismatch_raises(artefacts, fixations):
    from eve_bridge.heatmap_metrics import score_step_heatmaps
    from eve_bridge.store import GtHeatmapStore

    store = GtHeatmapStore.load(artefacts["heatmaps"],
                                fixations_path=artefacts["fixations"])
    keys = ["{}|{}".format(r["name"], r["subject"]) for r in fixations[:6]]
    gt = torch.from_numpy(store.get_batch(keys))
    with pytest.raises(ValueError) as exc:
        score_step_heatmaps(_pred(5), gt, [4] * 5, max_length=MAX_LENGTH)
    assert "shape mismatch" in str(exc.value) or "expected" in str(exc.value)


def test_argument_order_is_prediction_first(artefacts, fixations):
    """FR8.3 -- KLD is not symmetric; reversing it yields a plausible wrong number."""
    from eve_bridge.heatmap_metrics import score_step_heatmaps
    from eve_bridge.store import GtHeatmapStore

    store = GtHeatmapStore.load(artefacts["heatmaps"],
                                fixations_path=artefacts["fixations"])
    keys = ["{}|{}".format(r["name"], r["subject"]) for r in fixations[:4]]
    lengths = [min(r["length"], MAX_LENGTH) for r in fixations[:4]]
    gt = torch.from_numpy(store.get_batch(keys))
    pred = _pred(4)

    forward = score_step_heatmaps(pred, gt, lengths, max_length=MAX_LENGTH)
    # The reversed call: the ground truth in the prediction's slot. Shapes differ
    # (gt is (N, T, rows, cols), pred is (N, T, bins+1)), so the reversal is built
    # by re-shaping gt into the prediction layout -- which is precisely the mistake
    # that would go unnoticed if the wrapper accepted either order.
    gt_as_pred = torch.cat([torch.zeros((4, MAX_LENGTH, 1)),
                            gt.reshape(4, MAX_LENGTH, ROWS * COLS)], dim=-1)
    pred_as_gt = pred[:, :, 1:].reshape(4, MAX_LENGTH, ROWS, COLS)
    reverse = score_step_heatmaps(gt_as_pred, pred_as_gt, lengths,
                                  max_length=MAX_LENGTH)
    assert forward["KLD"] != reverse["KLD"]


def test_weighted_accumulation_equals_the_single_pass(artefacts, fixations):
    """FR8.4 -- and a plain mean of batch means is asserted to DIFFER."""
    from eve_bridge.heatmap_metrics import score_step_heatmaps
    from eve_bridge.store import GtHeatmapStore

    store = GtHeatmapStore.load(artefacts["heatmaps"],
                                fixations_path=artefacts["fixations"])
    recs = fixations[:6]
    keys = ["{}|{}".format(r["name"], r["subject"]) for r in recs]
    lengths = [min(r["length"], MAX_LENGTH) for r in recs]
    pred = _pred(6)

    whole = score_step_heatmaps(pred, torch.from_numpy(store.get_batch(keys)),
                                lengths, max_length=MAX_LENGTH)

    cut = 2                                   # deliberately unequal M per batch
    sums = {"NSS": 0.0, "CC": 0.0, "KLD": 0.0}
    total_m = 0
    batch_means = []
    for lo, hi in ((0, cut), (cut, 6)):
        s = score_step_heatmaps(pred[lo:hi],
                                torch.from_numpy(store.get_batch(keys[lo:hi])),
                                lengths[lo:hi], max_length=MAX_LENGTH)
        m = sum(lengths[lo:hi])
        for k in sums:
            sums[k] += s[k] * m
        total_m += m
        batch_means.append(s)

    assert total_m == sum(lengths)
    for k in sums:
        assert sums[k] / total_m == pytest.approx(whole[k], abs=1e-6), k

    naive = {k: sum(b[k] for b in batch_means) / len(batch_means) for k in sums}
    assert any(abs(naive[k] - whole[k]) > 1e-9 for k in sums), (
        "the fixture no longer distinguishes a weighted mean from a mean of means")


def test_store_load_rejects_a_changed_fixations_file(artefacts, tmp_path):
    """FR8.1 -- the sha gate is the safety net for POSITIONAL row addressing."""
    from eve_bridge.store import GtHeatmapStore

    tampered = tmp_path / "fixations.json"
    with open(artefacts["fixations"]) as fh:
        records = json.load(fh)
    records[0]["X"][0] += 1.0
    with open(tampered, "w") as fh:
        json.dump(records, fh)

    with pytest.raises(ValueError) as exc:
        GtHeatmapStore.load(artefacts["heatmaps"], fixations_path=str(tampered))
    assert "hash mismatch" in str(exc.value)

    # ...and a mere REORDER, with no value changed, is rejected too: the store is
    # addressed positionally, so a reordered file mis-indexes every row (convention 10).
    reordered = tmp_path / "reordered.json"
    with open(artefacts["fixations"]) as fh:
        records = json.load(fh)
    with open(reordered, "w") as fh:
        json.dump(list(reversed(records)), fh)
    with pytest.raises(ValueError):
        GtHeatmapStore.load(artefacts["heatmaps"], fixations_path=str(reordered))


def test_empty_heatmap_dir_disables_the_block_and_says_so(arg_parser_factory):
    """FR8.7 -- absence is stated, never implied."""
    parser = arg_parser_factory()
    defaults = {a.dest: a.default for a in parser._actions}
    assert defaults["heatmap_dir"] == ""

    with open(TEST_PY, encoding="utf-8") as fh:
        source = fh.read()
    # The store is constructed only when the flag is set, and the record carries the
    # reason when it is not.
    assert "if args.heatmap_dir else None" in source
    assert "heatmap_block = None if (store is None or hm_M == 0) else {" in source
    assert '"heatmap_reason"' in source
    assert '"--heatmap_dir not set"' in source
    assert '"no valid timesteps were scored (M = 0)"' in source
