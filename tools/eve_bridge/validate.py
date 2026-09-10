"""Artefact validator for the EVE bridge (FR9).

Runs the FR9.1 invariants in order against a written ``<out_dir>`` and raises
``BridgeValidationError`` on the first failure, naming the offending record.
numpy + h5py + json only.
"""

import json
import os
import random

import numpy as np

from .convert import CONDITION, RECORD_KEYS, TASK
from .store import GtHeatmapStore, trial_key


class BridgeValidationError(RuntimeError):
    pass


def _fail(msg):
    raise BridgeValidationError(msg)


def _where(i, rec):
    return "record {} ({}, subject {})".format(i, rec.get("name"), rec.get("subject"))


def validate(out_dir, bundle=None, sample_n=32, seed=0, subjects_per_image=None):
    """Validate the artefacts under ``out_dir``. Returns ``{check: "ok", ...}``."""
    results = {}

    fixations_path = os.path.join(out_dir, "fixations.json")
    if not os.path.isfile(fixations_path):
        _fail("fixations.json missing from {}".format(out_dir))
    with open(fixations_path) as fh:
        fixations = json.load(fh)
    if not fixations:
        _fail("fixations.json is empty")

    subject_map_path = os.path.join(out_dir, "subject_id_map.json")
    if not os.path.isfile(subject_map_path):
        _fail("subject_id_map.json missing from {}".format(out_dir))
    with open(subject_map_path) as fh:
        subject_id_map = json.load(fh)
    n_subjects = len(subject_id_map["to_dense"])

    origin_h, origin_w = 1080, 1920

    # --- 1. lengths agree ------------------------------------------------
    for i, rec in enumerate(fixations):
        if not (len(rec["X"]) == len(rec["Y"]) == len(rec["T"]) == rec["length"]):
            _fail("{}: len(X)={} len(Y)={} len(T)={} length={} disagree (FR9.1.1)".format(
                _where(i, rec), len(rec["X"]), len(rec["Y"]), len(rec["T"]),
                rec["length"]))
    results["lengths_agree"] = "ok"

    # --- 2. coordinate bounds -------------------------------------------
    for i, rec in enumerate(fixations):
        for axis, vals, hi in (("X", rec["X"], origin_w), ("Y", rec["Y"], origin_h)):
            for j, v in enumerate(vals):
                if not (1.0 <= v <= float(hi)):
                    _fail("{}: {}[{}] = {} outside [1.0, {}] (FR9.1.2)".format(
                        _where(i, rec), axis, j, v, float(hi)))
    results["coordinate_bounds"] = "ok"

    # --- 3. durations ----------------------------------------------------
    for i, rec in enumerate(fixations):
        for j, v in enumerate(rec["T"]):
            if not isinstance(v, int) or isinstance(v, bool):
                _fail("{}: T[{}] = {!r} is not an int (FR9.1.3)".format(
                    _where(i, rec), j, v))
            if v < 1:
                _fail("{}: T[{}] = {} is below 1 ms (FR9.1.3)".format(
                    _where(i, rec), j, v))
    results["durations"] = "ok"

    # --- 4. dense subject range -----------------------------------------
    subjects = set()
    for i, rec in enumerate(fixations):
        s = rec["subject"]
        if not isinstance(s, int) or isinstance(s, bool):
            _fail("{}: subject {!r} is not an int (FR9.1.4)".format(_where(i, rec), s))
        if not (0 <= s < n_subjects):
            _fail("{}: subject {} outside 0..{} (FR9.1.4)".format(
                _where(i, rec), s, n_subjects - 1))
        subjects.add(s)
    if subjects != set(range(n_subjects)):
        _fail("subject ids {} do not cover 0..{} (FR9.1.4)".format(
            sorted(subjects), n_subjects - 1))
    results["dense_subject_ids"] = "ok"

    # --- 5. splits -------------------------------------------------------
    train_names, test_names = set(), set()
    for i, rec in enumerate(fixations):
        if rec["split"] == "train":
            train_names.add(rec["name"])
        elif rec["split"] == "test":
            test_names.add(rec["name"])
        else:
            _fail("{}: split {!r} is not one of train/test (FR9.1.6)".format(
                _where(i, rec), rec["split"]))
    overlap = train_names & test_names
    if overlap:
        _fail("train/test stimulus overlap: {} (FR9.1.6, FR3.3)".format(sorted(overlap)))
    results["split_disjoint"] = "ok"

    # --- 6. uniform subject count on the SCORED split ---------------------
    # The frozen evaluator allocates its collectors ``(n_images, subject_num, …)``
    # and reduces them with a bare ``np.mean()`` carrying no ``!= -1`` filter, so
    # an image contributing fewer than ``subject_num`` records folds -1 sentinels
    # into every metric. The requirement is therefore a uniform COUNT per scored
    # image -- the subject IDENTITIES may differ from image to image, since the
    # evaluator's loops and diagonal are positional. The support split carries no
    # such contract and is deliberately ragged.
    by_name = {}
    for i, rec in enumerate(fixations):
        by_name.setdefault((rec["split"], rec["name"]), []).append((i, rec["subject"]))
    for (split, name), entries in sorted(by_name.items()):
        if len(set(s for _, s in entries)) != len(entries):
            _fail("stimulus {} ({} split) has duplicate subjects: {} "
                  "(FR9.1.5)".format(name, split, sorted(s for _, s in entries)))
    test_counts = {name: len(v) for (split, name), v in by_name.items() if split == "test"}
    if not test_counts:
        _fail("no records on the test split; there is nothing to score (FR9.1.5)")
    distinct = sorted(set(test_counts.values()))
    if len(distinct) != 1:
        offenders = {n: c for n, c in sorted(test_counts.items()) if c != distinct[-1]}
        _fail("scored images do not all carry the same number of subjects: counts {} "
              "(e.g. {}); the evaluator's bare np.mean() would fold -1 sentinels into "
              "every metric (FR9.1.5)".format(distinct, dict(list(offenders.items())[:5])))
    if subjects_per_image is not None and distinct[0] != subjects_per_image:
        _fail("scored images carry {} subjects each, expected subjects_per_image = {} "
              "(FR9.1.5)".format(distinct[0], subjects_per_image))
    if distinct[0] > n_subjects:
        _fail("scored images carry {} subjects each but the cohort has only {} "
              "(FR9.1.5)".format(distinct[0], n_subjects))
    results["uniform_subject_count"] = "ok ({} per scored image)".format(distinct[0])

    # --- 7. constants ----------------------------------------------------
    for i, rec in enumerate(fixations):
        if rec["condition"] != CONDITION or rec["task"] != TASK:
            _fail("{}: condition/task = {!r}/{!r}, expected {!r}/{!r} "
                  "(FR9.1.7)".format(_where(i, rec), rec["condition"], rec["task"],
                                     CONDITION, TASK))
        if list(rec.keys()) != RECORD_KEYS:
            _fail("{}: key order {} != {} (FR5.1)".format(
                _where(i, rec), list(rec.keys()), RECORD_KEYS))
    results["constants"] = "ok"

    # --- 8. stimulus files exist ----------------------------------------
    stim_dir = os.path.join(out_dir, "stimuli")
    for i, rec in enumerate(fixations):
        path = os.path.join(stim_dir, rec["name"])
        if not os.path.isfile(path):
            _fail("{}: stimulus file {} does not exist (FR9.1.8)".format(
                _where(i, rec), path))
    results["stimulus_files"] = "ok"

    # --- 9. no 'jpg' before the extension --------------------------------
    for i, rec in enumerate(fixations):
        stem = rec["name"][:-len(".jpg")] if rec["name"].endswith(".jpg") else rec["name"]
        if "jpg" in stem:
            _fail("{}: name contains 'jpg' before its extension (FR9.1.9)".format(
                _where(i, rec)))
    results["no_jpg_in_stem"] = "ok"

    # --- 10 / 11. heatmap store ------------------------------------------
    store_path = os.path.join(out_dir, "gt_heatmaps.h5")
    if not os.path.isfile(store_path):
        results["heatmap_store"] = "skipped (no gt_heatmaps.h5)"
        results["heatmap_sampling"] = "skipped (no gt_heatmaps.h5)"
        results["sampled_trial_keys"] = []
        return results

    try:
        store = GtHeatmapStore.load(store_path, fixations_path=fixations_path)
    except ValueError as exc:
        _fail("heatmap store rejects fixations.json: {} (FR9.1.10)".format(exc))

    if len(store.trial_keys) != len(fixations):
        _fail("heatmap store has {} rows but fixations.json has {} records "
              "(FR9.1.10)".format(len(store.trial_keys), len(fixations)))
    for i, rec in enumerate(fixations):
        expected = trial_key(rec["name"], rec["subject"])
        if store.trial_keys[i] != expected:
            _fail("{}: heatmap store row {} has trial_key {!r}, expected {!r} "
                  "(FR9.1.10)".format(_where(i, rec), i, store.trial_keys[i], expected))
    results["heatmap_store"] = "ok"

    # --- 11. sampled numeric checks --------------------------------------
    action_map = tuple(store.attrs["action_map"])
    max_length = store.attrs["max_length"]
    origin_size = tuple(store.attrs["origin_size"])
    downscale_x = origin_size[1] / action_map[1]
    downscale_y = origin_size[0] / action_map[0]

    rng = random.Random(seed)
    idx = list(range(len(fixations)))
    rng.shuffle(idx)
    idx = sorted(idx[:min(sample_n, len(idx))])

    sampled_keys = []
    for i in idx:
        rec = fixations[i]
        hm = store.heatmaps[i]
        n_valid = min(rec["length"], max_length)
        sampled_keys.append(store.trial_keys[i])

        for t in range(n_valid):
            total = float(hm[t].sum())
            if abs(total - 1.0) > 1e-5:
                _fail("{}: heatmap step {} sums to {} not 1.0 (FR9.1.11)".format(
                    _where(i, rec), t, total))
            # same arithmetic as build_step_heatmaps: float32, truncating cast
            exp_col = int(((np.float32(rec["X"][t]) - 1) / downscale_x).astype(np.int32))
            exp_row = int(((np.float32(rec["Y"][t]) - 1) / downscale_y).astype(np.int32))
            got = np.unravel_index(int(hm[t].argmax()), hm[t].shape)
            if (int(got[0]), int(got[1])) != (exp_row, exp_col):
                _fail("{}: heatmap step {} peaks at {} but (X, Y) discretize to "
                      "({}, {}) (FR9.1.11)".format(
                          _where(i, rec), t, (int(got[0]), int(got[1])),
                          exp_row, exp_col))
        for t in range(n_valid, max_length):
            if float(np.abs(hm[t]).sum()) != 0.0:
                _fail("{}: heatmap step {} beyond length is not zero "
                      "(FR9.1.11)".format(_where(i, rec), t))

        if bundle is not None:
            sp = bundle.get_scanpath(store.exp_keys[i])
            for t in range(rec["length"]):
                x_src = float(np.clip(float(sp[2][t]) + 1.0, 1.0, float(origin_size[1])))
                y_src = float(np.clip(float(sp[3][t]) + 1.0, 1.0, float(origin_size[0])))
                if abs(rec["X"][t] - x_src) > 1e-4 or abs(rec["Y"][t] - y_src) > 1e-4:
                    _fail("{}: fixation {} = ({}, {}) does not match the bundle's "
                          "({}, {}) (FR9.1.11, cross-check)".format(
                              _where(i, rec), t, rec["X"][t], rec["Y"][t],
                              x_src, y_src))
    results["heatmap_sampling"] = "ok"
    results["sampled_trial_keys"] = sampled_keys
    return results
