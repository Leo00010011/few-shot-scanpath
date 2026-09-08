"""EVE bundle -> canonical ``fixations.json`` records and stimulus export (FR1-FR6).

numpy + PIL only (plus the ``evedataset`` bundle object handed in by the caller).
"""

import difflib
import hashlib
import os
import random

import numpy as np
from PIL import Image

# Key order is part of the contract (FR5.1 / validation Group 3).
RECORD_KEYS = ["name", "subject", "X", "Y", "T", "length", "split", "condition", "task"]

CONDITION = "freeview"
TASK = "none"

COUNTER_NAMES = (
    "empty_scanpath",
    "non_finite",
    "no_stimulus",
    "duplicate_trial",
    "incomplete_stimulus",
    "clamped_coords",
    "zero_duration",
    "over_max_length",
    "short_scanpath",
)


def _new_counters():
    return {name: 0 for name in COUNTER_NAMES}


def _resolve_subjects(available, unseen_subjects):
    """FR1.1, FR1.2, FR1.4, FR1.5."""
    if unseen_subjects is None:
        unseen_subjects = [s for s in available if str(s).startswith("test")]
    unseen_subjects = sorted(set(str(s) for s in unseen_subjects))

    if len(unseen_subjects) < 2:
        raise ValueError(
            "at least 2 unseen subjects are required (got {}: {}); the evaluator's "
            "(subject x subject) matrix and the retrieval block are meaningless "
            "below 2".format(len(unseen_subjects), unseen_subjects))

    available_sorted = sorted(set(str(s) for s in available))
    for sid in unseen_subjects:
        if sid not in available_sorted:
            close = difflib.get_close_matches(sid, available_sorted, n=5, cutoff=0.0)
            raise ValueError(
                "subject {!r} is not present in samples_df['subject']; "
                "closest available ids: {}".format(sid, close))
    return unseen_subjects


def _convert_scanpath(sp, origin_size, counters):
    """FR4.2 - FR4.4. ``sp`` is (4, F): [start_t_sec, duration_ms, x_px, y_px]."""
    x = sp[2].astype(np.float64) + 1.0
    y = sp[3].astype(np.float64) + 1.0
    t = sp[1].astype(np.float64)

    H, W = origin_size
    counters["clamped_coords"] += int((x < 1.0).sum() + (x > float(W)).sum()
                                      + (y < 1.0).sum() + (y > float(H)).sum())
    x = np.clip(x, 1.0, float(W))
    y = np.clip(y, 1.0, float(H))

    T = [int(round(float(v))) for v in t]
    counters["zero_duration"] += sum(1 for v in T if v == 0)
    T = [max(v, 1) for v in T]

    return [float(v) for v in x], [float(v) for v in y], T


def build_fixations(bundle, unseen_subjects=None, support_pool_size=20,
                    seed=0, origin_size=(1080, 1920), max_length=16):
    """Build the canonical fixation records for the selected EVE participants.

    Returns ``(fixations, exp_keys_aligned, subject_id_map, counters)``.
    """
    df = bundle.samples_df
    counters = _new_counters()

    subjects = _resolve_subjects(df["subject"].tolist(), unseen_subjects)
    to_dense = {sid: i for i, sid in enumerate(subjects)}
    subject_id_map = {
        "to_dense": to_dense,
        "to_eve": {str(i): sid for sid, i in to_dense.items()},
    }
    n_subjects = len(subjects)

    cand = df[df["subject"].isin(subjects) & df["valid"].astype(bool)]

    per_stimulus = {}
    for row in cand.itertuples(index=False):
        exp_key = str(row.exp_key)
        stimulus_name = str(row.stimulus_name)
        dense = to_dense[str(row.subject)]

        if not str(row.stimulus_path):
            counters["no_stimulus"] += 1
            continue

        sp = bundle.get_scanpath(exp_key)
        if sp.shape[1] == 0:
            counters["empty_scanpath"] += 1
            continue
        if not np.isfinite(sp[1:4]).all():
            counters["non_finite"] += 1
            continue

        slot = per_stimulus.setdefault(stimulus_name, {})
        if dense in slot:
            counters["duplicate_trial"] += 1
            continue

        X, Y, T = _convert_scanpath(sp, origin_size, counters)
        slot[dense] = {"exp_key": exp_key, "X": X, "Y": Y, "T": T}

    # FR2.3 — equal-subject invariant.
    complete = [s for s, d in per_stimulus.items() if len(d) == n_subjects]
    counters["incomplete_stimulus"] = len(per_stimulus) - len(complete)

    # FR2.5 — the feature path is built by a literal str.replace('jpg', 'pth').
    for name in sorted(complete):
        if "jpg" in name:
            raise ValueError(
                "stimulus_name {!r} contains the substring 'jpg'; the feature path is "
                "built by a literal str.replace('jpg', 'pth') (TechStack 3.2)".format(name))

    # FR2.4
    if len(complete) < support_pool_size + 1:
        raise ValueError(
            "only {} stimuli survive filtering, which is fewer than "
            "support_pool_size + 1 = {}".format(len(complete), support_pool_size + 1))

    # FR3.1 — deterministic partition.
    names = sorted(complete)
    random.Random(seed).shuffle(names)
    train_names = set(names[:support_pool_size])
    test_names = set(names[support_pool_size:])
    assert not (train_names & test_names), (
        "support and query pools overlap: {}".format(sorted(train_names & test_names)))

    fixations = []
    exp_keys_aligned = []
    for stimulus_name in sorted(complete):
        split = "train" if stimulus_name in train_names else "test"
        for dense in range(n_subjects):
            draft = per_stimulus[stimulus_name][dense]
            length = len(draft["X"])
            if length > max_length:
                counters["over_max_length"] += 1
            if length < 3:
                counters["short_scanpath"] += 1
            fixations.append({
                "name": "{}.jpg".format(stimulus_name),
                "subject": int(dense),
                "X": draft["X"],
                "Y": draft["Y"],
                "T": draft["T"],
                "length": int(length),
                "split": split,
                "condition": CONDITION,
                "task": TASK,
            })
            exp_keys_aligned.append(draft["exp_key"])

    # FR5.3 — record order is sorted by (name, subject); the loop above already
    # emits that order, but sort explicitly so the contract does not depend on it.
    order = sorted(range(len(fixations)),
                   key=lambda i: (fixations[i]["name"], fixations[i]["subject"]))
    fixations = [fixations[i] for i in order]
    exp_keys_aligned = [exp_keys_aligned[i] for i in order]

    return fixations, exp_keys_aligned, subject_id_map, counters


def export_stimuli(bundle, fixations, exp_keys, out_dir):
    """FR6 — one native-resolution .jpg per unique stimulus name."""
    stim_dir = os.path.join(out_dir, "stimuli")
    os.makedirs(stim_dir, exist_ok=True)

    name_to_exp_keys = {}
    for rec, exp_key in zip(fixations, exp_keys):
        name_to_exp_keys.setdefault(rec["name"], []).append(exp_key)

    conflicts = 0
    for name in sorted(name_to_exp_keys):
        keys = sorted(name_to_exp_keys[name])
        first_digest = None
        for pos, exp_key in enumerate(keys):
            arr = bundle.get_stimulus(exp_key)
            digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            if pos == 0:
                first_digest = digest
                Image.fromarray(arr).save(
                    os.path.join(stim_dir, name), quality=95, subsampling=0)
            elif digest != first_digest:
                conflicts += 1

    return {"stimulus_image_conflict": conflicts}
