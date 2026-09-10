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
    "support_stimulus_private",
    "support_stimulus_shared",
    "unused_stimulus",
    "surplus_trial",
    "clamped_coords",
    "zero_duration",
    "over_max_length",
    "short_scanpath",
)


def _new_counters():
    return {name: 0 for name in COUNTER_NAMES}


def _resolve_subjects(available, unseen_subjects, subjects_per_image=3):
    """FR1.1, FR1.2, FR1.4, FR1.5.

    ``available`` is the participant list restricted to *valid* trials, so the
    default cohort cannot silently include a participant with nothing to score.
    (On the EVE bundle every ``test*`` participant is ``valid == False``
    throughout -- Roadmap F-A -- so the old "every id starting 'test'" default
    resolved to an empty cohort.)
    """
    if unseen_subjects is None:
        unseen_subjects = list(available)
    unseen_subjects = sorted(set(str(s) for s in unseen_subjects))

    if len(unseen_subjects) < subjects_per_image:
        raise ValueError(
            "at least subjects_per_image = {} subjects are required (got {}: {}); "
            "each scored image must contribute exactly that many records or the "
            "evaluator's bare np.mean() folds -1 sentinels into every metric"
            .format(subjects_per_image, len(unseen_subjects), unseen_subjects))

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
                    seed=0, origin_size=(1080, 1920), max_length=16,
                    subjects_per_image=3):
    """Build the canonical fixation records for the selected EVE participants.

    Returns ``(fixations, exp_keys_aligned, subject_id_map, counters)``.
    """
    df = bundle.samples_df
    counters = _new_counters()

    valid_subjects = sorted(set(
        str(x) for x in df[df["valid"].astype(bool)]["subject"].tolist()))
    subjects = _resolve_subjects(valid_subjects, unseen_subjects, subjects_per_image)
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

    # --- FR2.3 (revised 2026-09-10) — every SCORED image contributes exactly
    # ``subjects_per_image`` records; the subject IDENTITIES may differ per image.
    #
    # What the frozen evaluator requires is a uniform COUNT, not a common cohort.
    # ``comprehensive_evaluation_by_subject`` loops
    # ``for row_idx in range(len(predict_fix_vector))`` — the actual per-image
    # list length — and its diagonal is positional, so position i is the same
    # participant in the prediction and the ground truth whichever participant
    # that is. The collectors are merely ALLOCATED ``(n_images, subject_num, …)``
    # and reduced with a bare ``np.mean()`` carrying no ``!= -1`` filter on this
    # branch, so an image contributing fewer than ``subject_num`` records folds
    # -1 sentinels into every metric. That arithmetic — not any metric — is what
    # forces the uniform count.
    #
    # ``args.subject_num`` does NOT size the model's embedding table:
    # ``gazeformer.py`` line 100 (``nn.Embedding(subject_num, …)``) is commented
    # out and ``self.subject_embed`` is whatever tensor ``--user_emb_path``
    # holds, indexed by the record's dense subject id. So a cohort of many
    # participants can be scored with ``subject_num = 3``.
    K = subjects_per_image
    eligible = sorted(s for s, d in per_stimulus.items() if len(d) >= K)
    private = {}
    for dense in range(n_subjects):
        cands = sorted(s for s in per_stimulus
                       if len(per_stimulus[s]) < K and dense in per_stimulus[s])
        random.Random("{}-{}".format(seed, dense)).shuffle(cands)
        private[dense] = cands
    counters["incomplete_stimulus"] = len(per_stimulus) - len(eligible)

    # FR2.5 — the feature path is built by a literal str.replace('jpg', 'pth').
    for name in sorted(per_stimulus):
        if "jpg" in name:
            raise ValueError(
                "stimulus_name {!r} contains the substring 'jpg'; the feature path is "
                "built by a literal str.replace('jpg', 'pth') (TechStack 3.2)".format(name))

    if not eligible:
        raise ValueError(
            "no stimulus was seen by {} of the {} selected subjects, so the query "
            "split would be empty".format(K, n_subjects))

    # FR3.1 — support prefers stimuli that can never be scored anyway (seen by
    # fewer than K subjects). A query-eligible stimulus is diverted ONLY when some
    # subject would otherwise have no support at all, because diverting costs a
    # scored image for every subject (FR3.3 disjointness is by NAME).
    #
    # The pools are deliberately NOT trimmed to a common size: each subject's
    # embedding is built from its own num_fewshot scanpaths, so only the MINIMUM
    # matters, and levelling everyone down to the thinnest subject would discard
    # support the other subjects already have (median 24 vs min 9 on EVE) while
    # buying nothing.
    thinnest = min(len(private[d]) for d in range(n_subjects))
    need = support_pool_size if thinnest == 0 else 0

    # FR2.4
    if len(eligible) < need + 1:
        raise ValueError(
            "only {} stimuli were seen by {} subjects, which is fewer than "
            "support_pool_size + 1 = {}".format(len(eligible), K, support_pool_size + 1))

    eligible_pool = list(eligible)
    random.Random(seed).shuffle(eligible_pool)
    diverted = set(eligible_pool[:need])

    support_by_subject = {}
    for dense in range(n_subjects):
        picked = set(private[dense][:support_pool_size])
        if len(picked) < support_pool_size:
            for name in sorted(diverted):
                if len(picked) >= support_pool_size:
                    break
                if dense in per_stimulus[name]:
                    picked.add(name)
        support_by_subject[dense] = picked

    private_used = set()
    for names in support_by_subject.values():
        private_used |= {s for s in names if s not in diverted}
    counters["support_stimulus_private"] = len(private_used)
    counters["support_stimulus_shared"] = len(diverted)
    counters["unused_stimulus"] = (
        len(per_stimulus) - len(eligible) - len(private_used))

    # Query split: exactly K subjects per image, chosen deterministically. A
    # surplus trial on an image seen by more than K subjects is DROPPED rather
    # than routed to support -- FR3.3 disjointness is by stimulus NAME, and a
    # name in both splits would break it.
    query = {}
    for name in sorted(set(eligible) - diverted):
        chosen = sorted(per_stimulus[name])
        random.Random("{}-query-{}".format(seed, name)).shuffle(chosen)
        query[name] = sorted(chosen[:K])
        counters["surplus_trial"] += len(per_stimulus[name]) - K

    train_names = set(diverted) | private_used
    test_names = set(query)
    assert not (train_names & test_names), (
        "support and query pools overlap: {}".format(sorted(train_names & test_names)))

    fixations = []
    exp_keys_aligned = []

    def emit(stimulus_name, dense, split):
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

    for stimulus_name in sorted(query):
        for dense in query[stimulus_name]:
            emit(stimulus_name, dense, "test")

    for stimulus_name in sorted(train_names):
        for dense in sorted(per_stimulus[stimulus_name]):
            if stimulus_name in support_by_subject[dense]:
                emit(stimulus_name, dense, "train")

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
