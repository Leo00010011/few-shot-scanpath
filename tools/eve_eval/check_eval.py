"""The F5 preflight: every artefact handshake, before the checkpoint is built (FR3).

This is the gate ``bash/test_eve.sh`` runs on, and the same function ``src/test.py``
calls in-process so that a bare ``python src/test.py`` is as safe as the script path.

Two design rules, both learned rather than chosen:

* **Collect every failure, print every failure.** A first-failure-only preflight turns
  one broken shipment into N allocations. Each check appends to ``failures`` and the
  caller exits once, after printing all of them (FR11.2).
* **Write ``preflight.json`` either way.** A failing run still writes the report, with
  ``"ok": false`` and the failure list. An absent report is worse than a failing one:
  it is indistinguishable from a run that never started (D5).

Stdlib + numpy + h5py. **No torch** -- FR3.4 needs file *bytes*, not tensor loads, and
this module has to import on a login node with nothing staged (FR11.1). The only
non-stdlib reads are ``h5py`` for three root attrs and the two vlen-string datasets
``load_trial_exp_keys`` already knows how to read.

CLI::

    python tools/eve_eval/check_eval.py --bridge-dir DIR --senet-dir DIR \\
        --feature-dir DIR --weights-dir DIR [--fast] [--out preflight.json]

exit 0 = every check passed; 1 = at least one failed (all of them are printed).
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    # eve_prep is a sibling package, not a dependency that pip ever installed; the run
    # script puts tools/ on PYTHONPATH, and this makes a bare `python check_eval.py`
    # work too.
    sys.path.insert(0, _TOOLS)

try:
    from . import (ACTION_MAP, EveEvalError, MAX_LENGTH, MIN_PLAUSIBLE_MAX_T,
                   N_SUBJECTS, ORIGIN_SIZE, SCORED_CELLS, SCORED_IMAGES)
    from .parity import check_heatmap_parity
except ImportError:  # executed as a script, not as a package member
    from eve_eval import (ACTION_MAP, EveEvalError, MAX_LENGTH,
                          MIN_PLAUSIBLE_MAX_T, N_SUBJECTS, ORIGIN_SIZE,
                          SCORED_CELLS, SCORED_IMAGES)
    from eve_eval.parity import check_heatmap_parity

from eve_prep.trial_keys import exp_key_filename, load_trial_exp_keys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: FR3.7 -- the task embedding F1 scored with. Using the same bytes is what keeps the
#: OSIE baseline and the EVE run comparable on this input (TechStack section 3.9).
OSIE_EMBEDDINGS = os.path.join(REPO_ROOT, "ISP", "OSIE", "GazeformerISP", "src",
                               "data", "embeddings.npy")

TASK_EMB_KEY = "free-viewing"
TASK_EMB_SHAPE = (768,)

_MAX_LISTED = 10

GIT_IGNORE_NOTE = ("all of data/ is git-ignored, so nothing under it arrives by "
                   "`git pull` -- F2/F3/F4's artefacts are shipped to the cluster "
                   "by hand (Roadmap section 0)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _listing(items):
    items = list(items)
    shown = [str(_) for _ in items[:_MAX_LISTED]]
    text = ", ".join(shown)
    if len(items) > len(shown):
        text += " (+{} more)".format(len(items) - len(shown))
    return "{} total: {}".format(len(items), text)


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _find_user_embedding(senet_dir):
    """The ``(N, 384)`` tensor F3 wrote, located by its own naming convention."""
    hits = sorted(glob.glob(os.path.join(
        senet_dir, "eve_fewshot_user_embedding_*_seed*.pt")))
    return hits[0] if hits else os.path.join(
        senet_dir, "eve_fewshot_user_embedding_10_seed0.pt")


def _seed_in_name(path):
    """``..._seed3.pt`` -> 3, or ``None`` when the name carries no seed."""
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit("_seed", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        return None
    return int(tail[1])


def _report_seed(senet_report):
    """F3 writes the seed under ``args``; accept a top-level one too (FR3.3)."""
    if "seed" in senet_report:
        return senet_report["seed"]
    return senet_report.get("args", {}).get("seed")


def check_eval(bridge_dir, senet_dir, feature_dir, weights_dir, *,
               subject_num=3, num_fewshot=10, fast=False,
               fixations_path=None, heatmaps_path=None, subject_map_path=None,
               bridge_report_path=None, senet_report_path=None,
               feature_report_path=None, user_emb_path=None, emb_npy_path=None,
               feat_dir=None, checkpoint_path=None,
               osie_embeddings=OSIE_EMBEDDINGS, split="test",
               expected_cells=SCORED_CELLS, expected_images=SCORED_IMAGES,
               expected_subjects=N_SUBJECTS):
    """Run FR3.1-FR3.10. Returns the ``preflight.json`` dict; never raises on a
    *contract* failure -- ``report["ok"]`` and ``report["failures"]`` carry those, so
    the caller can print all of them at once. Malformed JSON and unreadable HDF5 still
    raise, because those are not contract violations but broken artefacts.
    """
    failures = []

    def fail(msg):
        failures.append(msg)

    # Path resolution: directories are the interface, but every individual path is
    # overridable so test.py can hand over exactly the ones it was given (FR6.3).
    fixations_path = fixations_path or os.path.join(bridge_dir, "fixations.json")
    heatmaps_path = heatmaps_path or os.path.join(bridge_dir, "gt_heatmaps.h5")
    subject_map_path = subject_map_path or os.path.join(bridge_dir,
                                                        "subject_id_map.json")
    bridge_report_path = bridge_report_path or os.path.join(bridge_dir,
                                                            "bridge_report.json")
    senet_report_path = senet_report_path or os.path.join(senet_dir,
                                                          "senet_report.json")
    feature_report_path = feature_report_path or os.path.join(feature_dir,
                                                              "feature_report.json")
    user_emb_path = user_emb_path or _find_user_embedding(senet_dir)
    emb_npy_path = emb_npy_path or os.path.join(feature_dir, "embeddings.npy")
    feat_dir = feat_dir or os.path.join(feature_dir, "image_features")
    checkpoint_path = checkpoint_path or os.path.join(weights_dir, "checkpoints",
                                                      "checkpoint_best.pth")

    paths = {
        "fixations": fixations_path, "heatmaps": heatmaps_path,
        "subject_map": subject_map_path, "bridge_report": bridge_report_path,
        "senet_report": senet_report_path, "feature_report": feature_report_path,
        "user_embedding": user_emb_path, "embeddings_npy": emb_npy_path,
        "feature_dir": feat_dir, "checkpoint": checkpoint_path,
        "osie_embeddings": osie_embeddings,
    }

    report = {
        "ok": False,
        "split": split,
        "subject_num": int(subject_num),
        "num_fewshot": int(num_fewshot),
        "paths": {k: os.path.abspath(v) for k, v in paths.items()},
        "feature_sha256_checked": not fast,
        "failures": failures,
    }

    # ---- FR3.1 existence ---------------------------------------------------
    missing_artefacts = []
    for key, path in paths.items():
        if key == "feature_dir":
            if not os.path.isdir(path):
                missing_artefacts.append("{} (directory): {}".format(key, path))
            continue
        if not os.path.isfile(path):
            missing_artefacts.append("{}: {}".format(key, path))
    if missing_artefacts:
        for item in missing_artefacts:
            fail("FR3.1 missing artefact -- {}. {}".format(item, GIT_IGNORE_NOTE))
        # Without fixations.json and the reports there is nothing further to check;
        # everything below would fail with an IOError rather than a contract message.
        report["failures"] = failures
        return report

    fixations = _load_json(fixations_path)
    bridge_report = _load_json(bridge_report_path)
    senet_report = _load_json(senet_report_path)
    feature_report = _load_json(feature_report_path)
    subject_map = _load_json(subject_map_path)

    # ---- FR3.2 the three-way fixations.json handshake ----------------------
    fix_sha = sha256_file(fixations_path)
    report["sha256"] = {"fixations": fix_sha}
    for label, rep in (("bridge_report.json", bridge_report),
                       ("senet_report.json", senet_report),
                       ("feature_report.json", feature_report)):
        claimed = rep.get("fixations_sha256")
        if claimed != fix_sha:
            # Every one of the three is reported, not just the first: which of them
            # disagrees says which upstream feature is describing a different build.
            fail("FR3.2 fixations.json hash mismatch against {}: the report was "
                 "built from {!r} but {} hashes to {!r}. A merely REORDERED "
                 "fixations.json changes this hash and is correctly rejected -- "
                 "bridge artefacts are addressed positionally.".format(
                     label, claimed, fixations_path, fix_sha))

    # ---- FR3.3 the subject-embedding handshake -----------------------------
    emb_sha = sha256_file(user_emb_path)
    report["sha256"]["user_embedding"] = emb_sha
    claimed_emb = senet_report.get("embedding_sha256")
    if claimed_emb != emb_sha:
        fail("FR3.3 subject-embedding hash mismatch: senet_report.json says {!r} "
             "but {} hashes to {!r}".format(claimed_emb, user_emb_path, emb_sha))
    name_seed = _seed_in_name(user_emb_path)
    report_seed = _report_seed(senet_report)
    if name_seed is not None and report_seed is not None and int(report_seed) != name_seed:
        fail("FR3.3 seed disagreement: {} is named _seed{} but senet_report.json "
             "records seed {}".format(os.path.basename(user_emb_path), name_seed,
                                      report_seed))

    # ---- FR3.5 the support-depth bound -------------------------------------
    min_support = bridge_report.get("min_support_per_subject")
    report["min_support_per_subject"] = min_support
    if min_support is None:
        fail("FR3.5 bridge_report.json carries no min_support_per_subject")
    elif int(num_fewshot) > int(min_support):
        fail("FR3.5 num_fewshot {} exceeds min_support_per_subject {}. The bound is "
             "the MINIMUM over subjects, never support_pool_size ({}): the support "
             "pools are per-subject and share no image names, so the thinnest pool "
             "binds (TechStack section 3.6).".format(
                 num_fewshot, min_support, bridge_report.get("support_pool_size")))

    # ---- FR3.6 origin_size and the store's attrs ---------------------------
    bridge_origin = bridge_report.get("origin_size")
    report["origin_size"] = bridge_origin
    if list(bridge_origin or []) != list(ORIGIN_SIZE):
        fail("FR3.6 bridge_report.json origin_size is {!r}, expected {!r} (H, W). "
             "The OSIE default (600, 800) would mis-scale every coordinate by "
             "2.4x / 1.8x and every metric would still return a number "
             "(D3).".format(bridge_origin, list(ORIGIN_SIZE)))

    with h5py.File(heatmaps_path, "r") as f:
        h5_attrs = {
            "origin_size": [int(v) for v in f.attrs["origin_size"]],
            "action_map": [int(v) for v in f.attrs["action_map"]],
            "max_length": int(f.attrs["max_length"]),
        }
    report["heatmap_attrs"] = h5_attrs
    for key, expected in (("origin_size", list(ORIGIN_SIZE)),
                          ("action_map", list(ACTION_MAP)),
                          ("max_length", MAX_LENGTH)):
        actual = h5_attrs[key]
        if (list(actual) if isinstance(actual, list) else actual) != expected:
            fail("FR3.6 gt_heatmaps.h5 attr {} is {!r}, expected {!r}".format(
                key, actual, expected))

    # ---- FR3.7 the task embedding ------------------------------------------
    emb_npy_sha = sha256_file(emb_npy_path)
    report["sha256"]["embeddings_npy"] = emb_npy_sha
    try:
        emb_dict = np.load(emb_npy_path, allow_pickle=True).item()
    except Exception as exc:                       # a corrupt .npy is a contract fail
        emb_dict = None
        fail("FR3.7 {} does not load as a dict: {!r}".format(emb_npy_path, exc))
    if emb_dict is not None:
        if not isinstance(emb_dict, dict) or TASK_EMB_KEY not in emb_dict:
            fail("FR3.7 {} has no {!r} key; keys are {!r}".format(
                emb_npy_path, TASK_EMB_KEY,
                sorted(emb_dict.keys()) if isinstance(emb_dict, dict) else type(emb_dict)))
        else:
            value = np.asarray(emb_dict[TASK_EMB_KEY])
            if value.shape != TASK_EMB_SHAPE or value.dtype != np.float32:
                fail("FR3.7 {}[{!r}] is {} {}, expected {} float32".format(
                    emb_npy_path, TASK_EMB_KEY, value.shape, value.dtype,
                    TASK_EMB_SHAPE))
    osie_sha = sha256_file(osie_embeddings)
    report["sha256"]["osie_embeddings_npy"] = osie_sha
    if emb_npy_sha != osie_sha:
        fail("FR3.7 {} is not byte-identical to the OSIE task embedding at {}. "
             "F1 scored with those bytes; a different file makes the two runs "
             "incomparable on this input (TechStack section 3.9).".format(
                 emb_npy_path, osie_embeddings))

    # ---- FR3.8 the cohort invariants ---------------------------------------
    scored = [r for r in fixations if r["split"] == split]
    by_name = collections.OrderedDict()
    for r in scored:
        by_name.setdefault(r["name"], []).append(r)

    subjects = sorted({int(r["subject"]) for r in scored})
    counters = {
        "n_cells": len(scored),
        "n_images": len(by_name),
        "n_subjects": len(subjects),
        "n_records_total": len(fixations),
    }
    report["counts"] = counters

    ragged = {n: len(v) for n, v in by_name.items() if len(v) != int(subject_num)}
    if ragged:
        fail("FR3.8 ragged scored images -- every scored name must carry exactly "
             "subject_num={} records. evaluation.py's collectors are -1-initialised "
             "and reduced by a bare np.mean() with no `!= -1` filter, so a ragged "
             "image folds -1 into every metric; {}".format(
                 subject_num, _listing(["{} -> {}".format(k, v)
                                        for k, v in sorted(ragged.items())])))
    # expected_* default to the realised cohort's constants; they are parameters only
    # so the test suite can drive the same code over a 6-image synthetic fixture. The
    # production call sites never pass them.
    if len(scored) != expected_cells:
        fail("FR3.8 the {!r} split holds {} records, expected {}".format(
            split, len(scored), expected_cells))
    if len(by_name) != expected_images:
        fail("FR3.8 the {!r} split holds {} distinct names, expected {}".format(
            split, len(by_name), expected_images))
    if subjects != list(range(expected_subjects)):
        fail("FR3.8 dense subject ids are not 0..{}: got {} distinct ids, "
             "{}".format(expected_subjects - 1, len(subjects), _listing(subjects)))

    max_T = max((max(r["T"]) for r in scored if r["T"]), default=0)
    report["max_T"] = int(max_T)
    if max_T <= MIN_PLAUSIBLE_MAX_T:
        fail("FR3.8 max(T) = {} over the {!r} split: this looks like the DECILE-BIN "
             "file (TechStack section 3.7), not a duration file. It passes every "
             "other invariant and silently invalidates ScanMatch-with-duration and "
             "MultiMatch's duration dimension, hence both SM and MM.".format(
                 max_T, split))

    bad_const = ["{}|{} condition={!r} task={!r}".format(
        r["name"], r["subject"], r.get("condition"), r.get("task"))
        for r in scored
        if r.get("condition") != "freeview" or r.get("task") != "none"]
    if bad_const:
        fail("FR3.8 condition/task constants violated (expected "
             "condition='freeview', task='none'); {}".format(_listing(bad_const)))

    # ---- FR3.9 / FR3.4 the per-trial feature cache -------------------------
    try:
        exp_of = load_trial_exp_keys(heatmaps_path, fixations_path)
    except Exception as exc:
        # Collected rather than raised: FR3.2 has usually already recorded WHY (a
        # regenerated fixations.json), and a preflight that aborts here would print
        # one failure where the caller needs all of them (FR11.2).
        exp_of = None
        fail("FR3.9 could not read the (name, subject) -> exp_key mapping from {}: "
             "{}".format(heatmaps_path, exc))
    scored_keys = [(r["name"], int(r["subject"])) for r in scored]

    no_exp_key, missing_pth, bad_sha, unsafe = [], [], [], []
    claimed_sha = feature_report.get("feature_sha256", {})
    checked = 0
    for key in (scored_keys if exp_of is not None else []):
        if key not in exp_of:
            no_exp_key.append("{}|{}".format(*key))
            continue
        exp_key = exp_of[key]
        try:
            rel = exp_key_filename(exp_key)        # charset + no-jpg rule, re-asserted
        except Exception as exc:
            unsafe.append("{}: {}".format(exp_key, exc))
            continue
        path = os.path.join(feat_dir, rel)
        if not os.path.isfile(path):
            missing_pth.append(rel)
            continue
        if fast:
            continue
        actual = sha256_file(path)
        checked += 1
        if claimed_sha.get(exp_key) != actual:
            bad_sha.append("{} report={!r} actual={!r}".format(
                exp_key, claimed_sha.get(exp_key), actual))

    report["counts"]["n_features_expected"] = len(scored_keys)
    report["counts"]["n_features_hashed"] = checked
    if no_exp_key:
        fail("FR3.9 scored trials with no exp_key in gt_heatmaps.h5 -- the store and "
             "fixations.json describe different builds; {}".format(
                 _listing(no_exp_key)))
    if unsafe:
        fail("FR3.9 unsafe exp_key(s); {}".format(_listing(unsafe)))
    if missing_pth:
        fail("FR3.9 feature tensors missing under {} -- never skipped into the mean "
             "(D7); {}. {}".format(feat_dir, _listing(missing_pth), GIT_IGNORE_NOTE))
    if bad_sha:
        fail("FR3.4 feature tensor hash mismatch against feature_report.json; "
             "{}".format(_listing(bad_sha)))

    # ---- D4: the dense -> EVE map must be total over the cohort (FR7.5) ----
    to_eve = subject_map.get("to_eve", {})
    absent = [s for s in subjects if str(s) not in to_eve]
    if absent:
        fail("FR7.5 dense subject id(s) absent from subject_id_map['to_eve'] -- "
             "prediction.json could not answer 'which of my real subjects is row "
             "i?'; {}".format(_listing(absent)))

    # ---- FR3.10 heatmap parity under THIS env's numpy -----------------------
    try:
        report["heatmap_parity"] = check_heatmap_parity()
    except EveEvalError as exc:
        report["heatmap_parity"] = {"bitwise_equal": False, "error": str(exc)}
        fail(str(exc))

    # ---- FR12.7 the runtime shape, printed and recorded --------------------
    report["expected_run"] = {
        "batches_at_batch_1": len(by_name),
        "cells": len(scored),
        "subjects_per_image": int(subject_num),
        "participants": len(subjects),
    }
    report["sha256"]["checkpoint"] = sha256_file(checkpoint_path)
    report["failures"] = failures
    report["ok"] = not failures
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="F5 preflight -- every artefact handshake before inference (FR3)")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--senet-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--subject-num", type=int, default=3)
    parser.add_argument("--num-fewshot", type=int, default=10)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fast", action="store_true",
                        help="skip FR3.4's per-tensor hash sweep (existence is still "
                             "checked); preflight.json records the omission")
    parser.add_argument("--out", default=None, help="where to write preflight.json")
    args = parser.parse_args(argv)

    report = check_eval(args.bridge_dir, args.senet_dir, args.feature_dir,
                        args.weights_dir, subject_num=args.subject_num,
                        num_fewshot=args.num_fewshot, fast=args.fast,
                        split=args.split)

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Written on failure too: an absent report is indistinguishable from a run
        # that never started (D5).
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2,
                     sort_keys=True))
    if not report["ok"]:
        sys.stderr.write("FATAL: {} preflight failure(s)\n".format(
            len(report["failures"])))
        for i, msg in enumerate(report["failures"], 1):
            sys.stderr.write("  [{}] {}\n".format(i, msg))
        return 1
    sys.stderr.write(
        "check_eval: OK -- {} cells over {} images, {} participants, "
        "{} at {} subjects/image (feature hashes {})\n".format(
            report["counts"]["n_cells"], report["counts"]["n_images"],
            report["counts"]["n_subjects"], report["split"], args.subject_num,
            "checked" if report["feature_sha256_checked"] else "SKIPPED (--fast)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
