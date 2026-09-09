"""Preflight the COCO-FreeView fixation labels (FR3.4, FR3.5).

Runs before any GPU allocation is spent. The invariant that matters most is the
equal-subject one (FR3.5c): ``COCOSearch_evaluation`` groups records by
``"{task}/{name}"`` and ``test.py`` slices predictions with
``index // args.subject_num``, so an image carrying the wrong number of subjects
does not crash -- it shifts every subsequent image's predictions against the wrong
ground truth and yields a plausible, wrong number (Mission P2, D4, D7).

Hard invariants (a) (b) (c) (d) (f) raise. The soft counters -- out-of-range
coordinates (e) and short scanpaths (FR14.3) -- are counted and returned, never
raised and never clamped: both are properties of the authors' labels, and short
scanpaths change the frozen evaluator's behaviour rather than invalidating it
(TechStack section 4).

stdlib only -- no torch, no numpy, so it runs on a login node.

CLI: ``py tools/cocofv_prep/check_fixations.py --fix PATH --images DIR
     --fewshot-subject 0 1 2 [--split test]``
Prints the counter dict as JSON on **stdout** (the run script tees it into
``preflight_fixations.json``); everything else goes to stderr.
"""

import argparse
import json
import os
import sys
from collections import Counter

try:
    from . import CocoFvPreflightError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cocofv_prep import CocoFvPreflightError

REQUIRED_KEYS = ("name", "subject", "X", "Y", "T", "length", "split", "task")

#: The ``COCOSearch_evaluation`` frame: labels are natively 512x320 and are fed to
#: the metrics unscaled (FR9.1). Half-open bounds: 511.9 is in, 512.0 is not.
FRAME_WIDTH = 512
FRAME_HEIGHT = 320

_MAX_LISTED = 10


def _listing(items, total=None):
    """Render at most ``_MAX_LISTED`` offenders plus a total, never just a count."""
    shown = list(items)[:_MAX_LISTED]
    total = len(items) if total is None else total
    text = ", ".join(str(_) for _ in shown)
    if total > len(shown):
        text += " ... (+{} more)".format(total - len(shown))
    return "{} total: {}".format(total, text)


def _where(rec):
    return "({}, {}, subject {})".format(rec.get("task"), rec.get("name"), rec.get("subject"))


def check_fixations(fix_path, image_root, fewshot_subjects, split="test"):
    """Validate ``fix_path`` for the query-set run. Returns a JSON-able counter dict.

    Raises :class:`CocoFvPreflightError` on any hard invariant, with the offending
    items enumerated.
    """
    with open(fix_path) as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise CocoFvPreflightError("{}: expected a JSON list, got {}".format(
            fix_path, type(records).__name__))

    # --- FR3.4 schema ----------------------------------------------------
    missing_keys = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise CocoFvPreflightError(
                "{}: record {} is {}, not an object (FR3.4)".format(
                    fix_path, i, type(rec).__name__))
        absent = [k for k in REQUIRED_KEYS if k not in rec]
        if absent:
            missing_keys.append("record {} missing {}".format(i, absent))
    if missing_keys:
        raise CocoFvPreflightError("{}: records missing required keys (FR3.4); {}".format(
            fix_path, _listing(missing_keys[:5], total=len(missing_keys))))

    wanted = set(fewshot_subjects)
    rows = [r for r in records if r["split"] == split and r["subject"] in wanted]
    if not rows:
        raise CocoFvPreflightError(
            "{}: no records with split={!r} and subject in {} (FR3.5)".format(
                fix_path, split, sorted(wanted)))

    # --- (a) length agreement -------------------------------------------
    bad_len = ["{} len(X)={} len(Y)={} len(T)={} length={}".format(
        _where(r), len(r["X"]), len(r["Y"]), len(r["T"]), r["length"])
        for r in rows
        if not (len(r["X"]) == len(r["Y"]) == len(r["T"]) == r["length"])]
    if bad_len:
        raise CocoFvPreflightError(
            "{}: X/Y/T/length disagree (FR3.5a) -- COCOSearch_evaluation.__getitem__ "
            "iterates range(length) and would IndexError; {}".format(
                fix_path, _listing(bad_len[:5], total=len(bad_len))))

    # --- (b) query-subject coverage --------------------------------------
    present = set(r["subject"] for r in rows)
    absent_subjects = sorted(wanted - present)
    if absent_subjects:
        raise CocoFvPreflightError(
            "{}: split={!r} has no records for subject(s) {} (FR3.5b); present: {}".format(
                fix_path, split, absent_subjects, sorted(present)))

    # --- (c) EQUAL-SUBJECT INVARIANT -- the silent corrupter --------------
    counts = Counter((r["task"], r["name"]) for r in rows)
    n_expected = len(fewshot_subjects)
    ragged = ["{}/{}: {} != {}".format(t, nm, c, n_expected)
              for (t, nm), c in sorted(counts.items()) if c != n_expected]
    if ragged:
        raise CocoFvPreflightError(
            "{}: images with the wrong number of subjects (FR3.5c). test.py slices "
            "predictions with index // subject_num, so this misaligns every "
            "subsequent image against the wrong ground truth; {}".format(
                fix_path, _listing(ragged)))

    # --- (d) the str.replace('jpg', 'pth') trap (FR4.4) -------------------
    jpg_tasks = sorted(set(r["task"] for r in rows if "jpg" in r["task"]))
    if jpg_tasks:
        raise CocoFvPreflightError(
            "{}: task name(s) containing the substring 'jpg' (FR3.5d): {}. "
            "COCOSearch_evaluation builds the feature path with an unanchored "
            "'{{task}}/{{name}}'.replace('jpg', 'pth'), which would corrupt the "
            "directory component too.".format(fix_path, jpg_tasks))

    # --- (f) stimulus presence -------------------------------------------
    missing_images = ["{}/{}".format(t, nm) for (t, nm) in sorted(counts)
                      if not os.path.isfile(os.path.join(image_root, t, nm))]
    if missing_images:
        raise CocoFvPreflightError(
            "{}: stimuli missing under {} (FR3.5f); {}".format(
                fix_path, image_root, _listing(missing_images)))

    # --- soft counters: reported, never raised, never clamped -------------
    oob = 0
    for r in rows:
        for x, y in zip(r["X"], r["Y"]):
            if not (0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT):
                oob += 1
    n_short = sum(1 for r in rows if r["length"] < 3)
    lengths = sorted(r["length"] for r in rows)

    return {
        "fix_path": fix_path,
        "split": split,
        "fewshot_subjects": list(fewshot_subjects),
        "n_records_total": len(records),
        "n_rows": len(rows),
        "n_images": len(counts),
        "n_cells": len(rows),
        "n_subjects": len(present),
        "n_tasks": len(set(t for (t, _) in counts)),
        "oob": oob,
        "n_short": n_short,
        "gt_length_min": lengths[0],
        "gt_length_median": lengths[len(lengths) // 2],
        "gt_length_max": lengths[-1],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preflight COCO-FreeView fixation labels")
    parser.add_argument("--fix", dest="fix_path", required=True)
    parser.add_argument("--images", dest="image_root", required=True)
    parser.add_argument("--fewshot-subject", dest="fewshot_subjects", nargs="+",
                        type=int, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    try:
        counters = check_fixations(args.fix_path, args.image_root,
                                   args.fewshot_subjects, args.split)
    except CocoFvPreflightError as exc:
        sys.stderr.write("FATAL preflight failure: {}\n".format(exc))
        return 1
    print(json.dumps(counters, indent=2))
    sys.stderr.write("check_fixations: OK -- {} images x {} subjects = {} cells\n".format(
        counters["n_images"], counters["n_subjects"], counters["n_cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
