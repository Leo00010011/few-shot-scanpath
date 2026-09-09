"""Preflight the OSIE fixation labels (FR3.4, FR3.5).

Runs before any GPU allocation is spent. The invariant that matters most is the
equal-subject one (FR3.5c): ``OSIE_evaluation`` groups records by ``fixation['name']``
and ``test.py`` slices predictions with ``index // args.subject_num``, so an image
carrying the wrong number of subjects does not crash -- it shifts every subsequent
image's predictions against the wrong ground truth and yields a plausible, wrong
number (Mission P2, D4, D7).

Hard invariants (a) (b) (c) (d) (f) raise. The soft counters -- out-of-range
coordinates (e) and short scanpaths (FR14.3) -- are counted and returned, never
raised and never clamped: both are properties of the authors' labels, and short
scanpaths change the frozen evaluator's behaviour rather than invalidating it
(TechStack section 4).

stdlib only -- no torch, no numpy, so it runs on a login node.

CLI: ``py tools/osie_prep/check_fixations.py --fix PATH --images DIR
     --fewshot-subject 10 11 12 13 14 [--split test]``
Prints the counter dict as JSON on **stdout** (the run script tees it into
``preflight_fixations.json``); everything else goes to stderr.
"""

import argparse
import json
import os
import sys
from collections import Counter

try:
    from . import OsiePreflightError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from osie_prep import OsiePreflightError

REQUIRED_KEYS = ("name", "subject", "X", "Y", "T", "length", "split")

#: The ``OSIE_evaluation`` frame. ``test.py`` does not pass ``origin_size``, so the
#: loader's ``(600, 800)`` default applies and coordinates are native 800x600 before
#: the rescale to the 512x384 metric screen (D3). Half-open bounds: 799.9 is in,
#: 800.0 is not. Overridable, because the default is the thing most likely to be
#: wrong on a re-export.
ORIGIN_WIDTH = 800
ORIGIN_HEIGHT = 600

#: Floor for "is ``T`` actually in milliseconds?" (FR3.5g). The shipped
#: ``src/data/fixations.json`` spans 20..1975 ms. The *other* OSIE label file in this
#: repo -- ``data/osie_fixations_update_duration.json`` -- carries identical X/Y and
#: identical keys but a ``T`` of 0..9: the **decile bin index** of the duration, a
#: training target for the duration head, not a duration. Verified 2026-09-09 by
#: joining the two files on (name, subject): bin 0 covers 20-96 ms, bin 9 covers
#: 358-1975 ms, ten equal-count buckets. Feeding it to the evaluator would not crash
#: -- it would silently destroy ScanMatch-with-duration (TempBin=50 assumes ms) and
#: MultiMatch's duration dimension, and therefore both headline SM and MM.
MIN_PLAUSIBLE_MAX_DURATION_MS = 20

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
    return "({}, subject {})".format(rec.get("name"), rec.get("subject"))


def check_fixations(fix_path, image_root, fewshot_subjects, split="test",
                    origin_width=ORIGIN_WIDTH, origin_height=ORIGIN_HEIGHT):
    """Validate ``fix_path`` for the query-set run. Returns a JSON-able counter dict.

    Raises :class:`OsiePreflightError` on any hard invariant, with the offending
    items enumerated.
    """
    with open(fix_path) as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise OsiePreflightError("{}: expected a JSON list, got {}".format(
            fix_path, type(records).__name__))

    # --- FR3.4 schema ----------------------------------------------------
    missing_keys = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise OsiePreflightError(
                "{}: record {} is {}, not an object (FR3.4)".format(
                    fix_path, i, type(rec).__name__))
        absent = [k for k in REQUIRED_KEYS if k not in rec]
        if absent:
            missing_keys.append("record {} missing {}".format(i, absent))
    if missing_keys:
        raise OsiePreflightError("{}: records missing required keys (FR3.4); {}".format(
            fix_path, _listing(missing_keys[:5], total=len(missing_keys))))

    wanted = set(fewshot_subjects)
    rows = [r for r in records if r["split"] == split and r["subject"] in wanted]
    if not rows:
        raise OsiePreflightError(
            "{}: no records with split={!r} and subject in {} (FR3.5)".format(
                fix_path, split, sorted(wanted)))

    # --- (a) length agreement -------------------------------------------
    bad_len = ["{} len(X)={} len(Y)={} len(T)={} length={}".format(
        _where(r), len(r["X"]), len(r["Y"]), len(r["T"]), r["length"])
        for r in rows
        if not (len(r["X"]) == len(r["Y"]) == len(r["T"]) == r["length"])]
    if bad_len:
        raise OsiePreflightError(
            "{}: X/Y/T/length disagree (FR3.5a) -- OSIE_evaluation.__getitem__ "
            "iterates range(length) and would IndexError; {}".format(
                fix_path, _listing(bad_len[:5], total=len(bad_len))))

    # --- (b) query-subject coverage --------------------------------------
    present = set(r["subject"] for r in rows)
    absent_subjects = sorted(wanted - present)
    if absent_subjects:
        raise OsiePreflightError(
            "{}: split={!r} has no records for subject(s) {} (FR3.5b); present: {}".format(
                fix_path, split, absent_subjects, sorted(present)))

    # --- (c) EQUAL-SUBJECT INVARIANT -- the silent corrupter --------------
    # OSIE keys on the bare name; there is no task component in the group key.
    counts = Counter(r["name"] for r in rows)
    n_expected = len(fewshot_subjects)
    ragged = ["{}: {} != {}".format(nm, c, n_expected)
              for nm, c in sorted(counts.items()) if c != n_expected]
    if ragged:
        raise OsiePreflightError(
            "{}: images with the wrong number of subjects (FR3.5c). test.py slices "
            "predictions with index // subject_num, so this misaligns every "
            "subsequent image against the wrong ground truth; {}".format(
                fix_path, _listing(ragged)))

    # --- (d) the str.replace('jpg', 'pth') trap (FR4.4) -------------------
    # OSIE_evaluation builds the feature path with an unanchored
    # name.replace('jpg', 'pth'). A name is safe only when 'jpg' occurs exactly once
    # and is the extension: "myjpg01.jpg" would yield "mypth01.pth".
    bad_names = sorted(nm for nm in counts
                       if not nm.endswith(".jpg") or nm.count("jpg") != 1)
    if bad_names:
        raise OsiePreflightError(
            "{}: stimulus name(s) that the unanchored replace('jpg', 'pth') would "
            "corrupt (FR3.5d); a name must end in '.jpg' and contain 'jpg' exactly "
            "once; {}".format(fix_path, _listing(bad_names)))

    # --- (f) stimulus presence (flat directory, no category component) ----
    missing_images = [nm for nm in sorted(counts)
                      if not os.path.isfile(os.path.join(image_root, nm))]
    if missing_images:
        raise OsiePreflightError(
            "{}: stimuli missing under {} (FR3.5f); {}".format(
                fix_path, image_root, _listing(missing_images)))

    # --- (g) T IS IN MILLISECONDS, NOT DURATION BINS -- the other silent corrupter
    durations = [t for r in rows for t in r["T"]]
    t_max = max(durations)
    if t_max <= MIN_PLAUSIBLE_MAX_DURATION_MS:
        raise OsiePreflightError(
            "{}: max T over split={!r} is {}, which cannot be a fixation duration in "
            "milliseconds (FR3.5g). This is the signature of "
            "data/osie_fixations_update_duration.json, whose T is the decile BIN "
            "INDEX (0-9) of the duration, not the duration. Point --fix at the "
            "branch's own src/data/fixations.json (T spans 20-1975 ms). Using bins "
            "would not crash -- it would silently invalidate ScanMatch-with-duration "
            "and MultiMatch's duration dimension, and so both SM and MM.".format(
                fix_path, split, t_max))

    # --- soft counters: reported, never raised, never clamped -------------
    oob = 0
    for r in rows:
        for x, y in zip(r["X"], r["Y"]):
            if not (0 <= x < origin_width and 0 <= y < origin_height):
                oob += 1
    n_short = sum(1 for r in rows if r["length"] < 3)
    lengths = sorted(r["length"] for r in rows)

    return {
        "fix_path": fix_path,
        "split": split,
        "fewshot_subjects": list(fewshot_subjects),
        "origin_size": [origin_height, origin_width],
        "n_records_total": len(records),
        "n_rows": len(rows),
        "n_images": len(counts),
        "n_cells": len(rows),
        "n_subjects": len(present),
        "t_min_ms": min(durations),
        "t_max_ms": t_max,
        "oob": oob,
        "n_short": n_short,
        "gt_length_min": lengths[0],
        "gt_length_median": lengths[len(lengths) // 2],
        "gt_length_max": lengths[-1],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preflight OSIE fixation labels")
    parser.add_argument("--fix", dest="fix_path", required=True)
    parser.add_argument("--images", dest="image_root", required=True)
    parser.add_argument("--fewshot-subject", dest="fewshot_subjects", nargs="+",
                        type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--origin-width", type=int, default=ORIGIN_WIDTH)
    parser.add_argument("--origin-height", type=int, default=ORIGIN_HEIGHT)
    args = parser.parse_args(argv)

    try:
        counters = check_fixations(args.fix_path, args.image_root,
                                   args.fewshot_subjects, args.split,
                                   args.origin_width, args.origin_height)
    except OsiePreflightError as exc:
        sys.stderr.write("FATAL preflight failure: {}\n".format(exc))
        return 1
    print(json.dumps(counters, indent=2))
    sys.stderr.write("check_fixations: OK -- {} images x {} subjects = {} cells\n".format(
        counters["n_images"], counters["n_subjects"], counters["n_cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
