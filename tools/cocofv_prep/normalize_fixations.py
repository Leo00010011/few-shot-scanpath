"""Normalize the authors' COCO-FreeView label file (FR3.3).

Reproduces exactly the three transformations
``ISP/COCO_FV/GazeformerISP/src/preprocess/preprocess_fixations.py`` performs, and
nothing more -- that script hardcodes both its input and its output path, so it is
unusable from a run script (FR3.1). This tool takes ``--in`` / ``--out`` instead so
no path is hardcoded (FR2.3).

Running it on an already-normalized file is a no-op that reports three zero
counters, which is itself worth logging.

CLI: ``py tools/cocofv_prep/normalize_fixations.py --in PATH --out PATH``
"""

import argparse
import json
import os
import sys

_TASK_FIXES = {"potted plant": "potted_plant", "stop sign": "stop_sign"}


def normalize(records):
    """Return a new list of records with ``split``/``task`` normalized.

    Input records are not mutated: ``preprocess_fixations.py`` mutates in place,
    but it owns its data and we do not.
    """
    out = []
    for r in records:
        r = dict(r)
        if r.get("split") == "val":
            r["split"] = "validation"
        r["task"] = _TASK_FIXES.get(r.get("task"), r.get("task"))
        out.append(r)
    return out


def count_transformations(records):
    """Return how many records each of the three fixes would touch."""
    counts = {"split_val_to_validation": 0, "potted plant": 0, "stop sign": 0}
    for r in records:
        if r.get("split") == "val":
            counts["split_val_to_validation"] += 1
        task = r.get("task")
        if task in _TASK_FIXES:
            counts[task] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalize COCO-FreeView fixation labels")
    parser.add_argument("--in", dest="in_path", required=True,
                        help="the authors' COCO-FreeView label JSON")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="where to write the normalized fixations.json")
    args = parser.parse_args(argv)

    with open(args.in_path) as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        print("FATAL: {}: expected a JSON list, got {}".format(
            args.in_path, type(records).__name__), file=sys.stderr)
        return 1

    counts = count_transformations(records)
    normalized = normalize(records)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w") as fh:
        json.dump(normalized, fh, indent=2)

    print("normalize_fixations: {} records -> {}".format(len(normalized), args.out_path))
    for key, value in counts.items():
        print("  {:26}: {}".format(key, value))
    if not any(counts.values()):
        print("  (all zero: the input was already normalized)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
