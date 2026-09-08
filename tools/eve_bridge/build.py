"""CLI: build the EVE bridge artefacts (FR11).

    py tools/eve_bridge/build.py --bundle-dir DIR --out-dir DIR
        [--unseen-subjects test01 test02 ...] [--support-pool-size 20] [--seed 0]
        [--skip-stimuli] [--skip-heatmaps]

Writes ``fixations.json``, ``subject_id_map.json``, ``stimuli/*.jpg``,
``gt_heatmaps.h5`` and ``bridge_report.json`` into ``--out-dir``, then validates
them and exits non-zero on any failure.
"""

import argparse
import json
import os
import sys

if __package__ in (None, ""):                     # allow `py tools/eve_bridge/build.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_bridge.convert import build_fixations, export_stimuli
    from eve_bridge.store import GtHeatmapStore, sha256_file
    from eve_bridge.validate import BridgeValidationError, validate
else:
    from .convert import build_fixations, export_stimuli
    from .store import GtHeatmapStore, sha256_file
    from .validate import BridgeValidationError, validate

ORIGIN_SIZE = (1080, 1920)
ACTION_MAP = (24, 32)
MAX_LENGTH = 16
BLUR_SIGMA = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the EVE -> ISP-SENet bridge artefacts.")
    p.add_argument("--bundle-dir", required=True,
                   help="directory containing bundle.h5 (no default, no absolute "
                        "path baked into the source)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--unseen-subjects", nargs="+", default=None,
                   help="EVE participant ids; default is every id starting 'test'")
    p.add_argument("--support-pool-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-stimuli", action="store_true")
    p.add_argument("--skip-heatmaps", action="store_true")
    return p.parse_args(argv)


def run(args, bundle=None):
    if bundle is None:
        from evedataset import EveBundle
        bundle = EveBundle.load(args.bundle_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    fixations, exp_keys, subject_id_map, counters = build_fixations(
        bundle,
        unseen_subjects=args.unseen_subjects,
        support_pool_size=args.support_pool_size,
        seed=args.seed,
        origin_size=ORIGIN_SIZE,
        max_length=MAX_LENGTH,
    )

    fixations_path = os.path.join(args.out_dir, "fixations.json")
    with open(fixations_path, "w") as fh:
        json.dump(fixations, fh, indent=4)
    fixations_sha256 = sha256_file(fixations_path)

    with open(os.path.join(args.out_dir, "subject_id_map.json"), "w") as fh:
        json.dump(subject_id_map, fh, indent=4)

    counters.setdefault("stimulus_image_conflict", 0)
    if not args.skip_stimuli:
        counters.update(export_stimuli(bundle, fixations, exp_keys, args.out_dir))

    if not args.skip_heatmaps:
        store = GtHeatmapStore.build(
            fixations, exp_keys, subject_id_map,
            attrs={"bundle_dir": args.bundle_dir,
                   "fixations_sha256": fixations_sha256},
            origin_size=ORIGIN_SIZE, action_map=ACTION_MAP,
            max_length=MAX_LENGTH, blur_sigma=BLUR_SIGMA,
        )
        store.save(os.path.join(args.out_dir, "gt_heatmaps.h5"))

    train_names = sorted({r["name"] for r in fixations if r["split"] == "train"})
    test_names = sorted({r["name"] for r in fixations if r["split"] == "test"})

    report = {
        "args": {
            "bundle_dir": args.bundle_dir,
            "out_dir": args.out_dir,
            "unseen_subjects": args.unseen_subjects,
            "support_pool_size": args.support_pool_size,
            "seed": args.seed,
            "skip_stimuli": args.skip_stimuli,
            "skip_heatmaps": args.skip_heatmaps,
        },
        "origin_size": list(ORIGIN_SIZE),
        "action_map": list(ACTION_MAP),
        "max_length": MAX_LENGTH,
        "support_pool_size": args.support_pool_size,
        "num_subjects": len(subject_id_map["to_dense"]),
        "num_stimuli_train": len(train_names),
        "num_stimuli_test": len(test_names),
        "num_trials": len(fixations),
        "counters": counters,
        "fixations_sha256": fixations_sha256,
        "heatmap_metric_denominator": (
            "NSS/CC/KLD average over valid timesteps, not over (image, subject) "
            "cells like the scanpath metrics (FR10.3)"),
    }

    if counters.get("short_scanpath", 0) > 0:
        sys.stderr.write(
            "WARNING (D7): {} trials have fewer than 3 fixations; the frozen "
            "evaluator pads them with (1., 1., 1e-3) before MultiMatch, which "
            "changes what the reported mean means.\n".format(counters["short_scanpath"]))

    report["validation"] = validate(args.out_dir, bundle=bundle)

    with open(os.path.join(args.out_dir, "bridge_report.json"), "w") as fh:
        json.dump(report, fh, indent=4)

    return report


def _print_summary(report):
    print("subjects        : {}".format(report["num_subjects"]))
    print("stimuli (train) : {}".format(report["num_stimuli_train"]))
    print("stimuli (test)  : {}".format(report["num_stimuli_test"]))
    print("trials          : {}".format(report["num_trials"]))
    print("origin_size     : {} (H, W)".format(tuple(report["origin_size"])))
    nonzero = {k: v for k, v in report["counters"].items() if v}
    if nonzero:
        print("counters (non-zero):")
        for k in sorted(nonzero):
            print("  {:24s} {}".format(k, nonzero[k]))
    else:
        print("counters        : all zero")


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except BridgeValidationError as exc:
        sys.stderr.write("BridgeValidationError: {}\n".format(exc))
        return 1
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
