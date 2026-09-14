"""Preflight the per-trial feature cache (FR7).

Asserts that every trial in the selected split(s) has a loadable ``(768, 2048)``
float32 tensor at the path the F5 loader would build -- ``join(feat_dir, exp_key +
'.pth')``, by concatenation, with no ``str.replace('jpg', 'pth')`` anywhere on the EVE
path (FR5.2).

The exit code is load bearing: ``bash/extract_eve_features.sh``'s
``if ! check_features`` guard (FR9.6) uses it to decide whether to spend GPU hours,
and the same script re-runs it *after* extraction where a failure is fatal.

This module deliberately does **not** import ``evedataset`` or open the bundle: the
guard has to be runnable before the bundle is staged on the cluster.

CLI: ``py tools/eve_prep/check_features.py --fix PATH --heatmaps PATH --feat-dir DIR
[--split both]``
"""

import argparse
import json
import os
import sys

import torch

try:
    from . import EvePrepError, FEATURE_SHAPE
    from .trial_keys import (exp_key_filename, load_trial_exp_keys, load_json,
                             split_of, _listing)
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_prep import EvePrepError, FEATURE_SHAPE
    from eve_prep.trial_keys import (exp_key_filename, load_trial_exp_keys,
                                     load_json, split_of, _listing)

SPLITS = ("both", "test", "train")


def select_trials(fixations, split="both"):
    """The ``(name, subject)`` trials in ``split``, sorted. Raises on an empty set."""
    if split not in SPLITS:
        raise EvePrepError(
            "unknown split {!r}; expected one of {} (FR5.4)".format(split, SPLITS))
    splits = split_of(fixations)
    wanted = sorted(k for k, s in splits.items() if split == "both" or s == split)
    if not wanted:
        raise EvePrepError("no records with split={!r} (FR7.1)".format(split))
    return wanted, splits


def check_features(fixations_path, heatmaps_path, feat_dir, split="both",
                   expected_shape=FEATURE_SHAPE):
    """Check the cache covers ``split``. Returns a JSON-able summary dict.

    Raises :class:`EvePrepError` on a hash mismatch (FR2.5), a phantom trial (FR2.1),
    an unsafe exp_key (FR7.4), or any missing or misshapen tensor (FR7.2). Missing
    files and wrong shapes are collected **separately**: they mean different things --
    an incomplete run versus a corrupt or stale one.
    """
    fixations = load_json(fixations_path)
    exp_of = load_trial_exp_keys(heatmaps_path, fixations_path)   # FR2.5 fires here
    wanted, splits = select_trials(fixations, split)

    expected_shape = tuple(expected_shape)
    missing, bad_shape = [], []
    for key in wanted:
        if key not in exp_of:
            raise EvePrepError(
                "trial {} is in {} but not in the heatmap store -- the mapping and "
                "the records describe different builds (FR2.1)".format(
                    key, fixations_path))
        rel = exp_key_filename(exp_of[key])          # FR7.4 charset + no-jpg rule
        path = os.path.join(feat_dir, rel)
        if not os.path.isfile(path):
            missing.append(rel)
            continue
        tensor = torch.load(path, map_location="cpu")
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32:
            bad_shape.append("{} {} {}".format(rel, tuple(tensor.shape), tensor.dtype))

    if missing:
        raise EvePrepError(
            "feature tensors missing under {} (FR7.2); {}".format(
                feat_dir, _listing(missing)))
    if bad_shape:
        raise EvePrepError(
            "feature tensors with the wrong shape or dtype under {}, expected {} "
            "float32 (FR7.2); {}".format(feat_dir, expected_shape, _listing(bad_shape)))

    return {
        "n_checked": len(wanted),
        "n_test": sum(1 for k in wanted if splits[k] == "test"),
        "n_train": sum(1 for k in wanted if splits[k] == "train"),
        "split": split,
        "missing": [],
        "bad_shape": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preflight the EVE per-trial feature cache (F4, FR7)")
    parser.add_argument("--fix", dest="fixations_path", required=True)
    parser.add_argument("--heatmaps", dest="heatmaps_path", required=True)
    parser.add_argument("--feat-dir", dest="feat_dir", required=True)
    parser.add_argument("--split", default="both", choices=list(SPLITS))
    args = parser.parse_args(argv)

    try:
        summary = check_features(args.fixations_path, args.heatmaps_path,
                                 args.feat_dir, args.split)
    except EvePrepError as exc:
        sys.stderr.write("FATAL preflight failure: {}\n".format(exc))
        return 1
    print(json.dumps(summary, indent=2))
    sys.stderr.write(
        "check_features: OK -- {} feature tensors ({} test / {} train)\n".format(
            summary["n_checked"], summary["n_test"], summary["n_train"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
