"""Preflight the Stage B image-feature cache (FR4.6).

Asserts that every ``(task, name)`` in the evaluated split has a loadable
``(768, 2048)`` feature tensor at the path ``COCOSearch_evaluation.__getitem__``
would build. The path construction below is a literal copy of the loader's,
unanchored ``str.replace('jpg', 'pth')`` included (FR4.4) -- checking a path the
loader would not build defeats the check.

This is the only torch importer in ``tools/cocofv_prep/``, mirroring
``heatmap_metrics.py``'s role in ``tools/eve_bridge/``. The exit code is load
bearing: the run script's ``if ! check_features`` guard (FR8.6) uses it to decide
whether to spend an hour re-extracting tens of GB of features.

CLI: ``py tools/cocofv_prep/check_features.py --fix PATH --feat-dir DIR [--split test]``
"""

import argparse
import json
import os
import sys

import torch

try:
    from . import CocoFvPreflightError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cocofv_prep import CocoFvPreflightError

EXPECTED_SHAPE = (768, 2048)

_MAX_LISTED = 10


def feature_rel_path(task, name):
    """The relative feature path, built exactly as the loader builds it.

    ``COCOSearch_evaluation.__getitem__`` does
    ``join(feat_dir, img_name.replace('jpg', 'pth'))`` with
    ``img_name == "{task}/{name}"``. Single-sourced here so no third derivation of
    this string exists anywhere in the package.
    """
    return "{}/{}".format(task, name).replace("jpg", "pth")


def _listing(items):
    shown = [str(_) for _ in items[:_MAX_LISTED]]
    text = ", ".join(shown)
    if len(items) > len(shown):
        text += " ... (+{} more)".format(len(items) - len(shown))
    return "{} total: {}".format(len(items), text)


def check_features(fix_path, feat_dir, split="test", expected_shape=EXPECTED_SHAPE):
    """Check the feature cache covers ``split``. Returns a JSON-able summary dict.

    Raises :class:`CocoFvPreflightError` on any missing or misshapen tensor.
    """
    with open(fix_path) as fh:
        records = json.load(fh)
    pairs = sorted(set((r["task"], r["name"]) for r in records if r["split"] == split))
    if not pairs:
        raise CocoFvPreflightError(
            "{}: no records with split={!r} (FR4.5)".format(fix_path, split))

    expected_shape = tuple(expected_shape)
    missing, bad_shape = [], []
    for task, name in pairs:
        rel = feature_rel_path(task, name)
        path = os.path.join(feat_dir, rel)
        if not os.path.isfile(path):
            missing.append(rel)
            continue
        tensor = torch.load(path, map_location="cpu")
        if tuple(tensor.shape) != expected_shape:
            bad_shape.append("{} {}".format(rel, tuple(tensor.shape)))

    if missing:
        raise CocoFvPreflightError(
            "{}: feature tensors missing under {} (FR4.6); {}".format(
                fix_path, feat_dir, _listing(missing)))
    if bad_shape:
        raise CocoFvPreflightError(
            "{}: feature tensors with the wrong shape under {}, expected {} (FR4.3); "
            "{}".format(fix_path, feat_dir, expected_shape, _listing(bad_shape)))

    return {"n_checked": len(pairs), "missing": [], "bad_shape": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preflight the COCO-FreeView feature cache")
    parser.add_argument("--fix", dest="fix_path", required=True)
    parser.add_argument("--feat-dir", dest="feat_dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    try:
        summary = check_features(args.fix_path, args.feat_dir, args.split)
    except CocoFvPreflightError as exc:
        sys.stderr.write("FATAL preflight failure: {}\n".format(exc))
        return 1
    print(json.dumps(summary, indent=2))
    sys.stderr.write("check_features: OK -- {} feature tensors\n".format(summary["n_checked"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
