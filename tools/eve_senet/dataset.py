"""EVE support scanpaths -> SE-Net fix-labels and per-item tensors (FR5).

Everything that shapes a tensor here is the authors' own code, reached by import or
by subclassing. This module contributes exactly three things they do not:

1. the 1920x1080 -> 512x320 rescale (FR5.3) that ``common/dataset.py::process_data``
   would have done, had it a branch for EVE (it raises ``NotImplementedError``
   instead, and adding one would edit a shared upstream file -- convention 2);
2. an out-of-bound **count** (FR5.5), because ``preprocess_fixations`` drops such
   fixations silently and silence is the D7 failure mode;
3. an anchor-only ``__getitem__`` (FR5.2), because the shipped triplet sampler
   cannot run on a single-subject dataset -- see :class:`EveSupportDataset`.

Imports SE-Net, so this module is cluster-side in spirit; it happens to import
cleanly on CPU because ``common/{utils,data}.py`` need neither Detectron2 nor
MSDeformAttn (only ``src/models.py`` does).
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENET_DIR = os.path.join(REPO_ROOT, "SE-Net")
if SENET_DIR not in sys.path:  # SE-Net expects to be importable as its own root
    sys.path.insert(0, SENET_DIR)

from common.data import Siamese_Triplet_Gaze  # noqa: E402
from common.utils import filter_scanpath, preprocess_fixations  # noqa: E402

try:
    from . import EveSenetError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_senet import EveSenetError

# EVE's native stimulus resolution (D3 / bridge_report["origin_size"] as (H, W)).
EVE_ORIGIN_H, EVE_ORIGIN_W = 1080, 1920


class EveSupportDataset(Siamese_Triplet_Gaze):
    """Anchor only (FR5.2). Everything else is inherited byte-for-byte.

    ``Siamese_Triplet_Gaze.__getitem__`` builds a full (anchor, positive, negative)
    triplet regardless of eval mode, drawing the negative from a *different* subject::

        random.choice([i for i, d in enumerate(self.fix_labels) if d[-3] != anchor_sid])

    F3 invokes SE-Net **one subject at a time** (FR4.3 -- the shipped
    ``select_fewshot_subject`` draws from the union across subjects, which on our
    stimulus-disjoint support pools would give each subject ~10/38 scanpaths). With a
    single subject in the dataset that comprehension is empty and ``random.choice``
    raises ``IndexError``. Upstream's ``num_fewshot == 1`` short-circuit would dodge
    it, but it is gated on the shot count, not on eval mode, so at the paper's n = 10
    the draw is live.

    The positive and negative exist for the *training* triplet loss;
    ``evaluate_user_siamese`` does ``batch = batch['anchor']`` and never reads them.
    So this override restores the authors' own semantics for this stage -- and drops
    two thirds of the image loads along with the crash.
    """

    def __getitem__(self, idx):
        return {"anchor": self.process_data(idx)}


def build_fix_labels(records, im_h=320, im_w=512, max_traj_length=20,
                     patch_size=(16, 16), patch_num=(32, 20)):
    """Rescale -> ``preprocess_fixations`` -> ``filter_scanpath``.

    ``records`` are ``fixations.json`` records (FR2.1), already decile-binned by
    :mod:`durations` if the caller wants FR3's input contract. Returns
    ``(fix_labels, counters)`` with exactly one terminal label per record.
    """
    counters = {"n_records": len(records), "oob_fixations_dropped": 0,
                "len1_scanpaths": 0, "truncated_scanpaths": 0}
    if not records:
        raise EveSenetError("build_fix_labels: no records (FR5)")

    ratio_w = im_w / float(EVE_ORIGIN_W)   # 512/1920  = 0.266666...
    ratio_h = im_h / float(EVE_ORIGIN_H)   # 320/1080  = 0.296296...

    scaled = []
    for r in records:
        if r["split"] != "train":  # FR11.4 -- a support/query leak, never a filter
            raise EveSenetError(
                "build_fix_labels: record {!r} subject {} has split={!r}, expected "
                "'train'. Reading a scored record into the support set invalidates "
                "every cell of F5's score matrix (FR2.2/FR11.4)".format(
                    r["name"], r["subject"], r["split"]))
        if r["length"] > max_traj_length:  # FR11.8 -- never silently truncated
            raise EveSenetError(
                "build_fix_labels: record {!r} subject {} has length {} > "
                "max_traj_length {} (FR11.8)".format(
                    r["name"], r["subject"], r["length"], max_traj_length))
        if r["length"] == 1:
            counters["len1_scanpaths"] += 1

        s = dict(r)
        s["X"] = [float(x) * ratio_w for x in r["X"]]
        s["Y"] = [float(y) * ratio_h for y in r["Y"]]
        s["dataset"] = "EVE"
        scaled.append(s)

        # Count what preprocess_fixations is about to drop, BEFORE it runs (FR5.5),
        # so the counter is independent of upstream behaviour rather than inferred
        # from it. Index 0 is exempt by upstream design and is not counted.
        for i in range(1, len(s["X"])):
            if not (0 <= s["X"][i] < im_w and 0 <= s["Y"][i] < im_h):
                counters["oob_fixations_dropped"] += 1

    fix_labels = preprocess_fixations(
        scaled,
        patch_size=list(patch_size),
        patch_num=list(patch_num),
        im_h=im_h,
        im_w=im_w,
        truncate_num=max_traj_length,
        has_stop=True,          # the ONLY source of an is_last label, and the
                                # stop_label's `dura` is the record's full T list
        sample_scanpath=False,
        min_traj_length_percentage=0,
        discretize_fix=False,
        remove_return_fixations=False,
        is_coco_dataset=False,  # EVE keeps its own first fixation; forcing the
                                # image centre is a COCO-Search18 convention
    )
    fix_labels = filter_scanpath(fix_labels)
    if len(fix_labels) != len(records):
        raise EveSenetError(
            "build_fix_labels: {} terminal labels for {} records -- expected exactly "
            "one per scanpath. A different count means has_stop was dropped and the "
            "only label carrying the full T list is gone (FR5.1)".format(
                len(fix_labels), len(records)))
    return fix_labels, counters


def rescale_ratios(im_h=320, im_w=512):
    """FR5.3's ratios, for the report. A *second, different* squash from F5's."""
    return [im_w / float(EVE_ORIGIN_W), im_h / float(EVE_ORIGIN_H)]
