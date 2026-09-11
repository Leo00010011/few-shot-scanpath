"""Validation Group 3 -- fix-label construction and the anchor-only override (FR5)."""

import os

import pytest
import torch
import torchvision.transforms as T

from eve_senet import EveSenetError
from eve_senet.dataset import (EVE_ORIGIN_H, EVE_ORIGIN_W, EveSupportDataset,
                               build_fix_labels, rescale_ratios)

from conftest import make_record

IM_H, IM_W = 320, 512


def _labels(recs, **kw):
    return build_fix_labels(recs, im_h=IM_H, im_w=IM_W, **kw)


def test_one_terminal_label_per_record(records):
    subset = [r for r in records if r["subject"] == 0]
    fix_labels, counters = _labels(subset)
    # filter_scanpath keeps only is_last entries and has_stop=True produces exactly
    # one per scanpath; any other count means has_stop was dropped and the terminal
    # label -- the only one carrying the full T list -- is gone
    assert len(fix_labels) == len(subset) == counters["n_records"]


def test_label_unpacks_to_the_nine_tuple(records):
    fix_labels, _ = _labels([records[0]])
    img_name, cat_name, condition, fixs, action, is_last, sid, dura, dataset = fix_labels[0]
    assert is_last is True
    assert cat_name == "none" and condition == "freeview"
    assert img_name == records[0]["name"] and sid == records[0]["subject"]
    assert dataset == "EVE"


def test_dura_is_the_full_T_list_not_a_scalar(records):
    """A scalar here means an intermediate label leaked past filter_scanpath, and
    process_data's len(dura) would raise later."""
    r = records[0]
    fix_labels, _ = _labels([r])
    dura = fix_labels[0][7]
    assert isinstance(dura, list)
    assert dura == r["T"]


def test_rescale_ratios_and_corner(records):
    ratio_w, ratio_h = rescale_ratios(IM_H, IM_W)
    assert abs(ratio_w - IM_W / float(EVE_ORIGIN_W)) < 1e-12
    assert abs(ratio_h - IM_H / float(EVE_ORIGIN_H)) < 1e-12
    r = make_record("a.jpg", 0, [1920.0], [1080.0], [200])
    scaled_x = 1920.0 * ratio_w
    scaled_y = 1080.0 * ratio_h
    assert abs(scaled_x - 512.0) < 1e-9 and abs(scaled_y - 320.0) < 1e-9
    # and the single fixation survives: index 0 is exempt from the bound check
    fix_labels, counters = _labels([r])
    assert counters["oob_fixations_dropped"] == 0
    assert len(fix_labels) == 1


def test_out_of_bound_second_fixation_is_counted_and_dropped():
    r = make_record("a.jpg", 0, [100.0, 1920.0, 200.0], [100.0, 540.0, 200.0],
                    [200, 200, 200])
    fix_labels, counters = _labels([r])
    assert counters["oob_fixations_dropped"] == 1
    assert len(fix_labels[0][3]) == r["length"] - 1  # FR5.5


def test_first_fixation_out_of_bound_is_not_counted():
    r = make_record("a.jpg", 0, [1920.0, 200.0], [1080.0, 200.0], [200, 200])
    _, counters = _labels([r])
    assert counters["oob_fixations_dropped"] == 0  # upstream exempts index 0


def test_length_one_scanpaths_are_counted_not_dropped():
    r = make_record("a.jpg", 0, [400.0], [400.0], [200])
    fix_labels, counters = _labels([r])
    assert counters["len1_scanpaths"] == 1
    assert len(fix_labels) == 1 and len(fix_labels[0][3]) == 1  # FR5.6


def test_over_length_record_raises_rather_than_being_truncated():
    n = 21
    r = make_record("a.jpg", 0, [100.0] * n, [100.0] * n, [200] * n)
    with pytest.raises(EveSenetError) as exc:
        _labels([r], max_traj_length=20)
    assert "max_traj_length" in str(exc.value)  # FR11.8


def test_a_test_split_record_raises():
    r = make_record("a.jpg", 0, [100.0], [100.0], [200], split="test")
    with pytest.raises(EveSenetError) as exc:
        _labels([r])
    assert "split" in str(exc.value)  # FR11.4


def test_first_fixation_is_not_forced_to_the_image_centre():
    """A centre-forced first fixation means the COCO-Search18 convention leaked in and
    every embedding would start from the same point (is_coco_dataset=False)."""
    r = make_record("a.jpg", 0, [300.0, 600.0], [200.0, 400.0], [200, 200])
    ratio_w, ratio_h = rescale_ratios(IM_H, IM_W)
    fix_labels, _ = _labels([r])
    x0, y0 = fix_labels[0][3][0]
    assert abs(x0 - 300.0 * ratio_w) < 1e-9
    assert abs(y0 - 200.0 * ratio_h) < 1e-9
    assert (x0, y0) != (IM_W // 2, IM_H // 2)


def test_empty_input_raises():
    with pytest.raises(EveSenetError):
        _labels([])


# ------------------------------------------------- the whole reason the subclass exists

def _one_subject_dataset(tmp_path, pa_stub):
    from PIL import Image
    recs = [make_record("only{}.jpg".format(k), 0,
                        [100.0 + 10 * k, 300.0], [100.0, 200.0], [200, 300])
            for k in range(4)]
    d = tmp_path / "stim"
    d.mkdir()
    for r in recs:
        Image.new("RGB", (16, 16), (5, 5, 5)).save(d / r["name"])
    pa_stub.image_path = str(d)
    fix_labels, _ = _labels(recs)
    transform = T.Compose([T.Resize((IM_H, IM_W)), T.ToTensor()])
    return fix_labels, transform, str(d)


@pytest.fixture
def pa_stub():
    """The handful of Data hparams process_data actually reads."""
    from common.config import JsonConfig
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return JsonConfig(os.path.join(repo, "SE-Net", "configs", "eve_useremb.json")).Data


def test_getitem_returns_anchor_only_and_does_not_raise(tmp_path, pa_stub):
    fix_labels, transform, _ = _one_subject_dataset(tmp_path, pa_stub)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = EveSupportDataset(os.path.join(repo, "data"), fix_labels, {}, pa_stub,
                           transform, {"none": 0}, torch.device("cpu"))
    item = ds[0]
    assert set(item) == {"anchor"}
    assert item["anchor"]["true_state"].shape[-2:] == (IM_H, IM_W)
    assert len(item["anchor"]["duration"]) == pa_stub.max_traj_length


def test_upstream_getitem_raises_on_a_single_subject_dataset(tmp_path, pa_stub):
    """FR5.2's premise, asserted rather than believed: with one subject the negative
    candidate list is empty and random.choice raises IndexError."""
    from common.data import Siamese_Triplet_Gaze
    fix_labels, transform, _ = _one_subject_dataset(tmp_path, pa_stub)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = Siamese_Triplet_Gaze(os.path.join(repo, "data"), fix_labels, {}, pa_stub,
                              transform, {"none": 0}, torch.device("cpu"))
    with pytest.raises(IndexError):
        ds[0]
