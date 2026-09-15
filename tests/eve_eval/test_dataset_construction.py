"""Group 2 -- ``EVE_evaluation`` construction, scaling and the two new sample keys."""

import numpy as np
import pytest
import torch

from conftest import TRIOS


def test_len_is_the_image_count(make_dataset):
    """FR4.5 -- one batch item yields all three subjects for its image."""
    ds = make_dataset()
    assert len(ds) == len(TRIOS) == 6
    assert len(ds.fixations) == 18


def test_resize_scales_are_the_squash(make_dataset):
    """FR4.1 / FR6.2 -- 3.75 / 2.8125, compared with ``==`` and not approx.

    The second half of this test is the point of it: constructing with the OSIE
    default instead gives 1.5625 / 1.5625, every coordinate is mis-scaled by 2.4x /
    1.8x, and every metric still returns a plausible number (D3).
    """
    ds = make_dataset()
    assert ds.resizescale_x == 3.75
    assert ds.resizescale_y == 2.8125

    wrong = make_dataset(origin_size=(600, 800))
    assert wrong.resizescale_x == 1.5625 and wrong.resizescale_y == 1.5625


def test_downscale_is_the_action_map_grid(make_dataset):
    ds = make_dataset()
    assert ds.downscale_x == 1920 / 32 == 60.0
    assert ds.downscale_y == 1080 / 24 == 45.0


def test_item_image_stack(make_dataset):
    ds = make_dataset()
    item = ds[0]
    assert item["image"].shape == (3, 768, 2048)
    assert item["image"].dtype == torch.float32


def test_item_trial_keys_and_lengths(make_dataset, fixations):
    """FR4.6 -- both new keys, in ``imgid_to_sub`` order alongside ``subject``."""
    ds = make_dataset()
    item = ds[0]
    name = item["img_name"]
    assert isinstance(item["trial_key"], list) and len(item["trial_key"]) == 3
    assert all(isinstance(k, str) for k in item["trial_key"])
    assert item["trial_key"] == ["{}|{}".format(name, s) for s in item["subject"]]

    by_key = {(r["name"], r["subject"]): r for r in fixations}
    assert item["length"].dtype == np.int32 and item["length"].shape == (3,)
    for i, subject in enumerate(item["subject"]):
        expected = min(by_key[(name, int(subject))]["length"], 16)
        assert int(item["length"][i]) == expected


def test_fix_vector_dtype_and_units(make_dataset, fixations):
    """FR4.7 -- resized 512x384 coordinates, durations in SECONDS.

    ``evaluation.py`` multiplies by 1000 itself for ScanMatch; a second conversion
    here would silently scale every duration by a thousand (D3, TechStack section 4).
    """
    ds = make_dataset()
    item = ds[0]
    fv = item["fix_vectors"][0]
    assert fv.dtype.names == ("start_x", "start_y", "duration")
    assert all(fv.dtype[n] == np.dtype("f8") for n in fv.dtype.names)

    rec = next(r for r in fixations
               if r["name"] == item["img_name"] and r["subject"] == item["subject"][0])
    assert fv["start_x"] == pytest.approx(np.float32(rec["X"]) / 3.75, abs=1e-12)
    assert fv["start_y"] == pytest.approx(np.float32(rec["Y"]) / 2.8125, abs=1e-12)
    assert fv["duration"] == pytest.approx(np.float32(rec["T"]) / 1000.0, abs=1e-12)


def test_x_1920_lands_exactly_on_the_screen_edge(make_dataset):
    """A fixation at the far edge maps to 512.0 -- the top of the metric screen."""
    ds = make_dataset()
    item = ds[0]
    assert max(float(fv["start_x"].max()) for fv in item["fix_vectors"]) == 512.0
    assert max(float(fv["start_y"].max()) for fv in item["fix_vectors"]) == 384.0


def test_collate_shapes_and_flattening_order(make_dataset):
    """FR4.6 / FR5.4 / FR8.2 -- the order the whole downstream depends on.

    Row ``k`` of the flattened image tensor must be the tensor for
    ``flat_trial_keys[k]``: the heatmap block indexes ``gt`` that way and the
    prediction writer takes its subject id that way.
    """
    ds = make_dataset()
    batch = ds.collate_func([ds[0], ds[1]])

    assert batch["images"].shape == (2, 3, 768, 2048)
    assert isinstance(batch["trial_keys"], list) and len(batch["trial_keys"]) == 2
    assert all(isinstance(per, list) and len(per) == 3 and
               all(isinstance(k, str) for k in per) for per in batch["trial_keys"])
    assert batch["lengths"].shape == (2, 3)
    assert batch["lengths"].dtype == torch.int32
    assert batch["subjects"].shape == (2, 3)

    flat_keys = [k for per in batch["trial_keys"] for k in per]
    flat_subjects = batch["subjects"].reshape(-1).tolist()
    assert flat_keys == ["{}|{}".format(*k.rsplit("|", 1)) for k in flat_keys]
    assert [int(k.rsplit("|", 1)[1]) for k in flat_keys] == flat_subjects

    flat_images = batch["images"].view(-1, *batch["images"].shape[2:])
    assert flat_images.shape == (6, 768, 2048)
    per_item = [ds[0]["image"], ds[1]["image"]]
    for k in range(6):
        expected = per_item[k // 3][k % 3]
        assert torch.equal(flat_images[k], expected), k
