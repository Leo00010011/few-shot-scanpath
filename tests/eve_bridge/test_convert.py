"""Validation Group 3 -- conversion and split (convert.py)."""

import numpy as np
import pytest

from conftest import FakeBundle, make_fake_bundle, make_scanpath
from eve_bridge.convert import RECORD_KEYS, build_fixations


def test_record_counts_and_split_sizes(fake_bundle):
    fixations, _, _, _ = build_fixations(fake_bundle, support_pool_size=2)
    assert len(fixations) == 15
    assert sum(1 for r in fixations if r["split"] == "train") == 6
    assert sum(1 for r in fixations if r["split"] == "test") == 9


def test_splits_are_disjoint_by_stimulus_and_by_record(fake_bundle):
    fixations, _, _, _ = build_fixations(fake_bundle, support_pool_size=2)
    train_names = {r["name"] for r in fixations if r["split"] == "train"}
    test_names = {r["name"] for r in fixations if r["split"] == "test"}
    assert train_names & test_names == set()

    train_pairs = {(r["name"], r["subject"]) for r in fixations if r["split"] == "train"}
    test_pairs = {(r["name"], r["subject"]) for r in fixations if r["split"] == "test"}
    assert train_pairs & test_pairs == set()


def test_key_order_and_constants(fake_bundle):
    fixations, _, _, _ = build_fixations(fake_bundle, support_pool_size=2)
    for rec in fixations:
        assert list(rec.keys()) == RECORD_KEYS
        assert rec["condition"] == "freeview"
        assert rec["task"] == "none"


def test_field_types_are_plain_python(fake_bundle):
    fixations, _, _, _ = build_fixations(fake_bundle, support_pool_size=2)
    for rec in fixations:
        assert len(rec["X"]) == len(rec["Y"]) == len(rec["T"]) == rec["length"]
        assert type(rec["length"]) is int
        assert type(rec["subject"]) is int
        for v in rec["X"] + rec["Y"]:
            assert type(v) is float
        for v in rec["T"]:
            assert type(v) is int


def test_negative_coordinate_is_clamped_and_counted():
    bundle = make_fake_bundle(n_subjects=2, n_stimuli=3, seed=3)
    sp = bundle.get_scanpath("test01_0").copy()
    sp[2][0] = -5.0
    bundle._scanpaths["test01_0"] = sp

    fixations, _, _, counters = build_fixations(bundle, support_pool_size=1)
    assert counters["clamped_coords"] == 1
    rec = [r for r in fixations if r["name"] == "stim00.jpg" and r["subject"] == 0][0]
    assert rec["X"][0] == 1.0


def test_zero_duration_is_raised_to_one_and_counted():
    bundle = make_fake_bundle(n_subjects=2, n_stimuli=3, seed=4)
    sp = bundle.get_scanpath("test01_0").copy()
    sp[1][0] = 0.4
    bundle._scanpaths["test01_0"] = sp

    fixations, _, _, counters = build_fixations(bundle, support_pool_size=1)
    assert counters["zero_duration"] == 1
    rec = [r for r in fixations if r["name"] == "stim00.jpg" and r["subject"] == 0][0]
    assert rec["T"][0] == 1


def test_coordinates_are_shifted_by_exactly_one():
    bundle = make_fake_bundle(n_subjects=2, n_stimuli=3, seed=5)
    sp = bundle.get_scanpath("test01_0").copy()
    # 59.0 -> X = 60.0 -> bin 0; without the +1 it would be 59.0 -> bin 0 too, so
    # pick a value whose bin moves under the shift: 60.0 -> 61.0 -> bin 1 vs bin 0.
    sp[2][0] = 60.0
    sp[3][0] = 45.0
    bundle._scanpaths["test01_0"] = sp

    fixations, _, _, _ = build_fixations(bundle, support_pool_size=1)
    rec = [r for r in fixations if r["name"] == "stim00.jpg" and r["subject"] == 0][0]
    assert rec["X"][0] == pytest.approx(61.0)
    assert rec["Y"][0] == pytest.approx(46.0)
    assert int((rec["X"][0] - 1) // 60.0) == 1      # would be 0 without the +1
    assert int((rec["Y"][0] - 1) // 45.0) == 1


def test_incomplete_stimulus_is_dropped_for_all_subjects():
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=4, seed=6)
    df = bundle._df
    bundle._df = df[df["exp_key"] != "test02_1"].reset_index(drop=True)

    fixations, _, _, counters = build_fixations(bundle, support_pool_size=1)
    assert counters["incomplete_stimulus"] == 1
    assert all(r["name"] != "stim01.jpg" for r in fixations)
    assert len(fixations) == 9


@pytest.mark.parametrize("mode,counter", [
    ("empty", "empty_scanpath"),
    ("nonfinite", "non_finite"),
    ("nostim", "no_stimulus"),
])
def test_per_trial_drop_reasons(mode, counter):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=4, seed=7)
    if mode == "empty":
        bundle._scanpaths["test02_1"] = np.zeros((4, 0), dtype=np.float32)
    elif mode == "nonfinite":
        sp = bundle.get_scanpath("test02_1").copy()
        sp[2][0] = np.nan
        bundle._scanpaths["test02_1"] = sp
    else:
        df = bundle._df.copy()
        df.loc[df["exp_key"] == "test02_1", "stimulus_path"] = ""
        bundle._df = df

    fixations, _, _, counters = build_fixations(bundle, support_pool_size=1)
    assert counters[counter] == 1
    assert sum(v for k, v in counters.items()
               if k in ("empty_scanpath", "non_finite", "no_stimulus")) == 1
    # only that trial is dropped; its stimulus then fails the equal-subject filter
    assert counters["incomplete_stimulus"] == 1
    assert all(r["name"] != "stim01.jpg" for r in fixations)


def test_unknown_subject_raises_with_close_matches(fake_bundle):
    with pytest.raises(ValueError) as exc:
        build_fixations(fake_bundle, unseen_subjects=["test01", "nope"],
                        support_pool_size=2)
    msg = str(exc.value)
    assert "nope" in msg
    assert "test01" in msg or "test02" in msg or "test03" in msg


def test_single_subject_raises(fake_bundle):
    with pytest.raises(ValueError):
        build_fixations(fake_bundle, unseen_subjects=["test01"], support_pool_size=2)


def test_jpg_substring_in_stimulus_name_raises():
    rows, scanpaths = [], {}
    rng = np.random.RandomState(8)
    for si in range(2):
        subject = "test{:02d}".format(si + 1)
        for k, stimulus in enumerate(["a-jpg-thing", "stim01", "stim02"]):
            exp_key = "{}_{}".format(subject, k)
            rows.append((exp_key, subject, stimulus, "test", True, "s/{}.png".format(k)))
            scanpaths[exp_key] = make_scanpath(4, rng)
    bundle = FakeBundle(rows, scanpaths)

    with pytest.raises(ValueError) as exc:
        build_fixations(bundle, support_pool_size=1)
    assert "a-jpg-thing" in str(exc.value)


def test_too_few_stimuli_raises_naming_both_counts():
    bundle = make_fake_bundle(n_subjects=2, n_stimuli=3, seed=9)
    with pytest.raises(ValueError) as exc:
        build_fixations(bundle, support_pool_size=5)
    msg = str(exc.value)
    assert "3" in msg and "6" in msg


def test_seed_determinism_and_seed_sensitivity():
    bundle = make_fake_bundle(n_subjects=2, n_stimuli=8, seed=10)
    a, _, _, _ = build_fixations(bundle, support_pool_size=3, seed=0)
    b, _, _, _ = build_fixations(bundle, support_pool_size=3, seed=0)
    assert a == b

    c, _, _, _ = build_fixations(bundle, support_pool_size=3, seed=1)
    part_a = {r["name"] for r in a if r["split"] == "train"}
    part_c = {r["name"] for r in c if r["split"] == "train"}
    assert part_a != part_c


def test_dense_ids_ignore_argument_order(fake_bundle):
    _, _, subject_id_map, _ = build_fixations(
        fake_bundle, unseen_subjects=["test03", "test01", "test02"],
        support_pool_size=2)
    assert subject_id_map["to_dense"] == {"test01": 0, "test02": 1, "test03": 2}
    assert subject_id_map["to_eve"] == {"0": "test01", "1": "test02", "2": "test03"}


def test_exp_keys_are_aligned_with_records(fake_bundle):
    fixations, exp_keys, subject_id_map, _ = build_fixations(
        fake_bundle, support_pool_size=2)
    assert len(exp_keys) == len(fixations)
    assert len(set(exp_keys)) == len(exp_keys)
    for rec, exp_key in zip(fixations, exp_keys):
        assert exp_key.startswith(subject_id_map["to_eve"][str(rec["subject"])] + "_")
