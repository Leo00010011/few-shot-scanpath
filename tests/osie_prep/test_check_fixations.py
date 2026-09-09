"""Validation groups 2 and 3 -- check_fixations (FR3.4, FR3.5, FR14.3).

Every hard-invariant case asserts both that OsiePreflightError is raised *and*
that the message names the offending item -- "check failed" with no offender is
not a loud failure (D7).
"""

import json

import pytest

from osie_prep import OsiePreflightError
from osie_prep.check_fixations import check_fixations
from osie_fixtures import make_record, make_records, write_fixations, write_images

SUBJECTS = [10, 11, 12, 13, 14]


def run(tmp_path, records, fewshot=SUBJECTS, split="test", images_from=None):
    fix = write_fixations(tmp_path, records)
    root = write_images(tmp_path, images_from if images_from is not None else records)
    return check_fixations(fix, root, fewshot, split)


# --- group 2: hard invariants ----------------------------------------------

def test_a_length_disagreement_raises(tmp_path):
    records = make_records()
    records[4]["T"] = records[4]["T"][:-1]          # len(T)=3, len(X)=len(Y)=length=4
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    msg = str(exc.value)
    assert "FR3.5a" in msg
    assert "subject {}".format(records[4]["subject"]) in msg
    assert records[4]["name"] in msg


def test_a_length_field_lying_raises(tmp_path):
    # len(X) == len(Y) == len(T) == 4 but length == 5. A naive
    # len(X)==len(Y)==len(T) check misses it, and __getitem__ iterates
    # range(length), so the loader would IndexError.
    records = make_records()
    records[0]["length"] = 5
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    assert "length=5" in str(exc.value)


def test_b_missing_query_subject_raises(tmp_path):
    records = make_records(subjects=(10, 11, 12, 13))
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    msg = str(exc.value)
    assert "FR3.5b" in msg
    assert "[14]" in msg


def test_c_ragged_too_few_raises(tmp_path):
    records = make_records()
    dropped = records.pop(0)                         # first image loses one subject
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    msg = str(exc.value)
    assert "FR3.5c" in msg
    assert "{}: 4 != 5".format(dropped["name"]) in msg


def test_c_ragged_too_many_raises(tmp_path):
    # A duplicate (name, subject) is as fatal as an omission: it shifts the slice.
    records = make_records()
    records.append(make_record(name=records[0]["name"], subject=10))
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    assert "{}: 6 != 5".format(records[0]["name"]) in str(exc.value)


def test_c_groups_on_the_bare_name_not_task_and_name(tmp_path):
    # OSIE_evaluation does imgid_to_sub.setdefault(fixation['name'], ...) -- unlike
    # the COCO branches there is no task component in the key. Every OSIE record
    # carries the same constant task="none", so a task-aware grouping would still
    # pass here; what this pins is that the counter dict has no task axis at all.
    counters = run(tmp_path, make_records())
    assert "n_tasks" not in counters
    assert counters["n_images"] == 3


def test_c_passes_on_a_well_formed_fixture(tmp_path):
    counters = run(tmp_path, make_records())
    assert counters["n_images"] == 3
    assert counters["n_cells"] == 15


def test_d_jpg_inside_a_stimulus_name_raises(tmp_path):
    records = make_records()
    for rec in records:
        if rec["name"] == "1001.jpg":
            rec["name"] = "myjpg1001.jpg"
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    assert "FR3.5d" in str(exc.value)
    assert "myjpg1001.jpg" in str(exc.value)
    # Why it is fatal: the loader's replace is unanchored, so the stem is corrupted
    # too and the path can never exist.
    assert "myjpg1001.jpg".replace("jpg", "pth") == "mypth1001.pth"


def test_d_a_plain_name_is_accepted(tmp_path):
    assert run(tmp_path, make_records())["n_images"] == 3
    assert "1001.jpg".replace("jpg", "pth") == "1001.pth"


def test_f_missing_stimulus_file_raises(tmp_path):
    records = make_records()
    # Write images for everything except the last image.
    present = [r for r in records if r["name"] != records[-1]["name"]]
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records, images_from=present)
    msg = str(exc.value)
    assert "FR3.5f" in msg
    assert records[-1]["name"] in msg


def test_split_filtering_happens_before_checking(tmp_path):
    clean_test = make_records(split="test")
    ragged_train = make_records(split="train")[:-1]   # deliberately ragged
    records = clean_test + ragged_train

    assert check_fixations(
        write_fixations(tmp_path, records), write_images(tmp_path, records),
        SUBJECTS, "test")["n_images"] == 3
    with pytest.raises(OsiePreflightError):
        check_fixations(write_fixations(tmp_path, records),
                        write_images(tmp_path, records), SUBJECTS, "train")


def test_subject_filtering_matches_select_fewshot_subject(tmp_path):
    # Subjects 10-14 are the query set; base-set subjects 0-9 appear on only one
    # image each. The checker must filter to the query set the way
    # select_fewshot_subject() does, before the equal-subject count.
    records = make_records()
    for subject in range(0, 10):
        records.append(make_record(name=records[0]["name"], subject=subject))
    assert run(tmp_path, records, fewshot=SUBJECTS)["n_images"] == 3


# --- group 3: soft counters -------------------------------------------------

def test_e_out_of_range_coordinates_are_counted_not_raised(tmp_path):
    records = make_records()
    records[0]["X"][0] = 800.0
    records[1]["Y"][0] = -1.0
    counters = run(tmp_path, records)
    assert counters["oob"] == 2


def test_e_bounds_are_the_native_osie_frame_and_half_open(tmp_path):
    # 800x600, not COCO-FreeView's 512x320: test.py never passes origin_size, so
    # OSIE_evaluation's (600, 800) default applies (D3).
    records = make_records()
    records[0]["X"][0] = 799.9
    records[0]["Y"][0] = 599.9
    assert run(tmp_path, records)["oob"] == 0

    records[0]["X"][0] = 800.0
    assert run(tmp_path, records)["oob"] == 1

    # A coordinate legal in the 800x600 frame but outside COCO's 512x320 one must
    # not be counted -- this is exactly the retarget that could silently regress.
    records[0]["X"][0] = 700.0
    records[0]["Y"][0] = 400.0
    assert run(tmp_path, records)["oob"] == 0


def test_e_origin_bounds_are_overridable(tmp_path):
    records = make_records()
    records[0]["X"][0] = 700.0
    fix = write_fixations(tmp_path, records)
    root = write_images(tmp_path, records)
    assert check_fixations(fix, root, SUBJECTS, "test",
                           origin_width=512, origin_height=320)["oob"] > 0


def test_origin_size_is_reported_as_height_width(tmp_path):
    # Reported in the (H, W) order OSIE_evaluation takes it, so the counter dict
    # can be pasted straight into notes.md without a transposition.
    assert run(tmp_path, make_records())["origin_size"] == [600, 800]


def test_n_short_counts_scanpaths_below_three(tmp_path):
    # One subject, four images of lengths 1, 2, 3, 4 -- length == 3 is not short.
    records = [make_record(name="{}.jpg".format(1001 + k), subject=10, length=length)
               for k, length in enumerate([1, 2, 3, 4])]
    counters = run(tmp_path, records, fewshot=[10])
    assert counters["n_short"] == 2
    assert counters["gt_length_min"] == 1
    assert counters["gt_length_max"] == 4


def test_counters_are_json_serialisable(tmp_path):
    counters = run(tmp_path, make_records())
    assert json.loads(json.dumps(counters)) == counters


# --- group 2 (continued): invariant (g), the duration-bin trap ---------------

def test_g_binned_durations_raise(tmp_path):
    # data/osie_fixations_update_duration.json carries identical X/Y to the branch's
    # fixations.json but a T of 0..9 -- the decile bin index of the duration. It
    # would not crash the evaluator; it would silently invalidate SM and MM.
    records = make_records()
    for rec in records:
        rec["T"] = [i % 10 for i in range(rec["length"])]
    with pytest.raises(OsiePreflightError) as exc:
        run(tmp_path, records)
    msg = str(exc.value)
    assert "FR3.5g" in msg
    assert "osie_fixations_update_duration.json" in msg


def test_g_real_millisecond_durations_pass(tmp_path):
    records = make_records()
    for rec in records:
        rec["T"] = [246, 136, 179, 1975][:rec["length"]]
    counters = run(tmp_path, records)
    assert counters["t_min_ms"] == 136
    assert counters["t_max_ms"] == 1975


def test_g_checks_the_evaluated_split_only(tmp_path):
    # A binned train split must not condemn a millisecond test split, and vice
    # versa -- (g) runs after the split/subject filter like every other invariant.
    test_rows = make_records(split="test")
    train_rows = make_records(split="train")
    for rec in train_rows:
        rec["T"] = [1] * rec["length"]
    records = test_rows + train_rows

    fix = write_fixations(tmp_path, records)
    root = write_images(tmp_path, records)
    assert check_fixations(fix, root, SUBJECTS, "test")["t_max_ms"] == 203
    with pytest.raises(OsiePreflightError) as exc:
        check_fixations(fix, root, SUBJECTS, "train")
    assert "FR3.5g" in str(exc.value)
