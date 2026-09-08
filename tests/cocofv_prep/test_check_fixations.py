"""Validation groups 2 and 3 -- check_fixations (FR3.4, FR3.5, FR14.3).

Every hard-invariant case asserts both that CocoFvPreflightError is raised *and*
that the message names the offending item -- "check failed" with no offender is
not a loud failure (D7).
"""

import json

import pytest

from cocofv_prep import CocoFvPreflightError
from cocofv_prep.check_fixations import check_fixations
from cocofv_fixtures import make_record, make_records, write_fixations, write_images

SUBJECTS = [0, 1, 2]


def run(tmp_path, records, fewshot=SUBJECTS, split="test", images_from=None):
    fix = write_fixations(tmp_path, records)
    root = write_images(tmp_path, images_from if images_from is not None else records)
    return check_fixations(fix, root, fewshot, split)


# --- group 2: hard invariants ----------------------------------------------

def test_a_length_disagreement_raises(tmp_path):
    records = make_records()
    records[4]["T"] = records[4]["T"][:-1]          # len(T)=3, len(X)=len(Y)=length=4
    with pytest.raises(CocoFvPreflightError) as exc:
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
    with pytest.raises(CocoFvPreflightError) as exc:
        run(tmp_path, records)
    assert "length=5" in str(exc.value)


def test_b_missing_query_subject_raises(tmp_path):
    records = make_records(subjects=(0, 1))
    with pytest.raises(CocoFvPreflightError) as exc:
        run(tmp_path, records)
    msg = str(exc.value)
    assert "FR3.5b" in msg
    assert "[2]" in msg


def test_c_ragged_too_few_raises(tmp_path):
    records = make_records()
    dropped = records.pop(0)                         # first image loses one subject
    with pytest.raises(CocoFvPreflightError) as exc:
        run(tmp_path, records)
    msg = str(exc.value)
    assert "FR3.5c" in msg
    assert "{}/{}: 2 != 3".format(dropped["task"], dropped["name"]) in msg


def test_c_ragged_too_many_raises(tmp_path):
    # A duplicate (name, subject) is as fatal as an omission: it shifts the slice.
    records = make_records()
    records.append(make_record(task=records[0]["task"], name=records[0]["name"],
                               subject=0))
    with pytest.raises(CocoFvPreflightError) as exc:
        run(tmp_path, records)
    assert "{}/{}: 4 != 3".format(records[0]["task"], records[0]["name"]) in str(exc.value)


def test_c_passes_on_a_well_formed_fixture(tmp_path):
    counters = run(tmp_path, make_records())
    assert counters["n_images"] == 3
    assert counters["n_cells"] == 9


def test_d_jpg_in_a_task_name_raises(tmp_path):
    records = make_records(task="jpgholder")
    with pytest.raises(CocoFvPreflightError) as exc:
        run(tmp_path, records)
    assert "FR3.5d" in str(exc.value)
    # Why it is fatal: the loader's replace is unanchored, so the directory
    # component is corrupted too and the path can never exist.
    assert "jpgholder/x.jpg".replace("jpg", "pth") == "pthholder/x.pth"


def test_f_missing_stimulus_file_raises(tmp_path):
    records = make_records()
    # Write images for everything except the last image.
    present = [r for r in records if r["name"] != records[-1]["name"]]
    with pytest.raises(CocoFvPreflightError) as exc:
        run(tmp_path, records, images_from=present)
    msg = str(exc.value)
    assert "FR3.5f" in msg
    assert "{}/{}".format(records[-1]["task"], records[-1]["name"]) in msg


def test_split_filtering_happens_before_checking(tmp_path):
    clean_test = make_records(split="test")
    ragged_train = make_records(split="train")[:-1]   # deliberately ragged
    records = clean_test + ragged_train

    assert check_fixations(
        write_fixations(tmp_path, records), write_images(tmp_path, records),
        SUBJECTS, "test")["n_images"] == 3
    with pytest.raises(CocoFvPreflightError):
        check_fixations(write_fixations(tmp_path, records),
                        write_images(tmp_path, records), SUBJECTS, "train")


def test_subject_filtering_matches_select_fewshot_subject(tmp_path):
    # Subjects 0-2 are well formed; 3-9 appear on only one image each. The checker
    # must filter to the query set the way select_fewshot_subject() does.
    records = make_records(subjects=(0, 1, 2))
    for subject in range(3, 10):
        records.append(make_record(name=records[0]["name"], subject=subject))
    assert run(tmp_path, records, fewshot=[0, 1, 2])["n_images"] == 3


# --- group 3: soft counters -------------------------------------------------

def test_e_out_of_range_coordinates_are_counted_not_raised(tmp_path):
    records = make_records()
    records[0]["X"][0] = 512.0
    records[1]["Y"][0] = -1.0
    counters = run(tmp_path, records)
    assert counters["oob"] == 2


def test_e_bounds_are_half_open(tmp_path):
    records = make_records()
    records[0]["X"][0] = 511.9
    records[0]["Y"][0] = 319.9
    assert run(tmp_path, records)["oob"] == 0

    records[0]["X"][0] = 512.0
    assert run(tmp_path, records)["oob"] == 1


def test_n_short_counts_scanpaths_below_three(tmp_path):
    # One subject, four images of lengths 1, 2, 3, 4 -- length == 3 is not short.
    records = [make_record(name="{:012d}.jpg".format(k + 1), subject=0, length=length)
               for k, length in enumerate([1, 2, 3, 4])]
    counters = run(tmp_path, records, fewshot=[0])
    assert counters["n_short"] == 2
    assert counters["gt_length_min"] == 1
    assert counters["gt_length_max"] == 4


def test_counters_are_json_serialisable(tmp_path):
    counters = run(tmp_path, make_records())
    assert json.loads(json.dumps(counters)) == counters
