"""Group 4 -- D4: the cohort, the remap, and the subject ids in ``prediction.json``.

Two unrecoverable failures live here, and both produce numbers that look entirely
plausible: a permuted ``--fewshot_subject`` (every participant scored against another
participant's embedding) and a positional subject id in ``prediction.json`` (an
artefact that answers "which of my real subjects is row 3?" with a confident lie).
"""

import ast
import json

import pytest

from conftest import EVE_IDS, N_POOL, TEST_PY


def test_descending_fewshot_raises(make_dataset):
    """FR4.4 -- THE trap's test. Without the assertion the run completes clean."""
    with pytest.raises(ValueError) as exc:
        make_dataset(fewshot_subject=tuple(reversed(range(N_POOL))))
    message = str(exc.value)
    assert "ascending" in message
    assert "D4" in message
    # It names the first offender, not merely "something changed".
    assert "->" in message


def test_rotated_fewshot_raises(make_dataset):
    """Not only reversal: any order but the identity permutes the cohort."""
    with pytest.raises(ValueError) as exc:
        make_dataset(fewshot_subject=(1, 2, 3, 4, 0))
    assert "ascending" in str(exc.value)


def test_ascending_leaves_every_subject_untouched(make_dataset, fixations):
    ds = make_dataset(fewshot_subject=tuple(range(N_POOL)))
    scored = [r for r in fixations if r["split"] == "test"]
    assert [int(r["subject"]) for r in ds.fixations] == [
        int(r["subject"]) for r in scored]
    # The D4 scaffolding key is stripped again -- the records stay the section 3.1
    # schema exactly.
    assert all("_d4_subject_before" not in r for r in ds.fixations)


def test_short_fewshot_list_is_the_preflights_failure_not_the_constructors(
        make_dataset, artefacts):
    """Which layer fires matters, so the two are asserted separately.

    A ``--fewshot_subject`` that does not cover the cohort DROPS records, which makes
    scored images ragged. That is a coverage failure, and FR3.8's uniform-count check
    owns it; the constructor's D4 assertion is about identity and legitimately passes.
    """
    from eve_eval.check_eval import check_eval

    ds = make_dataset(fewshot_subject=(0, 1, 2))       # must NOT raise
    counts = {}
    for rec in ds.fixations:
        counts[rec["name"]] = counts.get(rec["name"], 0) + 1
    assert any(v != 3 for v in counts.values()), "the fixture no longer goes ragged"

    report = check_eval(
        artefacts["bridge_dir"], artefacts["senet_dir"], artefacts["feature_dir"],
        artefacts["weights_dir"], subject_num=3, num_fewshot=2, fast=True,
        expected_cells=18, expected_images=6, expected_subjects=N_POOL)
    assert report["ok"], report["failures"]      # the FILE is fine; the argument was


def _prediction_records(ds, to_eve, subject_num=3):
    """Exactly what test.py writes, over the fixture instead of over a GPU run."""
    out = []
    for idx in range(len(ds)):
        item = ds[idx]
        batch_subjects = [int(s) for s in item["subject"]]     # positional, per image
        for subject_idx in range(subject_num):
            dense = batch_subjects[subject_idx]
            out.append({"name": item["img_name"], "subject": dense,
                        "subject_eve": to_eve[str(dense)]})
    return out


def test_prediction_subject_ids_match_the_ground_truth(make_dataset, subject_id_map,
                                                       fixations):
    """FR7.2 / FR7.4 -- and FR7.1's defect, asserted wrong alongside."""
    ds = make_dataset()
    to_eve = subject_id_map["to_eve"]
    records = _prediction_records(ds, to_eve)

    gt = {(r["name"], int(r["subject"])) for r in fixations if r["split"] == "test"}
    assert {(r["name"], r["subject"]) for r in records} == gt
    assert len(records) == len(gt) == 18
    for r in records:
        assert r["subject_eve"] == EVE_IDS[r["subject"]]

    # FR7.1 -- the UPSTREAM formula, computed over the same fixture and asserted
    # WRONG. get_prediction_list() writes args.fewshot_subject[subject_idx], which is
    # correct for OSIE only by accident (5 entries, subject_num 5). Here it stamps
    # 0/1/2 onto every image while the actual trio varies.
    fewshot_subject = list(range(N_POOL))
    upstream = [{"name": r["name"], "subject": fewshot_subject[i % 3]}
                for i, r in enumerate(records)]
    assert {(r["name"], r["subject"]) for r in upstream} != gt
    assert {r["subject"] for r in upstream} == {0, 1, 2}


def test_dense_id_absent_from_to_eve_raises(make_dataset, subject_id_map):
    """FR7.5 -- subject_id_map.json's to_eve is the sole dense -> EVE authority."""
    ds = make_dataset()
    holed = {k: v for k, v in subject_id_map["to_eve"].items() if k != "4"}
    with pytest.raises(KeyError) as exc:
        _prediction_records(ds, holed)
    assert "4" in str(exc.value)


def test_recover_subject_ids_is_never_called(make_dataset):
    """FR7.5 -- it is the --ex_subject path; F5 uses --fewshot_subject."""
    with open(TEST_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    called = [n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "recover_subject_ids" not in called
    assert "get_prediction_list" not in called


def test_prediction_subject_comes_from_the_batch(make_dataset):
    """The fix is structural, so assert it structurally: test.py must read the id
    out of the flattened batch, never out of ``args.fewshot_subject`` (FR7.2)."""
    with open(TEST_PY, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    subscripts = [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Subscript)]
    assert not any("fewshot_subject" in s and "subject_idx" in s for s in subscripts)
    assert "flat_subjects" in source
