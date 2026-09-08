"""Validation group 1 -- normalize_fixations (FR3.3)."""

import json

from cocofv_prep.normalize_fixations import main, normalize


def test_val_becomes_validation_without_mutating_input():
    original = {"split": "val", "task": "bottle", "name": "a.jpg", "subject": 0}
    out = normalize([original])
    assert out[0]["split"] == "validation"
    assert original["split"] == "val", "normalize() must not mutate its input"


def test_spaced_task_names_are_underscored():
    out = normalize([
        {"split": "test", "task": "potted plant"},
        {"split": "test", "task": "stop sign"},
        {"split": "test", "task": "potted_plant"},
    ])
    assert [r["task"] for r in out] == ["potted_plant", "stop_sign", "potted_plant"]


def test_unknown_task_passes_through_verbatim():
    # Regression guard: _TASK_FIXES.get(task, task), never _TASK_FIXES[task].
    out = normalize([{"split": "test", "task": "bottle"}])
    assert out[0]["task"] == "bottle"


def test_only_val_split_is_rewritten():
    out = normalize([{"split": "train", "task": "bottle"},
                     {"split": "test", "task": "bottle"}])
    assert [r["split"] for r in out] == ["train", "test"]


def test_extra_keys_survive():
    rec = {"split": "test", "task": "bottle", "name": "a.jpg", "subject": 3, "extra": 1}
    out = normalize([rec])[0]
    assert out == rec


def test_normalize_is_idempotent():
    records = [{"split": "val", "task": "potted plant"},
               {"split": "test", "task": "bottle"}]
    once = normalize(records)
    assert normalize(once) == once


def test_cli_reports_zero_counters_on_an_already_normalized_file(tmp_path, capsys):
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    src.write_text(json.dumps([{"split": "test", "task": "bottle"}]), encoding="utf-8")

    assert main(["--in", str(src), "--out", str(dst)]) == 0
    out = capsys.readouterr().out
    assert "already normalized" in out
    assert json.loads(dst.read_text(encoding="utf-8")) == [{"split": "test", "task": "bottle"}]
