"""Validation group 3 -- check_features (FR4.6) and package hygiene."""

import ast
import json
import os

import pytest
import torch

from cocofv_prep import CocoFvPreflightError
from cocofv_prep.check_features import check_features, feature_rel_path, main
from cocofv_fixtures import make_records, write_fixations

SUBJECTS = [0, 1, 2]


def write_features(tmp_path, records, shape=(768, 2048), skip=(), root_name="feat"):
    root = tmp_path / root_name
    pairs = sorted(set((r["task"], r["name"]) for r in records))
    for task, name in pairs:
        rel = feature_rel_path(task, name)
        if rel in skip:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(shape), str(path))
    return str(root)


def test_path_matches_the_loader_character_for_character():
    task, name = "potted_plant", "000000012345.jpg"
    feat_dir = os.path.join("work", "image_features")
    # The loader does join(feat_dir, img_name.replace('jpg','pth')) with
    # img_name == "{task}/{name}". Computed the same way, not re-derived.
    expected = os.path.join(feat_dir, "{}/{}".format(task, name).replace("jpg", "pth"))
    assert os.path.join(feat_dir, feature_rel_path(task, name)) == expected
    assert feature_rel_path(task, name) == "potted_plant/000000012345.pth"


def test_passes_on_a_complete_cache(tmp_path):
    records = make_records()
    summary = check_features(write_fixations(tmp_path, records),
                             write_features(tmp_path, records))
    assert summary == {"n_checked": 3, "missing": [], "bad_shape": []}


def test_deduplicates_across_subjects(tmp_path):
    # 3 subjects x 1 image must check ONE feature file, not three.
    records = make_records(n_images=1)
    assert len(records) == 3
    summary = check_features(write_fixations(tmp_path, records),
                             write_features(tmp_path, records))
    assert summary["n_checked"] == 1


def test_raises_on_a_missing_tensor(tmp_path):
    records = make_records()
    missing = feature_rel_path(records[0]["task"], records[0]["name"])
    with pytest.raises(CocoFvPreflightError) as exc:
        check_features(write_fixations(tmp_path, records),
                       write_features(tmp_path, records, skip=(missing,)))
    msg = str(exc.value)
    assert "FR4.6" in msg
    assert missing in msg
    assert "1 total" in msg


@pytest.mark.parametrize("shape", [(768, 1024), (2048, 768)])
def test_raises_on_a_misshapen_tensor(tmp_path, shape):
    records = make_records()
    with pytest.raises(CocoFvPreflightError) as exc:
        check_features(write_fixations(tmp_path, records),
                       write_features(tmp_path, records, shape=shape))
    msg = str(exc.value)
    assert str(shape) in msg
    assert "(768, 2048)" in msg


def test_cli_exit_codes_drive_the_run_script_guard(tmp_path, capsys):
    # bash/test_cocofv.sh uses `if ! check_features.py` to decide whether to spend
    # an hour re-extracting tens of GB, so the exit code is load bearing (FR8.6).
    records = make_records()
    fix = write_fixations(tmp_path, records)

    ok_dir = write_features(tmp_path, records, root_name="feat_ok")
    assert main(["--fix", fix, "--feat-dir", ok_dir]) == 0
    assert json.loads(capsys.readouterr().out)["n_checked"] == 3

    bad_dir = str(tmp_path / "feat_absent")
    assert main(["--fix", fix, "--feat-dir", bad_dir]) == 1


# --- package hygiene (TechStack section 6.9, applied to this package) --------

PURE_MODULES = ["__init__.py", "normalize_fixations.py", "check_fixations.py"]


def _parse(repo_root, name):
    path = os.path.join(repo_root, "tools", "cocofv_prep", name)
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def _jpg_to_pth_calls(repo_root, name):
    """Count actual ``<expr>.replace('jpg', 'pth')`` calls in a module.

    Matching source text would also hit the FR3.5(d) error message, which explains
    the trap rather than falling into it. The invariant is about calls.
    """
    hits = 0
    for node in ast.walk(_parse(repo_root, name)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and [getattr(a, "value", None) for a in node.args] == ["jpg", "pth"]):
            hits += 1
    return hits


def _imported_names(repo_root, name):
    path = os.path.join(repo_root, "tools", "cocofv_prep", name)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_login_node_modules_do_not_import_torch(repo_root):
    for name in PURE_MODULES:
        imported = _imported_names(repo_root, name)
        assert not any(m == "torch" or m.startswith("torch.") for m in imported), \
            "{} must run without torch".format(name)


def test_package_never_imports_frozen_metric_code(repo_root):
    for name in PURE_MODULES + ["check_features.py"]:
        for module in _imported_names(repo_root, name):
            assert "evaluation" not in module, (name, module)
            assert "evaltools" not in module, (name, module)
            assert "GazeformerISP" not in module, (name, module)


def test_feature_path_is_single_sourced(repo_root):
    # No third derivation of the loader's path string anywhere in the package.
    counts = {name: _jpg_to_pth_calls(repo_root, name)
              for name in PURE_MODULES + ["check_features.py"]}
    assert counts == {"__init__.py": 0, "normalize_fixations.py": 0,
                      "check_fixations.py": 0, "check_features.py": 1}
