"""Validation group 3 -- check_features (FR4.6) and package hygiene."""

import ast
import json
import os

import pytest
import torch

from osie_prep import OsiePreflightError
from osie_prep.check_features import check_features, feature_rel_path, main
from osie_fixtures import make_records, write_fixations

SUBJECTS = [10, 11, 12, 13, 14]


def write_features(tmp_path, records, shape=(768, 2048), skip=(), root_name="feat"):
    root = tmp_path / root_name
    for name in sorted(set(r["name"] for r in records)):
        rel = feature_rel_path(name)
        if rel in skip:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(shape), str(path))
    return str(root)


def test_path_matches_the_loader_character_for_character():
    name = "1001.jpg"
    feat_dir = os.path.join("src", "data", "image_features")
    # OSIE_evaluation does join(feature_dir, name.replace('jpg','pth')) on the BARE
    # name -- no "{task}/{name}" prefix. Computed the same way, not re-derived.
    expected = os.path.join(feat_dir, name.replace("jpg", "pth"))
    assert os.path.join(feat_dir, feature_rel_path(name)) == expected
    assert feature_rel_path(name) == "1001.pth"


def test_path_has_no_category_component():
    # The COCO-FreeView version of this tool emitted "potted_plant/000...pth".
    # OSIE's stimulus and feature directories are flat; a stray separator here
    # would send every lookup into a directory that does not exist.
    assert "/" not in feature_rel_path("1001.jpg")
    assert os.sep not in feature_rel_path("1001.jpg")


def test_passes_on_a_complete_cache(tmp_path):
    records = make_records()
    summary = check_features(write_fixations(tmp_path, records),
                             write_features(tmp_path, records))
    assert summary == {"n_checked": 3, "missing": [], "bad_shape": []}


def test_deduplicates_across_subjects(tmp_path):
    # 5 subjects x 1 image must check ONE feature file, not five.
    records = make_records(n_images=1)
    assert len(records) == 5
    summary = check_features(write_fixations(tmp_path, records),
                             write_features(tmp_path, records))
    assert summary["n_checked"] == 1


def test_raises_on_a_missing_tensor(tmp_path):
    records = make_records()
    missing = feature_rel_path(records[0]["name"])
    with pytest.raises(OsiePreflightError) as exc:
        check_features(write_fixations(tmp_path, records),
                       write_features(tmp_path, records, skip=(missing,)))
    msg = str(exc.value)
    assert "FR4.6" in msg
    assert missing in msg
    assert "1 total" in msg


@pytest.mark.parametrize("shape", [(768, 1024), (2048, 768)])
def test_raises_on_a_misshapen_tensor(tmp_path, shape):
    records = make_records()
    with pytest.raises(OsiePreflightError) as exc:
        check_features(write_fixations(tmp_path, records),
                       write_features(tmp_path, records, shape=shape))
    msg = str(exc.value)
    assert str(shape) in msg
    assert "(768, 2048)" in msg


def test_raises_when_the_split_is_empty(tmp_path):
    records = make_records(split="train")
    with pytest.raises(OsiePreflightError) as exc:
        check_features(write_fixations(tmp_path, records),
                       write_features(tmp_path, records), split="test")
    assert "FR4.5" in str(exc.value)


def test_cli_exit_codes_drive_the_run_script_guard(tmp_path, capsys):
    # bash/test_osie.sh uses `if ! check_features.py` to decide whether to spend an
    # hour re-extracting the feature cache, so the exit code is load bearing (FR8.6).
    records = make_records()
    fix = write_fixations(tmp_path, records)

    ok_dir = write_features(tmp_path, records, root_name="feat_ok")
    assert main(["--fix", fix, "--feat-dir", ok_dir]) == 0
    assert json.loads(capsys.readouterr().out)["n_checked"] == 3

    bad_dir = str(tmp_path / "feat_absent")
    assert main(["--fix", fix, "--feat-dir", bad_dir]) == 1


# --- package hygiene (TechStack section 6.9, applied to this package) --------

PURE_MODULES = ["__init__.py", "check_fixations.py"]


def _parse(repo_root, name):
    path = os.path.join(repo_root, "tools", "osie_prep", name)
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
    names = []
    for node in ast.walk(_parse(repo_root, name)):
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
    # No second derivation of the loader's path string anywhere in the package.
    counts = {name: _jpg_to_pth_calls(repo_root, name)
              for name in PURE_MODULES + ["check_features.py"]}
    assert counts == {"__init__.py": 0, "check_fixations.py": 0,
                      "check_features.py": 1}


def test_normalize_fixations_is_gone(repo_root):
    # OSIE's shipped labels are already canonical; a leftover COCO normaliser would
    # silently rewrite them into the wrong schema (Roadmap F1, 2026-09-09).
    assert not os.path.exists(
        os.path.join(repo_root, "tools", "osie_prep", "normalize_fixations.py"))
