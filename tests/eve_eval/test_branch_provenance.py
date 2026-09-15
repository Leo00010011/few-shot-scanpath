"""Group 1 -- branch provenance and D1.

The EVE branch is a verbatim copy of the OSIE one with two files changed. Copying is
exactly what produced the COCO_FV drift TechStack section 4.2 documents -- six real
divergences in one frozen file, found long after the fact. The defence is not
discipline, it is this file: a drift is a red test rather than a quiet six-month-old
divergence.
"""

import ast
import hashlib
import os

import pytest

from conftest import EVE_SRC, OSIE_SRC, TEST_PY

FROZEN = ("utils/evaluation.py",
          "utils/evaltools/scanmatch.py",
          "utils/evaltools/visual_attention_metrics.py")

ALSO_VERBATIM = ("models/gazeformer.py", "models/models.py", "models/sampling.py",
                 "models/loss.py", "models/positional_encodings.py",
                 "utils/logger.py", "utils/data_postprocess.py",
                 "preprocess/feature_extractor.py",
                 "preprocess/preprocess_fixations.py",
                 "preprocess/pseudo_fixations.py")

MAY_DIFFER = {"dataset/dataset.py", "test.py"}


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _tree(root):
    """Relative paths of every source file under ``root``.

    ``__pycache__`` is excluded (convention 6). So are ``src/data/`` and ``*.pth`` /
    ``*.pt``: ``.gitignore``'s ``data/`` rule matches at ANY depth, so the EVE branch's
    copies of ``embeddings.npy`` / ``fixations.json`` are untracked and a fresh
    checkout has only OSIE's, and a checkpoint symlink appears under ``assets/`` only
    on the cluster. Neither is part of the code the copy is about.
    """
    out = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            full = os.path.join(base, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if rel.startswith("data/") or rel.endswith((".pth", ".pt")):
                continue
            out[rel] = full
    return out


@pytest.mark.parametrize("rel", FROZEN)
def test_frozen_files_byte_identical(rel):
    """FR2.2, D1 -- the three files that must never drift."""
    eve, osie = os.path.join(EVE_SRC, rel), os.path.join(OSIE_SRC, rel)
    assert os.path.isfile(eve), "the EVE branch is missing {}".format(rel)
    a, b = _sha(eve), _sha(osie)
    assert a == b, (
        "{} has DRIFTED from the OSIE original (D1): EVE {} vs OSIE {}. The metric "
        "code is frozen -- change the call site, never the function.".format(rel, a, b))


@pytest.mark.parametrize("rel", ALSO_VERBATIM)
def test_supporting_files_byte_identical(rel):
    """FR2.3 -- models/, logger, data_postprocess and preprocess/ are copies too."""
    assert _sha(os.path.join(EVE_SRC, rel)) == _sha(os.path.join(OSIE_SRC, rel)), rel


def test_only_two_files_differ():
    """FR2.3 -- and the differing set is EXACTLY those two, not merely a superset."""
    eve, osie = _tree(EVE_SRC), _tree(OSIE_SRC)
    differing = {rel for rel in set(eve) & set(osie)
                 if _sha(eve[rel]) != _sha(osie[rel])}
    assert differing == MAY_DIFFER, (
        "files differing between the two src/ trees: {}; expected exactly {} "
        "(FR2.3)".format(sorted(differing), sorted(MAY_DIFFER)))


def test_no_path_present_in_one_tree_only():
    """FR2.1 -- the copy is full; nothing was added and nothing was left behind."""
    eve, osie = _tree(EVE_SRC), _tree(OSIE_SRC)
    only_eve = sorted(set(eve) - set(osie))
    only_osie = sorted(set(osie) - set(eve))
    assert not only_eve, "present only in the EVE tree: {}".format(only_eve)
    assert not only_osie, "present only in the OSIE tree: {}".format(only_osie)


def _module(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def test_test_py_imports_only_eve_evaluation():
    """FR2.4 -- ``test.py`` binds EVE_evaluation and never OSIE_evaluation."""
    imported = []
    for node in ast.walk(_module(TEST_PY)):
        if isinstance(node, ast.ImportFrom) and node.module == "dataset.dataset":
            imported.extend(a.name for a in node.names)
    assert imported == ["EVE_evaluation"], imported
    # A name check, not a text check: test.py's comments mention OSIE repeatedly on
    # purpose (the origin_size default, the task embedding's provenance, OPEN-7), and
    # explaining what the EVE branch does differently is the point of those comments.
    names = {n.id for n in ast.walk(_module(TEST_PY)) if isinstance(n, ast.Name)}
    assert "OSIE_evaluation" not in names


def _eve_evaluation_getitem():
    tree = _module(os.path.join(EVE_SRC, "dataset", "dataset.py"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "EVE_evaluation")
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")


def test_no_replace_call_in_getitem():
    """FR2.5 -- the upstream ``img_name.replace('jpg', 'pth')`` idiom is unanchored.

    A key containing 'jpg' anywhere would be silently corrupted by it, which is why
    F5 builds the path by concatenation through ``exp_key_filename()``.
    """
    offenders = [node.lineno for node in ast.walk(_eve_evaluation_getitem())
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "replace"]
    assert not offenders, (
        "EVE_evaluation.__getitem__ calls .replace() at line(s) {} -- FR2.5 forbids "
        "it; build the feature path by concatenation".format(offenders))


def test_getitem_opens_no_image():
    """FR4.2 -- the bridge's stimuli/ export is unused downstream (OPEN-6)."""
    calls = []
    for node in ast.walk(_eve_evaluation_getitem()):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("open", "imread"):
                owner = getattr(func.value, "id", None)
                if owner in ("Image", "io", "plt"):
                    calls.append("{}.{}".format(owner, func.attr))
    assert not calls, calls
