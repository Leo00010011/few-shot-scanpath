"""Fixtures for the F5 eval-branch tests.

Self-contained and CPU-only, mirroring ``tests/eve_prep/``: a six-image / three-subject
synthetic ``fixations.json`` whose trios are drawn from a five-subject pool (so subject
identities genuinely vary by image, the shape OPEN-5 resolved to), a matching
``gt_heatmaps.h5`` built through ``GtHeatmapStore``, one distinct ``(768, 2048)`` tensor
per trial, a ``(5, 384)`` subject embedding, and the three handshake reports.

Markers:

* ``bridge`` -- reads the real ``data/eve_bridge/`` artefacts; skipped when absent
  (that directory is git-ignored, so a fresh checkout has none).

``ISP/EVE/GazeformerISP/src`` is put on ``sys.path`` so ``dataset.dataset`` imports as
the branch's own module, exactly as ``src/test.py`` sees it. ``src/test.py`` itself is
NEVER imported: it parses argv at module scope and pulls ``utils/evaluation.py``, which
imports ``multimatch_gaze`` -- a cluster-only dependency. Group 7 reads it by AST and
rebuilds its parser instead, which is also what makes those checks independent of
whether torch is installed.
"""

import ast
import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(REPO_ROOT, "tools")
EVE_SRC = os.path.join(REPO_ROOT, "ISP", "EVE", "GazeformerISP", "src")
OSIE_SRC = os.path.join(REPO_ROOT, "ISP", "OSIE", "GazeformerISP", "src")
BRIDGE_DIR = os.path.join(REPO_ROOT, "data", "eve_bridge")
OSIE_EMBEDDINGS = os.path.join(OSIE_SRC, "data", "embeddings.npy")
TEST_PY = os.path.join(EVE_SRC, "test.py")

for path in (TOOLS, EVE_SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

FEATURE_SHAPE = (768, 2048)
SUBJECT_FEATURE_DIM = 384

#: Six scored stimuli, three subjects each, drawn from a five-subject pool -- the
#: per-image-varying trio that FR13.2 says a reader will otherwise assume is fixed.
TRIOS = {
    "img0.jpg": (0, 1, 2),
    "img1.jpg": (1, 2, 3),
    "img2.jpg": (2, 3, 4),
    "img3.jpg": (0, 2, 4),
    "img4.jpg": (0, 1, 4),
    "img5.jpg": (1, 3, 4),
}
N_POOL = 5
EVE_IDS = {i: "train{:02d}".format(i) for i in range(N_POOL)}


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "bridge: needs the real data/eve_bridge/ artefacts")


def pytest_collection_modifyitems(config, items):
    no_bridge = not os.path.isfile(os.path.join(BRIDGE_DIR, "fixations.json"))
    skip_bridge = pytest.mark.skip(reason="needs data/eve_bridge/ (git-ignored)")
    for item in items:
        if no_bridge and "bridge" in item.keywords:
            item.add_marker(skip_bridge)


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make_record(name, subject, split="test", n=4):
    """One record in the TechStack section 3.1 schema, in EVE's 1920x1080 space.

    Coordinates are 1-indexed originals; the loader divides by (3.75, 2.8125) to reach
    the 512x384 metric screen, and the last record of each trial reaches x = 1920
    exactly so the top-of-screen mapping is checkable.
    """
    X = [120.0 + 37.0 * i + 11.0 * subject for i in range(n - 1)] + [1920.0]
    Y = [80.0 + 29.0 * i + 7.0 * subject for i in range(n - 1)] + [1080.0]
    T = [150 + 11 * i + 3 * subject for i in range(n)]
    return {"name": name, "subject": int(subject), "X": X, "Y": Y, "T": T,
            "length": n, "split": split, "condition": "freeview", "task": "none"}


def build_fixations():
    """Six scored images x three subjects, plus one support record per subject."""
    records = []
    for name, trio in TRIOS.items():
        for i, subject in enumerate(trio):
            records.append(make_record(name, subject, "test", n=3 + i))
    for subject in range(N_POOL):
        records.append(make_record("support{}.jpg".format(subject), subject,
                                   "train", n=4))
    return records


def exp_key_for(name, subject):
    """``exp_key`` is ``[A-Za-z0-9_]+`` and must never contain 'jpg' (FR5.1)."""
    return "{}_s{}".format(name[:-4], subject)


class Args(object):
    """The handful of ``args`` attributes ``EVE_evaluation`` actually reads."""

    def __init__(self, fewshot_subject, subject_num=3, num_fewshot=2,
                 random_support=0, ex_subject=(-1,)):
        self.fewshot_subject = list(fewshot_subject)
        self.ex_subject = list(ex_subject)
        self.subject_num = subject_num
        self.num_fewshot = num_fewshot
        self.random_support = random_support


@pytest.fixture(scope="session")
def fixations():
    return build_fixations()


@pytest.fixture(scope="session")
def subject_id_map():
    return {"to_dense": {v: k for k, v in EVE_IDS.items()},
            "to_eve": {str(k): v for k, v in EVE_IDS.items()}}


@pytest.fixture(scope="session")
def artefacts(tmp_path_factory, fixations, subject_id_map):
    """A complete synthetic artefact set, built once for the whole session.

    18 scored + 5 support tensors at the real ``(768, 2048)`` float32 shape is ~145 MB
    of tmp, which is why this is session-scoped; tests that need a *broken* artefact
    copy the small files rather than rebuilding the cache.
    """
    import torch

    from eve_bridge.store import GtHeatmapStore

    root = tmp_path_factory.mktemp("eve_eval_artefacts")
    bridge = root / "eve_bridge"
    senet = root / "eve_senet" / "seed0"
    features = root / "eve_features"
    feat_dir = features / "image_features"
    weights = root / "weights"
    for d in (bridge, senet, feat_dir, weights / "checkpoints"):
        d.mkdir(parents=True, exist_ok=True)

    fix_path = bridge / "fixations.json"
    with open(fix_path, "w") as fh:
        json.dump(fixations, fh)
    fix_sha = sha256_file(str(fix_path))

    exp_keys = [exp_key_for(r["name"], r["subject"]) for r in fixations]
    store = GtHeatmapStore.build(
        fixations, exp_keys, subject_id_map,
        {"fixations_sha256": fix_sha, "bundle_dir": "<synthetic>"},
        origin_size=(1080, 1920), action_map=(24, 32), max_length=16, blur_sigma=1)
    hm_path = bridge / "gt_heatmaps.h5"
    store.save(str(hm_path))

    with open(bridge / "subject_id_map.json", "w") as fh:
        json.dump(subject_id_map, fh)

    with open(bridge / "bridge_report.json", "w") as fh:
        json.dump({"fixations_sha256": fix_sha, "origin_size": [1080, 1920],
                   "action_map": [24, 32], "max_length": 16,
                   "min_support_per_subject": 10, "support_pool_size": 20,
                   "subjects_per_image": 3, "num_subjects": N_POOL,
                   "counters": {"short_scanpath": 0, "clamped_coords": 5,
                                "incomplete_stimulus": 602, "surplus_trial": 73}}, fh)

    # One DISTINCT tensor per trial: a constant equal to a hash of the exp_key, so
    # "two subjects on one image received different tensors" is checkable by value.
    feature_sha = {}
    for i, (rec, key) in enumerate(zip(fixations, exp_keys)):
        tensor = torch.full(FEATURE_SHAPE, float(i + 1), dtype=torch.float32)
        path = feat_dir / (key + ".pth")
        torch.save(tensor, str(path))
        feature_sha[key] = sha256_file(str(path))

    import shutil
    shutil.copyfile(OSIE_EMBEDDINGS, str(features / "embeddings.npy"))
    with open(features / "feature_report.json", "w") as fh:
        json.dump({"fixations_sha256": fix_sha, "keying": "exp_key",
                   "feature_shape": list(FEATURE_SHAPE),
                   "feature_sha256": feature_sha}, fh)

    emb = torch.zeros((N_POOL, SUBJECT_FEATURE_DIM), dtype=torch.float32)
    for i in range(N_POOL):
        emb[i] = float(i + 1)
    emb_path = senet / "eve_fewshot_user_embedding_10_seed0.pt"
    torch.save(emb, str(emb_path))
    with open(senet / "senet_report.json", "w") as fh:
        json.dump({"fixations_sha256": fix_sha,
                   "embedding_sha256": sha256_file(str(emb_path)),
                   "num_subjects": N_POOL, "embedding_dim": SUBJECT_FEATURE_DIM,
                   "args": {"seed": 0}}, fh)

    with open(weights / "checkpoints" / "checkpoint_best.pth", "wb") as fh:
        fh.write(b"not a real checkpoint -- the preflight only hashes it")

    return {
        "root": str(root), "bridge_dir": str(bridge), "senet_dir": str(senet),
        "feature_dir": str(features), "feat_dir": str(feat_dir),
        "weights_dir": str(weights), "fixations": str(fix_path),
        "heatmaps": str(hm_path), "subject_map": str(bridge / "subject_id_map.json"),
        "bridge_report": str(bridge / "bridge_report.json"),
        "senet_report": str(senet / "senet_report.json"),
        "feature_report": str(features / "feature_report.json"),
        "embeddings_npy": str(features / "embeddings.npy"),
        "user_embedding": str(emb_path),
        "checkpoint": str(weights / "checkpoints" / "checkpoint_best.pth"),
        "exp_keys": dict(zip([(r["name"], r["subject"]) for r in fixations],
                             exp_keys)),
        "feature_sha256": feature_sha,
    }


@pytest.fixture
def exp_key_map(artefacts):
    from eve_prep.trial_keys import load_trial_exp_keys

    return load_trial_exp_keys(artefacts["heatmaps"],
                               fixations_path=artefacts["fixations"])


@pytest.fixture
def make_dataset(artefacts, exp_key_map):
    """Construct ``EVE_evaluation`` over the synthetic artefacts.

    ``fewshot_subject`` defaults to the ascending whole pool -- the only order that
    yields the identity remap (FR4.4).
    """
    from dataset.dataset import EVE_evaluation

    def _make(fewshot_subject=tuple(range(N_POOL)), subject_num=3,
              origin_size=(1080, 1920), resize=(384, 512), stimuli_dir=None,
              feature_dir=None, exp_keys=None, max_length=16):
        args = Args(fewshot_subject, subject_num=subject_num)
        return EVE_evaluation(
            args,
            stimuli_dir if stimuli_dir is not None
            else os.path.join(artefacts["root"], "no-such-stimuli-dir"),
            feature_dir or artefacts["feat_dir"],
            artefacts["fixations"], artefacts["embeddings_npy"],
            heatmaps_dir=artefacts["heatmaps"],
            exp_key_map=dict(exp_key_map) if exp_keys is None else exp_keys,
            action_map=(24, 32), origin_size=origin_size, resize=resize,
            max_length=max_length, type="test", transform=None)

    return _make


# ---------------------------------------------------------------------------
# Group 7 -- src/test.py without importing it
# ---------------------------------------------------------------------------

def _test_py_source():
    with open(TEST_PY, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="session")
def test_py_source():
    return _test_py_source()


@pytest.fixture(scope="session")
def test_py_ast(test_py_source):
    return ast.parse(test_py_source)


def _is_parser_stmt(node):
    """``parser = argparse.ArgumentParser(...)`` or ``parser.add_argument(...)``."""
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "parser"
                   for t in node.targets)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        return (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "parser")
    return False


def _is_repeat_guard(node):
    """The top-level ``if args.eval_repeat_num != 1:`` rejection (FR6.5)."""
    return (isinstance(node, ast.If)
            and "eval_repeat_num" in ast.dump(node.test))


@pytest.fixture(scope="session")
def test_py_functions(test_py_ast):
    """``src/test.py``'s module-level helpers, executed without importing the module.

    ``reporting_notes`` and ``multimatch_nan_drops`` need numpy and nothing else, so
    the real functions can be exercised directly -- the alternative, transcribing them
    into the test, is how a test ends up agreeing with a version of the code that no
    longer exists.
    """
    import collections as _collections

    wanted = {"reporting_notes", "multimatch_nan_drops"}
    nodes = [n for n in test_py_ast.body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted, [n.name for n in nodes]
    ns = {"np": np, "collections": _collections, "json": json, "os": os, "sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), TEST_PY, "exec"), ns)
    return ns


@pytest.fixture(scope="session")
def arg_parser_factory(test_py_ast):
    """Rebuild ``src/test.py``'s parser, and its FR6.5 guard, without importing it.

    Only the ``parser`` statements and the ``eval_repeat_num`` guard are executed, so
    nothing here touches torch, tensorboard or ``utils/evaluation.py`` (whose
    ``multimatch_gaze`` import is cluster-only). The parser under test is therefore the
    real one, byte-for-byte, rather than a transcription that could drift.
    """
    import argparse as _argparse

    parser_nodes = [n for n in test_py_ast.body if _is_parser_stmt(n)]
    guard_nodes = [n for n in test_py_ast.body if _is_repeat_guard(n)]
    assert parser_nodes, "no parser statements found in src/test.py"
    assert guard_nodes, "no --eval_repeat_num guard found in src/test.py (FR6.5)"

    def _factory(argv=None, apply_guard=True):
        ns = {"argparse": _argparse}
        exec(compile(ast.Module(body=parser_nodes, type_ignores=[]),
                     TEST_PY, "exec"), ns)
        parser = ns["parser"]
        if argv is None:
            return parser
        args = parser.parse_args(argv)
        if apply_guard:
            ns["args"] = args
            exec(compile(ast.Module(body=guard_nodes, type_ignores=[]),
                         TEST_PY, "exec"), ns)
        return args

    return _factory
