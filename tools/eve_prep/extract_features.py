"""Stage B for EVE -- one feature tensor per **trial** (FR3, FR4, FR5, FR6, FR8).

The only torch importer in this package. Reads each trial's screen capture through
``EveBundle.get_stimulus(exp_key)`` -- the declared D2 deviation (FR12) -- and writes
``image_features/<exp_key>.pth``, a ``(768, 2048)`` float32 CPU tensor.

Two things here are load bearing and easy to get wrong:

* The backbone class is **imported** from upstream, unmodified (FR4.1, convention 2).
  Only the three-line transform chain is transcribed, because ``image_data()`` globs
  ``*.jpg`` from a hardcoded ``<dataset_path>/train/`` and F4 has uint8 arrays in
  memory -- staging 1804 PNGs as JPEGs would add a lossy re-encode to the one thing
  FR11.2 says must be preserved exactly. The transcription is *provable*, not assumed:
  ``tests/eve_prep/test_preprocess.py`` runs both paths on one file and asserts
  ``torch.equal`` (FR4.4). If that test fails, nothing in F4 is trustworthy.
* The 1920x1080 -> 1024x768 resize is a **non-uniform squash** (16:9 into 4:3),
  x = 0.5333, y = 0.7111 -- a *third* distortion alongside F5's 512x384 metric screen
  (OPEN-4) and F3's SE-Net 512x320 input. It is recorded in ``feature_report.json``
  and must reach F7 (FR4.6, FR11.5).

CLI: ``py tools/eve_prep/extract_features.py --bundle-dir DIR --bridge-dir DIR
--out-dir DIR [--split both|test|train] [--overwrite] [--cuda 0]``
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import sys
import types

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

try:
    from . import EvePrepError, FEATURE_SHAPE, RESIZE_INPUT, STIMULUS_SHAPE
    from .check_features import select_trials
    from .trial_keys import (crosscheck_exp_keys, derive_trial_exp_keys,
                             exp_key_filename, load_json, load_trial_exp_keys,
                             sha256_file)
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_prep import EvePrepError, FEATURE_SHAPE, RESIZE_INPUT, STIMULUS_SHAPE
    from eve_prep.check_features import select_trials
    from eve_prep.trial_keys import (crosscheck_exp_keys, derive_trial_exp_keys,
                                     exp_key_filename, load_json,
                                     load_trial_exp_keys, sha256_file)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISP_SRC = os.path.join(REPO_ROOT, "ISP", "OSIE", "GazeformerISP", "src")
OSIE_EMBEDDINGS = os.path.join(ISP_SRC, "data", "embeddings.npy")

TASK_KEY = "free-viewing"
TASK_EMB_SHAPE = (768,)

COUNTER_NAMES = ("missing_stimulus", "bad_stimulus_shape", "bad_feature_shape",
                 "exp_key_mismatch", "skipped_existing")

# Upstream's own two transforms, constructed once (feature_extractor.py:42-43).
_resize = T.Resize(RESIZE_INPUT)
_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ----------------------------------------------------------------------
# 3a -- the upstream import, with the sentence_transformers stub
# ----------------------------------------------------------------------

def _stub_sentence_transformers():
    """FR4.2 -- the call-site workaround ``bash/test_osie.sh`` established.

    ``feature_extractor.py`` imports ``SentenceTransformer`` at module scope but uses
    it only inside ``text_data()``, which F4 never calls (``embeddings.npy`` is copied,
    FR6). Registering a stub avoids pulling transformers + tokenizers -- and pip's
    opinion about torch -- into a shared env. The upstream file is untouched.
    """
    if "sentence_transformers" in sys.modules:
        return
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sentence_transformers")

        class _Unavailable(object):
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "text_data() is not part of F4; embeddings.npy is copied from "
                    "the OSIE branch (FR6.1)")

        stub.SentenceTransformer = _Unavailable
        sys.modules["sentence_transformers"] = stub


def import_backbone_class(isp_src=ISP_SRC):
    """FR4.1 -- ``preprocess.feature_extractor.ResNetCOCO``, unmodified."""
    _stub_sentence_transformers()
    if isp_src not in sys.path:
        sys.path.insert(0, isp_src)
    try:
        from preprocess.feature_extractor import ResNetCOCO
    except ImportError as exc:
        raise EvePrepError(
            "cannot import preprocess.feature_extractor from {}; PYTHONPATH must "
            "include the OSIE branch's src/ directory (FR4.1): {!r}".format(
                isp_src, exc))
    return ResNetCOCO


def build_backbone(device, isp_src=ISP_SRC):
    """FR4.1, FR4.8 -- constructed once, ``.eval()``, on ``device``."""
    resnet_coco = import_backbone_class(isp_src)
    return resnet_coco(device=device).to(device).eval()


# ----------------------------------------------------------------------
# 3b -- preprocessing, the transcription under test (FR4.3)
# ----------------------------------------------------------------------

def preprocess(arr):
    """``(1080, 1920, 3)`` uint8 -> ``(1, 3, 768, 1024)`` float32 (FR4.3).

    A literal transcription of ``feature_extractor.py:50-51``, differing only in that
    the ``PIL.Image`` comes from an array rather than from ``PIL.Image.open``.
    The order ``to_tensor -> resize -> normalize`` is load bearing: resizing a *float
    tensor* rather than a PIL image is what upstream does and it selects a different
    interpolation path. Do not "improve" it.
    """
    if tuple(arr.shape) != STIMULUS_SHAPE or arr.dtype != np.uint8:
        raise EvePrepError(
            "stimulus is {} {}, expected {} uint8 (FR3.2)".format(
                tuple(arr.shape), arr.dtype, STIMULUS_SHAPE))
    pil_image = PIL.Image.fromarray(arr)
    return _normalize(_resize(T.functional.to_tensor(pil_image))).unsqueeze(0)


# ----------------------------------------------------------------------
# 3c -- one trial
# ----------------------------------------------------------------------

@torch.no_grad()
def extract_one(backbone, arr):
    """``(1080, 1920, 3)`` uint8 -> ``(768, 2048)`` float32 on CPU (FR4.5, FR4.8)."""
    out = backbone(preprocess(arr)).squeeze().detach().cpu()
    if tuple(out.shape) != FEATURE_SHAPE or out.dtype != torch.float32:
        raise EvePrepError(
            "feature is {} {}, expected {} float32 (FR4.5)".format(
                tuple(out.shape), out.dtype, FEATURE_SHAPE))
    return out


# ----------------------------------------------------------------------
# 3d -- the cache
# ----------------------------------------------------------------------

def _get_stimulus(bundle, exp_key):
    """FR3.1, FR3.3 -- the array as returned, with the exp_key attached on failure."""
    try:
        arr = bundle.get_stimulus(exp_key)
    except Exception as exc:
        raise EvePrepError(
            "get_stimulus({!r}) failed: {!r} -- a silently missing feature is the D7 "
            "failure mode, so this is not skipped (FR3.3)".format(exp_key, exc))
    arr = np.asarray(arr)
    if tuple(arr.shape) != STIMULUS_SHAPE or arr.dtype != np.uint8:
        raise EvePrepError(
            "stimulus for exp_key {!r} is {} {}, expected {} uint8; the whole "
            "coordinate contract depends on origin_size (FR3.2)".format(
                exp_key, tuple(arr.shape), arr.dtype, STIMULUS_SHAPE))
    return arr


def extract_all(bundle, exp_keys, out_dir, device, overwrite=False, backbone=None,
                isp_src=ISP_SRC, progress_every=100):
    """Extract one tensor per exp_key into ``<out_dir>/image_features/`` (FR4.7, FR8.1).

    Iterates in **sorted** order so a partial run resumes deterministically. Writes to
    a ``.tmp`` sibling and ``os.replace``s it: an interrupted run must not leave a
    truncated ``.pth`` that ``check_features`` would later "load".
    """
    feat_dir = os.path.join(out_dir, "image_features")
    os.makedirs(feat_dir, exist_ok=True)

    counters = {name: 0 for name in COUNTER_NAMES}
    feature_sha256 = {}
    n_extracted = 0

    keys = sorted(exp_keys)
    for i, exp_key in enumerate(keys, 1):
        rel = exp_key_filename(exp_key)          # FR5.2 charset + no-jpg rule
        path = os.path.join(feat_dir, rel)

        if os.path.isfile(path) and not overwrite:
            counters["skipped_existing"] += 1
            feature_sha256[exp_key] = sha256_file(path)
            continue

        if backbone is None:
            backbone = build_backbone(device, isp_src)

        tensor = extract_one(backbone, _get_stimulus(bundle, exp_key))
        tmp = path + ".tmp"
        torch.save(tensor, tmp)
        os.replace(tmp, path)
        feature_sha256[exp_key] = sha256_file(path)
        n_extracted += 1

        if progress_every and i % progress_every == 0:
            sys.stderr.write("  {}/{} trials ({} extracted, {} skipped)\n".format(
                i, len(keys), n_extracted, counters["skipped_existing"]))
            sys.stderr.flush()

    return {"n_extracted": n_extracted,
            "n_skipped_existing": counters["skipped_existing"],
            "counters": counters,
            "feature_sha256": feature_sha256}


# ----------------------------------------------------------------------
# 3e -- task embeddings (FR6)
# ----------------------------------------------------------------------

def copy_task_embeddings(src, dst):
    """Copy the OSIE ``embeddings.npy`` and verify the **destination** (FR6).

    Verifying the copy rather than the source is what catches a truncated write. Using
    the byte-identical file F1 scored with is what keeps F1's baseline and F5's run
    comparable on this input (FR6.3).
    """
    if not os.path.isfile(src):
        raise EvePrepError("task embedding source {} does not exist (FR6.1)".format(src))
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    shutil.copyfile(src, dst)

    try:
        loaded = np.load(dst, allow_pickle=True).item()
    except Exception as exc:
        raise EvePrepError(
            "{} does not load as a pickled dict (FR6.2): {!r}".format(dst, exc))
    if not isinstance(loaded, dict):
        raise EvePrepError(
            "{} loads as {}, expected dict (FR6.2)".format(dst, type(loaded).__name__))
    if TASK_KEY not in loaded:
        raise EvePrepError(
            "{} has keys {} and is missing {!r}; the loader hardcodes that task "
            "(FR6.2)".format(dst, sorted(loaded), TASK_KEY))

    value = np.asarray(loaded[TASK_KEY])
    if tuple(value.shape) != TASK_EMB_SHAPE or value.dtype != np.float32:
        raise EvePrepError(
            "{}[{!r}] is {} {}, expected {} float32 (args.lm_hidden_dim) "
            "(FR6.2)".format(dst, TASK_KEY, tuple(value.shape), value.dtype,
                             TASK_EMB_SHAPE))

    src_sha, dst_sha = sha256_file(src), sha256_file(dst)
    if src_sha != dst_sha:
        raise EvePrepError(
            "embeddings.npy copy is not byte-identical (FR6.3): {} vs {}".format(
                src_sha, dst_sha))

    return {"src": os.path.abspath(src), "dst": os.path.abspath(dst),
            "sha256": dst_sha, "key": TASK_KEY,
            "shape": list(TASK_EMB_SHAPE), "dtype": "float32"}


# ----------------------------------------------------------------------
# Step 5 -- version probing, single-sourced
# ----------------------------------------------------------------------

def _version_of(module_name, attr="__version__"):
    try:
        module = __import__(module_name)
        return str(getattr(module, attr, "unknown"))
    except Exception as exc:
        return "MISSING ({!r})".format(exc)


def versions(device=None):
    """FR1.2, FR1.5 -- one resolution, both printed and embedded in the report.

    ``evedataset`` is imported **whole** (FR1.3): it transitively imports
    ``torchvision.transforms.v2``, so a torchvision too old for v2 must fail here at
    preflight rather than mid-run.
    """
    out = {
        "python": platform.python_version(),
        "torch": _version_of("torch"),
        "torchvision": _version_of("torchvision"),
        "numpy": _version_of("numpy"),
        "PIL": _version_of("PIL"),
        "h5py": _version_of("h5py"),
        "pandas": _version_of("pandas"),
        "evedataset": _version_of("evedataset"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device is not None:
        out["device"] = str(device)
    return out


def require_environment(resolved, require_cuda=True):
    """FR1.2, FR1.4 -- a missing module or absent CUDA raises before anything costly."""
    missing = sorted(k for k, v in resolved.items()
                     if isinstance(v, str) and v.startswith("MISSING"))
    if missing:
        raise EvePrepError(
            "required modules are missing from this env (FR1.2): {}".format(
                ", ".join("{}={}".format(k, resolved[k]) for k in missing)))
    if require_cuda and not resolved.get("cuda_available"):
        raise EvePrepError(
            "torch.cuda.is_available() is False; F4 extracts 1804 tensors and a "
            "silently CPU-bound run must not look normal (FR1.4)")


# ----------------------------------------------------------------------
# 3f -- the CLI
# ----------------------------------------------------------------------

def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_bundle(bundle_dir):
    try:
        from evedataset import EveBundle
    except ImportError as exc:
        raise EvePrepError(
            "cannot import evedataset (FR1.2/FR12): {!r}".format(exc))
    return EveBundle.load(bundle_dir)


def _assert_counts(selected, splits, bridge_report, split):
    """Cross-check the selected trial counts against ``bridge_report.json`` (FR3f).

    Derived independently on both sides, so agreement is evidence rather than
    tautology. A disagreement means F2's artefacts and this run's view of them have
    diverged and nothing downstream should be trusted.
    """
    n_test = sum(1 for k in selected if splits[k] == "test")
    n_train = sum(1 for k in selected if splits[k] == "train")
    expected = {
        "test": int(bridge_report["num_trials_test"]),
        "train": int(bridge_report["num_trials_train"]),
        "both": int(bridge_report["num_trials"]),
    }
    got = {"test": n_test, "train": n_train, "both": n_test + n_train}
    if got[split] != expected[split]:
        raise EvePrepError(
            "selected {} trials for split={!r} but bridge_report.json says {} "
            "(FR8); F2's artefacts and this run disagree".format(
                got[split], split, expected[split]))
    return n_test, n_train


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract per-trial image features for the EVE cohort (F4)")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--bridge-dir", default=os.path.join("data", "eve_bridge"))
    parser.add_argument("--out-dir", default=os.path.join("data", "eve_features"))
    parser.add_argument("--split", default="both", choices=["both", "test", "train"])
    parser.add_argument("--osie-embeddings", default=OSIE_EMBEDDINGS)
    parser.add_argument("--isp-src", default=ISP_SRC)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="skip the FR1.4 CUDA requirement; for dev only")
    args = parser.parse_args(argv)

    fixations_path = os.path.join(args.bridge_dir, "fixations.json")
    heatmaps_path = os.path.join(args.bridge_dir, "gt_heatmaps.h5")
    subject_map_path = os.path.join(args.bridge_dir, "subject_id_map.json")
    bridge_report_path = os.path.join(args.bridge_dir, "bridge_report.json")

    device = (torch.device("cuda:{}".format(args.cuda))
              if not args.allow_cpu else torch.device("cpu"))
    resolved = versions(device)
    require_environment(resolved, require_cuda=not args.allow_cpu)
    sys.stderr.write(json.dumps(resolved, indent=2) + "\n")

    fixations = load_json(fixations_path)
    subject_id_map = load_json(subject_map_path)
    bridge_report = load_json(bridge_report_path)

    # FR2.1/FR2.5 -- the authority, hash-gated against the records it describes.
    exp_of = load_trial_exp_keys(heatmaps_path, fixations_path)

    # FR12's containment: every bundle read is re-anchored to a bridge artefact.
    bundle = _open_bundle(args.bundle_dir)
    derived = derive_trial_exp_keys(bundle.samples_df, fixations, subject_id_map)
    crosscheck = crosscheck_exp_keys(exp_of, derived)

    selected, splits = select_trials(fixations, args.split)
    n_test, n_train = _assert_counts(selected, splits, bridge_report, args.split)
    exp_keys = [exp_of[k] for k in selected]

    os.makedirs(args.out_dir, exist_ok=True)
    sys.stderr.write("== F4 Stage B: {} trials ({} test / {} train), split={} ==\n"
                     .format(len(selected), n_test, n_train, args.split))

    result = extract_all(bundle, exp_keys, args.out_dir, device,
                         overwrite=args.overwrite, isp_src=args.isp_src)
    embeddings = copy_task_embeddings(
        args.osie_embeddings, os.path.join(args.out_dir, "embeddings.npy"))

    report = {
        "args": vars(args),
        "versions": resolved,
        "created_utc": _utc_now(),
        "bundle_dir": os.path.abspath(args.bundle_dir),
        "fixations_path": os.path.abspath(fixations_path),
        "heatmaps_path": os.path.abspath(heatmaps_path),
        "fixations_sha256": sha256_file(fixations_path),
        "n_trials": len(selected),
        "n_trials_test": n_test,
        "n_trials_train": n_train,
        "n_extracted": result["n_extracted"],
        "n_skipped_existing": result["n_skipped_existing"],
        "feature_shape": list(FEATURE_SHAPE),
        "resize_input": list(RESIZE_INPUT),
        # FR4.6 -- the third distortion in the stack; F7 states all three.
        "squash": {"x": RESIZE_INPUT[1] / float(STIMULUS_SHAPE[1]),
                   "y": RESIZE_INPUT[0] / float(STIMULUS_SHAPE[0]),
                   "uniform": False},
        # FR11.3 -- the OPEN-6 resolution, recorded as data so it stays checkable.
        "keying": "exp_key",
        "embeddings": embeddings,
        "exp_key_crosscheck": crosscheck,
        "counters": result["counters"],
        "feature_sha256": result["feature_sha256"],
    }
    # FR8.3 -- written even when the cache was already complete and no work was done.
    with open(os.path.join(args.out_dir, "feature_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("feature_sha256", "args", "versions")}, indent=2))
    sys.stderr.write("== F4 done: {} extracted, {} skipped, report in {} ==\n".format(
        result["n_extracted"], result["n_skipped_existing"], args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
