"""Derive SE-Net's encoder init pickles from the released checkpoint (F3 preflight).

``SE-Net/src/models.py::ImageFeatureEncoder.__init__`` unconditionally loads **two**
files and strict-loads each into a freshly built module:

* ``data/resnet50.yaml``'s ``MODEL.WEIGHTS`` -> ``data/M2F_R50.pkl`` (the ResNet), and
* the same path with ``_MSDeformAttnPixelDecoder`` appended ->
  ``data/M2F_R50_MSDeformAttnPixelDecoder.pkl`` (the deformable pixel decoder).

Neither ships with this repository nor with the released checkpoint bundle, and both
loads are unguarded (the authors' own ``if os.path.exists(...)`` lines are commented
out), so model construction dies with a bare ``FileNotFoundError`` several frames
inside SE-Net before F3 reaches a single forward pass.

They are *initialisation* files. ``load_model()`` calls
``model.load_state_dict(ckp["model"], strict=False)`` immediately afterwards, and the
released checkpoint carries **every** tensor of both modules — 265 under
``encoder.backbone.*`` and 117 under ``encoder.pixel_decoder.*``. Each value these
pickles supply is overwritten before any image is seen. What they have to be is
*structurally* complete, not any particular set of numbers.

Hence this tool: rather than download third-party Mask2Former pickles whose provenance
we cannot check against the checkpoint we actually run, we lift both subtrees **out of
that checkpoint**. The encoder then provably ends at the released weights by two
independent routes, and there is no new external dependency.

Key naming is left exactly as the checkpoint stores it. Reconciling the ResNet's
stage names against whatever the installed detectron2 calls them is
:func:`embed.align_stage_prefix`'s job, in one place (detectron2 0.6 wants bare
``res*``; the authors' version wanted ``stages.res*``). What this tool *does* check is
that no key would be caught by ``models.py``'s own rename loops — which rewrite
leading ``res`` / ``adapter`` / ``layer`` keys — since such a key would be renamed and
then fail the strict load somewhere confusing. **Those strict loads are the
completeness check**: a missing tensor raises there.

Nothing under ``SE-Net/`` is edited (convention 2); this writes two files under
``data/``, which is git-ignored (convention 5).

CPU-only. Run it once per checkout, before ``embed_eve_subjects.sh``:

    python tools/eve_senet/make_backbone_init.py \\
        --checkpoint weights/OSIE-.../OSIE/ckp_11999.pt
"""

import argparse
import hashlib
import json
import os
import sys

import torch

try:
    from . import EveSenetError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_senet import EveSenetError

# name -> (checkpoint prefix, filename suffix, key prefixes models.py would rename,
#          tensor count in the released OSIE ckp_11999.pt -- reported, not enforced)
COMPONENTS = {
    "backbone": ("encoder.backbone.", "", ("res",), 265),
    "pixel_decoder": ("encoder.pixel_decoder.", "_MSDeformAttnPixelDecoder",
                      ("adapter", "layer"), 117),
}
BACKBONE_PREFIX = COMPONENTS["backbone"][0]
EXPECTED_N_TENSORS = COMPONENTS["backbone"][3]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(checkpoint_path):
    """The checkpoint's state dict, wrapped (``{"model": ...}``) or bare."""
    if not os.path.isfile(checkpoint_path):
        raise EveSenetError(
            "make_backbone_init: no such checkpoint: {}".format(checkpoint_path))
    ckp = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckp, dict) and "model" in ckp:
        return ckp["model"]
    if isinstance(ckp, dict):
        return ckp
    raise EveSenetError(
        "make_backbone_init: {} holds {}, not a state dict".format(
            checkpoint_path, type(ckp).__name__))


def extract_component(checkpoint_path, component="backbone", prefix=None):
    """``{stripped_key: tensor}`` for one encoder subtree of the checkpoint."""
    if component not in COMPONENTS:
        raise EveSenetError(
            "make_backbone_init: unknown component {!r}, expected one of {}".format(
                component, sorted(COMPONENTS)))
    default_prefix, _, renamed_by_models, _ = COMPONENTS[component]
    prefix = prefix or default_prefix

    state = load_state(checkpoint_path)
    out = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not out:
        raise EveSenetError(
            "make_backbone_init: {} carries no {!r} keys. Either this is not an "
            "SE-Net checkpoint, or its encoder is named differently -- in which case "
            "the real Mask2Former pickle is needed and this shortcut does not "
            "apply".format(checkpoint_path, prefix))
    stray = sorted(k for k in out
                   if any(k.startswith(r) for r in renamed_by_models))
    if stray:
        # models.py rewrites keys with these prefixes before its strict load. Ours are
        # already in post-rename form, so the loop is a no-op; a key in pre-rename
        # form here would be rewritten and then fail the load in a confusing place.
        raise EveSenetError(
            "make_backbone_init: {} {} key(s) are in models.py's pre-rename form "
            "({} ...) -- it would rewrite them and the strict load would then "
            "disagree with the built module".format(
                len(stray), component, stray[:5]))
    return out


def extract_backbone(checkpoint_path, prefix=BACKBONE_PREFIX):
    """Backwards-compatible alias for :func:`extract_component`'s backbone case."""
    return extract_component(checkpoint_path, "backbone", prefix)


def component_path(out_path, component):
    """``data/M2F_R50.pkl`` -> that component's filename, as models.py derives it."""
    _, suffix, _, _ = COMPONENTS[component]
    if not suffix:
        return out_path
    root, ext = os.path.splitext(out_path)
    return root + suffix + ext


def write_component_init(checkpoint_path, out_path, component="backbone", force=False,
                         prefix=None):
    """Writes one init pickle; returns a summary. Never overwrites silently."""
    path = component_path(out_path, component)
    if os.path.exists(path) and not force:
        raise EveSenetError(
            "make_backbone_init: {} already exists. If it is the authors' genuine "
            "Mask2Former pickle, keep it -- it is the more faithful input. Pass "
            "--force only to deliberately replace it".format(path))
    state = extract_component(checkpoint_path, component, prefix)
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(state, path)
    expected = COMPONENTS[component][3]
    return {
        "component": component,
        "out": path,
        "out_sha256": sha256_file(path),
        "prefix": prefix or COMPONENTS[component][0],
        "n_tensors": len(state),
        "n_tensors_expected_osie": expected,
        "top_level_groups": sorted({k.split(".")[0] for k in state}),
    }


def write_backbone_init(checkpoint_path, out_path, force=False,
                        prefix=BACKBONE_PREFIX):
    """Backwards-compatible alias for :func:`write_component_init`'s backbone case."""
    return write_component_init(checkpoint_path, out_path, "backbone", force, prefix)


def write_all(checkpoint_path, out_path, force=False):
    """Both init pickles, plus provenance. This is what the run script calls."""
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "components": [write_component_init(checkpoint_path, out_path, c, force)
                       for c in ("backbone", "pixel_decoder")],
        "note": ("Initialisation only. load_model() overwrites every one of these "
                 "tensors with the same checkpoint's values via load_state_dict(..., "
                 "strict=False); the strict loads inside ImageFeatureEncoder are what "
                 "check completeness."),
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="F3 preflight -- derive SE-Net's encoder init pickles from the "
                    "released checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=os.path.join("data", "M2F_R50.pkl"),
                   help="the backbone pickle; the pixel decoder's name is derived "
                        "from it exactly as models.py derives it")
    p.add_argument("--component", choices=sorted(COMPONENTS) + ["all"], default="all")
    p.add_argument("--force", action="store_true",
                   help="replace existing files (e.g. the authors' own pickles)")
    args = p.parse_args(argv)

    try:
        if args.component == "all":
            summary = write_all(args.checkpoint, args.out, args.force)
            components = summary["components"]
        else:
            summary = write_component_init(args.checkpoint, args.out, args.component,
                                           args.force)
            components = [summary]
    except EveSenetError as exc:
        sys.stderr.write("FATAL F3 preflight: {}\n".format(exc))
        return 1

    print(json.dumps(summary, indent=2))
    for c in components:
        if c["n_tensors"] != c["n_tensors_expected_osie"]:
            sys.stderr.write(
                "make_backbone_init: NOTE -- {} has {} tensors, not the {} seen in "
                "the released OSIE checkpoint. Not an error, but say so in the run "
                "record\n".format(c["component"], c["n_tensors"],
                                  c["n_tensors_expected_osie"]))
    sys.stderr.write("make_backbone_init: OK -- wrote {}\n".format(
        ", ".join(c["out"] for c in components)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
