"""Derive ``data/M2F_R50.pkl`` from the released SE-Net checkpoint (F3 preflight).

``SE-Net/src/models.py::ImageFeatureEncoder.__init__`` unconditionally does
``torch.load(cfg.MODEL.WEIGHTS)`` — ``data/M2F_R50.pkl`` per ``data/resnet50.yaml`` —
and loads it into the freshly built Detectron2 ResNet **strictly**. That file ships
with neither this repository nor the released checkpoint bundle, so model construction
dies with ``FileNotFoundError`` before F3 reaches a single forward pass.

It is an *initialisation* file. ``load_model()`` calls
``model.load_state_dict(ckp["model"], strict=False)`` immediately afterwards, and the
released checkpoint carries **all 265** backbone tensors under
``encoder.backbone.*`` — so every value the pickle supplies is overwritten before any
image is seen. What the file has to be is *structurally* complete, not any particular
set of numbers.

Hence this tool: rather than download a third-party Mask2Former R50 pickle whose
provenance we cannot check against the checkpoint we actually run, we lift the
backbone weights **out of that checkpoint**. The encoder then provably ends at the
released weights by two independent routes, and there is no new external dependency.

Key naming is left exactly as the checkpoint stores it — ``stem.*`` and
``stages.res*``. Reconciling that against whatever the installed detectron2 calls its
stages is :func:`embed.align_stage_prefix`'s job, in one place, for both the init file
and the checkpoint (detectron2 0.6 wants bare ``res*``; the authors' version wanted
``stages.res*``). **The strict load inside ``ImageFeatureEncoder`` is the completeness
check** — a missing tensor raises there.

Nothing under ``SE-Net/`` is edited (convention 2); this writes one file under
``data/``, which is git-ignored (convention 5).

CPU-only. Run it once per checkout, on the cluster, before ``embed_eve_subjects.sh``:

    python tools/eve_senet/make_backbone_init.py \
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

BACKBONE_PREFIX = "encoder.backbone."
EXPECTED_N_TENSORS = 265   # the released OSIE ckp_11999.pt; reported, never enforced


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_backbone(checkpoint_path, prefix=BACKBONE_PREFIX):
    """``{stripped_key: tensor}`` for every ``prefix``-prefixed entry."""
    if not os.path.isfile(checkpoint_path):
        raise EveSenetError(
            "make_backbone_init: no such checkpoint: {}".format(checkpoint_path))
    ckp = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckp, dict) and "model" in ckp:
        state = ckp["model"]
    elif isinstance(ckp, dict):
        state = ckp
    else:
        raise EveSenetError(
            "make_backbone_init: {} holds {}, not a state dict".format(
                checkpoint_path, type(ckp).__name__))

    out = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not out:
        raise EveSenetError(
            "make_backbone_init: {} carries no {!r} keys. Either this is not an "
            "SE-Net checkpoint, or its encoder is named differently -- in which case "
            "the real Mask2Former R50 pickle is needed and this shortcut does not "
            "apply".format(checkpoint_path, prefix))
    stray = sorted(k for k in out if k.startswith("res"))
    if stray:
        # models.py renames leading-"res" keys to "stages." + k. Ours are already
        # "stages.res*", so the loop is a no-op; a bare "res*" key here would be
        # renamed and then fail the strict load in a confusing place.
        raise EveSenetError(
            "make_backbone_init: {} key(s) begin with 'res' ({} ...). models.py "
            "would rename them to 'stages.' + key and the strict load would then "
            "disagree with the built backbone".format(len(stray), stray[:5]))
    return out


def write_backbone_init(checkpoint_path, out_path, force=False,
                        prefix=BACKBONE_PREFIX):
    """Writes ``out_path``; returns a JSON-able summary. Never overwrites silently."""
    if os.path.exists(out_path) and not force:
        raise EveSenetError(
            "make_backbone_init: {} already exists. If it is the authors' genuine "
            "Mask2Former pickle, keep it -- it is the more faithful input. Pass "
            "--force only to deliberately replace it".format(out_path))
    state = extract_backbone(checkpoint_path, prefix)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(state, out_path)
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "out": out_path,
        "out_sha256": sha256_file(out_path),
        "prefix": prefix,
        "n_tensors": len(state),
        "n_tensors_expected_osie": EXPECTED_N_TENSORS,
        "top_level_groups": sorted({k.split(".")[0] for k in state}),
        "note": ("Backbone initialisation only. load_model() overwrites every one of "
                 "these with the same checkpoint's values via load_state_dict(..., "
                 "strict=False); the strict load inside ImageFeatureEncoder is what "
                 "checks completeness."),
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="F3 preflight -- derive data/M2F_R50.pkl from the SE-Net checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=os.path.join("data", "M2F_R50.pkl"))
    p.add_argument("--prefix", default=BACKBONE_PREFIX)
    p.add_argument("--force", action="store_true",
                   help="replace an existing --out (e.g. the authors' own pickle)")
    args = p.parse_args(argv)
    try:
        summary = write_backbone_init(args.checkpoint, args.out, args.force, args.prefix)
    except EveSenetError as exc:
        sys.stderr.write("FATAL F3 preflight: {}\n".format(exc))
        return 1
    print(json.dumps(summary, indent=2))
    if summary["n_tensors"] != EXPECTED_N_TENSORS:
        sys.stderr.write(
            "make_backbone_init: NOTE -- {} tensors, not the {} seen in the released "
            "OSIE checkpoint. Not an error, but say so in the run record\n".format(
                summary["n_tensors"], EXPECTED_N_TENSORS))
    sys.stderr.write("make_backbone_init: OK -- wrote {}\n".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
