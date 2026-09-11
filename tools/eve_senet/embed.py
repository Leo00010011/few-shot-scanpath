"""Stage C: EVE support scanpaths -> a (38, 384) subject-embedding tensor (F3).

Drives the released SE-Net checkpoint over each participant's own 10-shot support set
and writes the tensor ``ISP/EVE/.../src/test.py --user_emb_path`` consumes (F5).

``SE-Net/src/builder.py::build()`` is deliberately not used. It calls
``common/dataset.py::process_data``, which raises ``NotImplementedError`` for any
dataset outside ``{OSIE, COCO-Search18, COCO-Freeview, MIT1003, CAT2000}``; it builds
two DataLoaders F3 has no use for; it constructs an AdamW optimiser (D8); and it
builds the eval loader with ``drop_last=True`` at ``batch_size // 2 == 8``, which on a
10-scanpath support set silently discards 20 % of the evidence. We construct
``UserEmbeddingNet`` directly with the same arguments ``build()`` passes, read from
the same config file. Nothing under ``SE-Net/`` is edited (convention 2).

Cluster-only: the ``UserEmbeddingNet`` import reaches Detectron2 and MSDeformAttn, so
it is deferred into :func:`load_model` and this module stays importable on the Windows
dev machine for :func:`select_support`'s unit tests (FR1.6).

CLI: see ``--help``, or ``bash/embed_eve_subjects.sh`` for the documented invocation.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENET_DIR = os.path.join(REPO_ROOT, "SE-Net")
if SENET_DIR not in sys.path:
    sys.path.insert(0, SENET_DIR)

from common.config import JsonConfig  # noqa: E402
from common.utils import transform_fixations  # noqa: E402

try:
    from . import EveSenetError
    from .check_env import write_versions
    from .dataset import EveSupportDataset, build_fix_labels, rescale_ratios
    from .durations import bin_occupancy, bin_scanpath_durations, decile_bins
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_senet import EveSenetError
    from eve_senet.check_env import write_versions
    from eve_senet.dataset import EveSupportDataset, build_fix_labels, rescale_ratios
    from eve_senet.durations import bin_occupancy, bin_scanpath_durations, decile_bins

EMBEDDING_DIM = 384          # == ISP's args.subject_feature_dim (FR7.3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Keys the released checkpoint legitimately does not carry. Established by running
# load_model() once against the real checkpoint and deciding per key (plan Step 8);
# anything outside this list raises (FR11.6). It is EMPTY until that run has
# happened -- an allowlist guessed in advance would defeat the check it exists for.
# `subject_predictor.*` is never allowlistable: such a key means `num_subjects`
# disagrees with the checkpoint and `strict=False` has left the head randomly
# initialised (FR6.2, validation Group 4).
MISSING_KEY_ALLOWLIST = ()
NEVER_ALLOWLIST_PREFIX = "subject_predictor."


# --------------------------------------------------------------------------- utils

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seeds(seed):
    """FR7.6 / D5. Pinned before any forward pass."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------- support selection

def select_support(records, num_fewshot, seed):
    """``{dense_subject: [name, ...]}`` -- exactly ``num_fewshot`` names each (FR4).

    The draw is per subject, over that subject's **lexicographically sorted**
    candidate names, seeded with the string ``"{seed}:{subject}"``. Sorting is what
    makes the result independent of input record order and of ``PYTHONHASHSEED``; the
    per-subject seed is what makes a second ``--seed`` a genuinely different draw
    from the same pool (OPEN-3, ``support_pool_size = 20 > num_fewshot = 10``).

    SE-Net's own ``select_fewshot_subject()`` is **not** used: it draws names from the
    *union* across the fewshot subjects and keeps whichever subjects have each. Our
    support pools are disjoint by stimulus, so one union draw of 10 names would give
    each subject ~10/38 scanpaths, unequally (FR4.3).
    """
    by_subject = defaultdict(list)
    for r in records:
        if r["split"] != "train":  # FR11.4
            raise EveSenetError(
                "select_support: record {!r} subject {} has split={!r}; only the "
                "'train' support pool may be selected from (FR2.2/FR11.4)".format(
                    r["name"], r["subject"], r["split"]))
        by_subject[r["subject"]].append(r)

    out = {}
    for s in sorted(by_subject):
        names = sorted(set(r["name"] for r in by_subject[s]))
        if len(names) < num_fewshot:  # FR11.2 -- names the thinnest subject
            raise EveSenetError(
                "select_support: dense subject {} has {} support scanpaths, fewer "
                "than num_fewshot={} (FR2.3/FR11.2)".format(s, len(names), num_fewshot))
        rng = random.Random("{}:{}".format(seed, s))
        out[s] = sorted(rng.sample(names, num_fewshot))
    return out


def assert_stimuli_exist(selection, image_dir):
    """FR11.3 -- a named stimulus absent from ``stimuli/`` is an error, not a skip."""
    missing = sorted({name for names in selection.values() for name in names
                      if not os.path.isfile(os.path.join(image_dir, name))})
    if missing:
        raise EveSenetError(
            "assert_stimuli_exist: {} selected stimuli absent from {} (FR11.3): "
            "{}{}".format(len(missing), image_dir, ", ".join(missing[:10]),
                          " ..." if len(missing) > 10 else ""))


# ------------------------------------------------------------------- model loading

def load_model(config_path, checkpoint, device, allow_missing=()):
    """Construct ``UserEmbeddingNet`` and load the released weights (FR6).

    Returns ``(model, hparams, missing_keys, unexpected_keys)``. Raises on any missing
    key outside ``MISSING_KEY_ALLOWLIST + allow_missing``, and unconditionally on any
    ``subject_predictor.*`` key (FR6.2/FR11.6).
    """
    from src.models import UserEmbeddingNet  # deferred: reaches Detectron2/MSDeformAttn

    hparams = JsonConfig(config_path)
    pa, mp, tp = hparams.Data, hparams.Model, hparams.Train
    if mp.embedding_dim != EMBEDDING_DIM:  # FR11.8
        raise EveSenetError(
            "load_model: Model.embedding_dim is {} but ISP's subject_feature_dim is "
            "{} (FR7.3/FR11.8)".format(mp.embedding_dim, EMBEDDING_DIM))

    model = UserEmbeddingNet(
        pa,
        num_decoder_layers=mp.n_dec_layers,
        hidden_dim=mp.embedding_dim,
        nhead=mp.n_heads,
        ntask=1,                       # free-viewing: one task (builder.py's n_tasks)
        num_output_layers=mp.num_output_layers,
        train_encoder=tp.train_backbone,
        train_pixel_decoder=tp.train_pixel_decoder,
        dropout=tp.dropout,
        dim_feedforward=mp.hidden_dim,
        num_encoder_layers=mp.n_enc_layers,
    ).to(device)

    ckp = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckp["model"], strict=False)
    missing, unexpected = list(missing), list(unexpected)

    head = [k for k in missing if k.startswith(NEVER_ALLOWLIST_PREFIX)]
    if head:
        raise EveSenetError(
            "load_model: the checkpoint does not carry {} -- Data.num_subjects ({}) "
            "disagrees with the checkpoint's, so strict=False has left the subject "
            "head randomly initialised. Fix num_subjects; do not allowlist "
            "(FR6.2)".format(head, pa.num_subjects))
    allowed = tuple(MISSING_KEY_ALLOWLIST) + tuple(allow_missing)
    unexplained = [k for k in missing if not any(k.startswith(a) for a in allowed)]
    if unexplained:
        raise EveSenetError(
            "load_model: {} missing key(s) outside the allowlist (FR11.6): {}. Decide "
            "per key whether the checkpoint genuinely omits it (then record it in "
            "MISSING_KEY_ALLOWLIST with a comment naming why) or whether the "
            "construction arguments are wrong.".format(len(unexplained), unexplained))

    model.eval()  # FR6.4 -- no optimiser is ever constructed (D8)
    return model, hparams, missing, unexpected


def build_transform(im_h, im_w):
    """Identical to ``common/dataset.py``'s ``transform_test``."""
    return T.Compose([
        T.Resize(size=(im_h, im_w)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ------------------------------------------------------------ per-subject embedding

@torch.no_grad()
def embed_subject(model, records, pa, device, transform, data_root, batch_size=8):
    """One embedding row: the **mean** of ``user_emb`` over the support set (FR7.1).

    This reproduces ``evaluate_user_siamese``'s accumulate-then-divide without its two
    defects: the ``num_fewshot == 1`` early return that saves an *un-normalised sum*,
    and the ``drop_last=True`` loader that would silently drop 2 of 10 scanpaths.
    """
    fix_labels, counters = build_fix_labels(
        records, im_h=pa.im_h, im_w=pa.im_w, max_traj_length=pa.max_traj_length,
        patch_size=pa.patch_size, patch_num=pa.patch_num)
    ds = EveSupportDataset(data_root, fix_labels, {}, pa, transform,
                           {"none": 0}, device, blur_action=True)
    if len(ds) != len(records):
        # Siamese_Triplet_Gaze.__init__ filters len(fixs) > max_traj_length; FR11.8
        # already raised on that, so a discrepancy here means something else did it.
        raise EveSenetError(
            "embed_subject: dataset holds {} items for {} records (FR5/FR11.8)".format(
                len(ds), len(records)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)  # FR7.2

    embs, n_seen = [], 0
    for batch in loader:
        b = batch["anchor"]
        inp_seq, inp_seq_high = transform_fixations(
            b["normalized_fixations"], b["is_padding"], pa, False, return_highres=True)
        inp_seq = inp_seq.to(device)
        logits = model(b["true_state"].to(device),
                       inp_seq,
                       (inp_seq == pa.pad_idx),
                       inp_seq_high.to(device),
                       b["duration"].to(device),
                       b["task_emb"].to(device))
        user_emb = logits["user_emb"].detach().cpu()
        if user_emb.shape[-1] != EMBEDDING_DIM:  # FR11.8
            raise EveSenetError(
                "embed_subject: user_emb has dim {}, expected {} (FR11.8)".format(
                    user_emb.shape[-1], EMBEDDING_DIM))
        embs.append(user_emb)
        n_seen += int(b["true_state"].size(0))

    if n_seen != len(records):  # FR11.5 -- the drop_last check
        raise EveSenetError(
            "embed_subject: {} items forward-passed for {} support scanpaths. At "
            "batch_size=8 with drop_last=True this would be 8 -- a silent 20 % loss "
            "of support evidence (FR7.2/FR11.5)".format(n_seen, len(records)))

    row = torch.cat(embs, dim=0).mean(dim=0).to(torch.float32)  # FR7.1
    if not bool(torch.isfinite(row).all()) or bool((row == 0).all()):  # FR11.5
        raise EveSenetError(
            "embed_subject: embedding row is all-zero or non-finite (FR7.5/FR11.5)")
    counters["forward_passes"] = n_seen
    return row, counters


# ------------------------------------------------------------------- full pipeline

def build_embeddings(fixations_path, subject_map_path, report_path, image_dir,
                     checkpoint, config, out_dir, num_fewshot=10, seed=0,
                     device="cuda", data_root=None, use_raw_durations=False,
                     allow_missing=()):
    """The whole of Stage C. Returns ``((N, 384) tensor, senet_report dict)``."""
    set_seeds(seed)
    data_root = data_root or os.path.join(REPO_ROOT, "data")

    with open(report_path) as fh:
        bridge = json.load(fh)
    with open(fixations_path) as fh:
        fixations = json.load(fh)
    with open(subject_map_path) as fh:
        subject_map = json.load(fh)

    fixations_sha = sha256_file(fixations_path)
    if fixations_sha != bridge["fixations_sha256"]:  # FR2.4 / FR11.1
        raise EveSenetError(
            "build_embeddings: sha256({}) = {} but bridge_report says {}. F2's "
            "artefacts are addressed positionally (convention 10), so a merely "
            "REORDERED fixations.json is correctly rejected too (FR11.1)".format(
                fixations_path, fixations_sha, bridge["fixations_sha256"]))

    n_subjects = int(bridge["num_subjects"])
    expected = {str(i) for i in range(n_subjects)}
    if set(subject_map["to_eve"]) != expected:  # FR11.7
        raise EveSenetError(
            "build_embeddings: subject_id_map to_eve keys are not {{0..{}}} "
            "(FR11.7)".format(n_subjects - 1))
    if set(r["subject"] for r in fixations) != set(range(n_subjects)):  # FR11.7
        raise EveSenetError(
            "build_embeddings: fixations.json dense subject set disagrees with "
            "bridge_report num_subjects={} (FR11.7)".format(n_subjects))
    for i in range(n_subjects):  # D4 roundtrip
        eve_id = subject_map["to_eve"][str(i)]
        if subject_map["to_dense"].get(eve_id) != i:
            raise EveSenetError(
                "build_embeddings: subject_id_map roundtrip fails at dense id {} "
                "({!r}) (D4/FR11.7)".format(i, eve_id))

    train = [r for r in fixations if r["split"] == "train"]  # FR2.2
    if int(bridge["min_support_per_subject"]) < num_fewshot:  # FR2.3 / FR11.2
        raise EveSenetError(
            "build_embeddings: min_support_per_subject={} < num_fewshot={}. The "
            "support pools are per-subject and share no image names, so the THINNEST "
            "pool binds -- not support_pool_size (FR2.3/FR11.2)".format(
                bridge["min_support_per_subject"], num_fewshot))

    edges, bin_of = decile_bins([t for r in train for t in r["T"]])  # FR3.2
    if use_raw_durations:
        # Validation Group 5's control arm only. The tensor must come back bitwise
        # identical, pinning the dead duration channel (FR3.4).
        prepared = [dict(r) for r in train]
    else:
        prepared = bin_scanpath_durations(train, bin_of)
    by_key = {(r["name"], r["subject"]): r for r in prepared}

    selection = select_support(train, num_fewshot, seed)  # FR4
    if sorted(selection) != list(range(n_subjects)):
        raise EveSenetError(
            "build_embeddings: support selection covers subjects {}, expected "
            "0..{} (FR4.1)".format(sorted(selection), n_subjects - 1))
    assert_stimuli_exist(selection, image_dir)  # FR11.3

    # Support/query disjointness, re-checked on the artefact F3 actually consumed
    # rather than trusted from the bridge report.
    test_names = {r["name"] for r in fixations if r["split"] == "test"}
    leaked = sorted({n for names in selection.values() for n in names} & test_names)
    if leaked:
        raise EveSenetError(
            "build_embeddings: {} selected support stimuli also appear in the scored "
            "test split (FR2.2): {}".format(len(leaked), leaked[:10]))

    model, hparams, missing, unexpected = load_model(
        config, checkpoint, device, allow_missing=allow_missing)
    pa = hparams.Data
    pa.image_path = image_dir  # read directly by Siamese_Triplet_Gaze.process_data
    pa.num_fewshot = num_fewshot
    transform = build_transform(pa.im_h, pa.im_w)

    task_emb_path = os.path.join(data_root, "osie_embeddings.npy")  # FR6.5
    if not os.path.isfile(task_emb_path):
        raise EveSenetError(
            "build_embeddings: task embedding {} not found. Siamese_Triplet_Gaze "
            "hardcodes this path for every non-COCO-Search18 dataset "
            "(FR6.5)".format(task_emb_path))
    task_emb_dict = np.load(task_emb_path, allow_pickle=True).item()
    if not task_emb_dict:
        raise EveSenetError("build_embeddings: {} holds an empty dict (FR6.5)".format(
            task_emb_path))
    task_emb_key = list(task_emb_dict)[0]  # positional: process_data discards the key

    rows, counters = [], Counter()
    per_subject = {}
    for s in range(n_subjects):  # FR7.4 -- ascending dense id, always
        recs = [by_key[(name, s)] for name in selection[s]]
        row, c = embed_subject(model, recs, pa, device, transform, data_root)
        rows.append(row)
        counters.update(c)
        per_subject[str(s)] = {"eve_id": subject_map["to_eve"][str(s)],
                               "n_forward": c["forward_passes"],
                               "row_norm": float(torch.linalg.norm(row))}

    emb = torch.stack(rows, dim=0)  # (N, 384) FR7.3
    if tuple(emb.shape) != (n_subjects, EMBEDDING_DIM) or emb.dtype != torch.float32:
        raise EveSenetError(
            "build_embeddings: tensor is {} {}, expected ({}, {}) float32 "
            "(FR7.3)".format(tuple(emb.shape), emb.dtype, n_subjects, EMBEDDING_DIM))
    zero_rows = [i for i in range(n_subjects) if bool((emb[i] == 0).all())]
    if zero_rows:  # FR7.5 / FR11.5
        raise EveSenetError(
            "build_embeddings: all-zero embedding row(s) for dense subject(s) {} "
            "({}). The released OSIE tensor's rows 5-9 are zero for exactly this "
            "reason; ours must have none (FR7.5)".format(
                zero_rows, [subject_map["to_eve"][str(i)] for i in zero_rows]))

    os.makedirs(out_dir, exist_ok=True)
    emb_name = "eve_fewshot_user_embedding_{}_seed{}.pt".format(num_fewshot, seed)
    emb_path = os.path.join(out_dir, emb_name)
    torch.save(emb, emb_path)

    report = {
        "args": {
            "fixations": fixations_path, "subject_map": subject_map_path,
            "report": report_path, "image_dir": image_dir, "checkpoint": checkpoint,
            "config": config, "out_dir": out_dir, "num_fewshot": num_fewshot,
            "seed": seed, "device": str(device),
            "use_raw_durations": bool(use_raw_durations),
        },
        "embedding_file": emb_name,
        "num_subjects": n_subjects,
        "embedding_dim": EMBEDDING_DIM,
        "duration_bin_edges": [float(e) for e in edges],           # FR3.3
        "duration_bin_occupancy": bin_occupancy(
            bin_scanpath_durations(train, bin_of)),
        "support_selection": {str(k): v for k, v in sorted(selection.items())},  # FR4.4
        "per_subject": per_subject,
        "senet_input_size": [pa.im_h, pa.im_w],                    # FR5.3
        "senet_rescale": rescale_ratios(pa.im_h, pa.im_w),         # FR5.3
        "task_emb_source": os.path.relpath(task_emb_path, REPO_ROOT).replace("\\", "/"),
        "task_emb_key": str(task_emb_key),
        "missing_keys": missing,                                   # FR6.2
        "unexpected_keys": unexpected,
        "fixations_sha256": fixations_sha,
        "checkpoint_sha256": sha256_file(checkpoint),
        "embedding_sha256": sha256_file(emb_path),                 # FR8.2
        "counters": {                                              # D7 -- always present
            "n_records": int(counters["n_records"]),
            "forward_passes": int(counters["forward_passes"]),
            "oob_fixations_dropped": int(counters["oob_fixations_dropped"]),
            "len1_scanpaths": int(counters["len1_scanpaths"]),
            "truncated_scanpaths": int(counters["truncated_scanpaths"]),
        },
        "notes": {
            "duration_channel": (
                "SE-Net/src/models.py adds duration_encoding into ventral_pos and "
                "calls ventral_pos.fill_(0) on the next line, after ventral_embs += "
                "ventral_pos. The duration never reaches the network (FR3.4). Bins "
                "are fed anyway as the faithful input contract; validation Group 5 "
                "pins the deadness with a bitwise-identity check."),
            "senet_squash": (
                "1920x1080 -> 512x320 is a SECOND, different squash from F5's "
                "1920x1080 -> 512x384 metric screen (OPEN-4). Both reach F7."),
            "checkpoint_provenance": (
                "The SE-Net checkpoint is OSIE-trained; the embeddings are ours. "
                "What transfers is the encoder, not the subjects (F7)."),
        },
    }
    with open(os.path.join(out_dir, "senet_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")
    write_versions(os.path.join(out_dir, "versions.txt"))  # FR1.5
    return emb, report


def main(argv=None):
    p = argparse.ArgumentParser(description="F3 -- EVE subject embeddings (Stage C)")
    p.add_argument("--fixations", default="data/eve_bridge/fixations.json")
    p.add_argument("--subject-map", dest="subject_map",
                   default="data/eve_bridge/subject_id_map.json")
    p.add_argument("--report", default="data/eve_bridge/bridge_report.json")
    p.add_argument("--image-dir", dest="image_dir", default="data/eve_bridge/stimuli")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="SE-Net/configs/eve_useremb.json")
    p.add_argument("--out-dir", dest="out_dir", default="data/eve_senet")
    p.add_argument("--data-root", dest="data_root", default=None,
                   help="holds osie_embeddings.npy; defaults to <repo>/data")
    p.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-cpu", dest="allow_cpu", action="store_true",
                   help="permit --device cpu; without it a silently CPU-bound run "
                        "cannot be mistaken for a normal one")
    p.add_argument("--raw-durations", dest="raw_durations", action="store_true",
                   help="validation Group 5's control arm: feed raw ms instead of "
                        "decile bins. The tensor must come back bitwise identical")
    p.add_argument("--allow-missing-key", dest="allow_missing", action="append",
                   default=[], help="state-dict key prefix the checkpoint may omit; "
                                    "record resolved entries in MISSING_KEY_ALLOWLIST")
    args = p.parse_args(argv)

    if args.device.startswith("cpu") and not args.allow_cpu:
        sys.stderr.write("FATAL: --device cpu requires --allow-cpu (F3 is GPU work)\n")
        return 2
    try:
        emb, report = build_embeddings(
            args.fixations, args.subject_map, args.report, args.image_dir,
            args.checkpoint, args.config, args.out_dir,
            num_fewshot=args.num_fewshot, seed=args.seed, device=args.device,
            data_root=args.data_root, use_raw_durations=args.raw_durations,
            allow_missing=tuple(args.allow_missing))
    except EveSenetError as exc:
        sys.stderr.write("FATAL F3 failure: {}\n".format(exc))
        return 1
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("support_selection", "per_subject")}, indent=2))
    sys.stderr.write("embed: OK -- {} wrote {}\n".format(
        tuple(emb.shape), report["embedding_file"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
