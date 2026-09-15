import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import scipy.stats

import collections

import time
import os
import argparse
from os.path import join
from tqdm import tqdm
import datetime
import json
import sys

from dataset.dataset import EVE_evaluation
from utils.evaluation import comprehensive_evaluation_by_subject
from utils.logger import Logger
from utils.data_postprocess import *
from models.sampling import Sampling

from models.gazeformer import gazeformer
from models.models import Transformer

# tools/ carries eve_eval (the preflight), eve_prep (the torch-free trial_keys) and
# eve_bridge (the heatmap store and the authors' NSS/CC/KLD wrapper). bash/test_eve.sh
# exports PYTHONPATH; this makes a bare `python src/test.py` work too.
PROJECT_ROOT = os.path.abspath(join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", ".."))
if join(PROJECT_ROOT, "tools") not in sys.path:
    sys.path.insert(0, join(PROJECT_ROOT, "tools"))

from eve_eval import EveEvalError
from eve_eval.check_eval import check_eval, sha256_file
from eve_prep.trial_keys import load_trial_exp_keys
from eve_bridge.store import GtHeatmapStore
from eve_bridge.heatmap_metrics import score_step_heatmaps

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="test", help="Selecting running mode (default: test)")
parser.add_argument("--img_dir", type=str, default="../../../data/eve_bridge/stimuli", help="Directory to the image data (stimuli) -- accepted for signature parity and NEVER read (FR4.2, OPEN-6)")
parser.add_argument("--feat_dir", type=str, default="../../../data/eve_features/image_features",
                    help="Directory to the PER-TRIAL image feature data, keyed by exp_key")
parser.add_argument("--emb_dir", type=str, default="../../../data/eve_features/embeddings.npy",
                        help="Directory to the task text embedding (byte-identical to OSIE's)")
parser.add_argument("--width", type=int, default=512, help="Width of input data")
parser.add_argument("--height", type=int, default=384, help="Height of input data")
parser.add_argument("--origin_width", type=int, default=1920, help="original Width of input data")
parser.add_argument("--origin_height", type=int, default=1080, help="original Height of input data")
parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--gpu_ids", type=list, default=[0], help="Used gpu ids")
parser.add_argument("--lambda_1", type=float, default=1.0, help="Hyper-parameter for duration loss term")

parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation -- PINNED to 1 (FR6.5)")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
parser.add_argument('--lm_hidden_dim', default=768, type=int,
                    help="Dimensionality of target embeddings from language model")
parser.add_argument('--encoder_dropout', default=0.1, type=float, help="Encoder dropout rate")
parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")

parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
parser.add_argument("--subject_feature_dim", type=int, default=384, help="The dim of the subject feature")
parser.add_argument("--action_map_num", type=int, default=4, help="The dim of action map")

parser.add_argument("--fix_dir", type=str, default="../../../data/eve_bridge/fixations.json", help="Directory to the raw fixation file")
parser.add_argument("--evaluation_dir", type=str,
                    default="src/assets/EVE-eval")
parser.add_argument("--user_emb_path", default="../../../data/eve_senet/seed0/eve_fewshot_user_embedding_10_seed0.pt", type=str, help="F3's (38, 384) subject embedding table")
parser.add_argument("--ex_subject", nargs='+', type=int, default=[-1], help='Skip unseen subjects for training and evaluation on base set')
parser.add_argument("--fewshot_subject", nargs='+', type=int, default=[-1], help="Unseen subject ids for scanpath prediction on query set -- MUST be ascending 0..N-1 (D4, FR4.4)")
parser.add_argument("--num_fewshot", type=int, default=10, help="number of images from the new subject in few-shot learning")
parser.add_argument("--fewshot_finetune_path", type=str, default="", help="pretrained model for few-shot learning")
parser.add_argument("--subject_num", type=int, default=3, help="The number of subjects scored per image (EVE: 3)")
parser.add_argument("--random_support", type=int, default=0, help="random seed to choose support set")

# ---- FR6.3, the EVE-only arguments ----------------------------------------------
parser.add_argument("--max_batches", type=int, default=-1,
                    help="-1 = no cap. Replaces upstream's `if i_batch > 100: break`, "
                         "which is inert for OSIE (70 batches) and LIVE for EVE (354)")
parser.add_argument("--heatmap_dir", type=str, default="",
                    help="path to gt_heatmaps.h5; empty disables the NSS/CC/KLD block")
parser.add_argument("--subject_map_path", type=str, default="",
                    help="path to subject_id_map.json -- required (FR7)")
parser.add_argument("--bridge_report", type=str, default="", help="bridge_report.json")
parser.add_argument("--senet_report", type=str, default="", help="senet_report.json")
parser.add_argument("--feature_report", type=str, default="", help="feature_report.json")
parser.add_argument("--metrics_json", type=str, default="metrics.json",
                    help="filename, under the log folder, for the FR9 record")

args = parser.parse_args()

# FR6.5 -- evaluation.py's collectors are shaped (n_images, subject_num, subject_num)
# and a repeat drives row_idx past subject_num into an IndexError (TechStack section 5).
if args.eval_repeat_num != 1:
    parser.error(
        "--eval_repeat_num must be 1 on the EVE branch, got {}. evaluation.py's "
        "collectors are shaped (n_images, subject_num={}, subject_num) and any repeat "
        "drives row_idx past subject_num into an IndexError (TechStack section 5)."
        .format(args.eval_repeat_num, args.subject_num))

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform = transforms.Compose([
                                transforms.Resize((args.height * 2, args.width * 2)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


def resolved_versions():
    """FR1.2 -- the stack this run actually resolved to, recorded not enforced.

    The one exception is multimatch-gaze, which the caller pins: TechStack section 1.1
    records that F1's green run was several major versions off the pin list on
    everything else and still landed on the published row, but that 0.1.3 is the one
    metric-critical pin that was held.
    """
    import importlib

    out = {"python": sys.version.replace("\n", " ")}
    for name in ("torch", "torchvision", "numpy", "scipy", "skimage", "cv2",
                 "multimatch_gaze", "h5py", "pandas"):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "<none>")
        except Exception as exc:
            out[name] = "MISSING: {!r}".format(exc)
    try:
        out["cuda"] = torch.version.cuda
        out["gpu_name"] = (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "<no cuda device>")
    except Exception as exc:
        out["gpu_name"] = "MISSING: {!r}".format(exc)
    return out


def multimatch_nan_drops(score_details, subject_num):
    """FR13.6 -- how many diagonal cells `is_eliminating_nan=True` silently dropped.

    ``evaluation.py`` drops any diagonal MultiMatch row containing a NaN *before* the
    mean, which changes the denominator; SED/STDE get no such treatment. The count has
    to be surfaced (D7), so it is recovered from ``score_details`` -- the evaluator's
    own ``scores_of_each_images``, whose first five channels on the diagonal are the
    five MultiMatch dimensions.
    """
    arr = np.asarray(score_details, dtype=np.float64)
    if arr.ndim != 4:
        return {"n_diagonal_cells": 0, "n_dropped": 0, "per_dimension": {}}
    names = ("vector", "direction", "length", "position", "duration")
    n_images = arr.shape[0]
    k = min(subject_num, arr.shape[1], arr.shape[2])
    diag = np.stack([arr[:, s, s, :5] for s in range(k)], axis=1)  # (I, k, 5)
    flat = diag.reshape(-1, 5)
    dropped = np.isnan(flat.sum(axis=1))
    return {
        "n_diagonal_cells": int(flat.shape[0]),
        "n_dropped": int(dropped.sum()),
        "fraction_dropped": float(dropped.mean()) if flat.shape[0] else 0.0,
        "per_dimension": {names[d]: int(np.isnan(flat[:, d]).sum()) for d in range(5)},
        "note": ("is_eliminating_nan=True drops these rows before the MultiMatch mean "
                 "and therefore changes its denominator; SED/STDE are not filtered "
                 "(D7, FR13.6). n_images={}".format(n_images)),
    }


def reporting_notes(args, nan_drops, min_support=None):
    """FR13.1 - FR13.6. Strings, so the record carries its own caveats (D5, D6)."""
    return {
        "retrieval": (
            "FR13.1 -- retrieval is reported SEPARATELY from the paper-comparable "
            "block. At subject_num = {} every rank lies in {{0..{}}}, so R@3 is "
            "STRUCTURALLY SATURATED at 100% and R@5 doubly so. Quote R@1 and MRR. "
            "This branch's p2g() computes r3 as `rank < 3`, so the field really is "
            "R@3 here -- the `rank < 2` defect is COCO_FV's (TechStack section 4.2)."
            .format(args.subject_num, args.subject_num - 1)),
        "cohort": (
            "FR13.2 -- 38 participants over 354 scored stimuli, 1062 cells, 3 "
            "subjects per image WITH IDENTITIES VARYING BY IMAGE. The evaluator's "
            "diagonal is positional, so this is legitimate -- but a reader will "
            "assume a fixed trio unless told (Mission P4, OPEN-5)."),
        "squashes": (
            "FR13.3 -- three independent distortions apply and all three reach F7: "
            "the metric squash 1920x1080 -> 512x384 (3.75 / 2.8125, OPEN-4); F4's "
            "feature squash 1920x1080 -> 1024x768 (0.5333 / 0.7111); F3's SE-Net "
            "input squash to 512x320 (0.2667 / 0.2963). None is uniform."),
        "bridge_counters": (
            "FR13.4 -- carried from bridge_report.json: short_scanpath (padded to "
            "length 3 INSIDE the frozen evaluator, and the padded array then replaces "
            "the original for all subsequent metrics in that cell), clamped_coords, "
            "incomplete_stimulus, surplus_trial (images seen by 4 participants "
            "contribute only 3)."),
        "checkpoint_provenance": (
            "FR13.5 -- both the ISP checkpoint and the SE-Net checkpoint that produced "
            "our embeddings were trained on OSIE. What transfers is the "
            "representation, not the subjects (OPEN-2). And OPEN-7: our own OSIE "
            "reproduction lands ~1% on the WORSE side of the published row on all "
            "three headline metrics, so any comparison of these numbers to the paper "
            "inherits that offset."),
        "multimatch_nan_drops": (
            "FR13.6 -- {} of {} diagonal MultiMatch cells were dropped as NaN before "
            "the mean (is_eliminating_nan=True), which changes the MultiMatch "
            "denominator. SED/STDE are not filtered.".format(
                nan_drops["n_dropped"], nan_drops["n_diagonal_cells"])),
        "sed_stde_aliases": (
            "FR9.3 -- SED_best and STDE_best are ALIASES of SED and STDE, not a "
            "best-of-N: evaluation.py assigns them outright with no selection step, "
            "in every configuration and at any eval_repeat_num. Report SED and STDE "
            "once each; listing the _best pair alongside them reads as corroboration "
            "that does not exist (TechStack section 4)."),
        "uninitialised_is_minus_one": (
            "An uninitialised evaluator cell is -1, NOT NaN. An all--1 block means the "
            "loop never ran, not a bad score (TechStack section 4)."),
        "std_provenance": (
            "FR10.4 / D6 -- per_cell_std is null here: cur_metrics_std is deferred to "
            "F6, which recomputes it from prediction.json on CPU. The spread quoted "
            "for F5 is the ACROSS-SEED one over seeds 0/1/2, which is a different "
            "quantity from the per-(image, subject)-cell std."),
        "min_support_per_subject": (
            "FR3.5 -- num_fewshot = {} against min_support_per_subject = {}; the bound "
            "is the MINIMUM over subjects, never support_pool_size.".format(
                args.num_fewshot, min_support)),
    }


def main():

    # load logger
    log_dir = args.evaluation_dir
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    log_info_folder = os.path.join('result', log_dir.split('/')[-1], "log")
    log_file = os.path.join(log_info_folder, "log_test_subject_{}_{}.txt".format(args.num_fewshot, args.random_support))
    open(log_file, 'w').close() \
        if os.makedirs(log_info_folder, exist_ok=True) is None else None
    logger = Logger(log_file)

    logger.info("The args corresponding to testing process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    # ---- FR3 / FR6.3: every handshake BEFORE the model is built -----------------
    # Run in-process so a direct `python src/test.py` is as safe as the script path.
    # fast=True: bash/test_eve.sh runs the full 1062-tensor hash sweep once per
    # allocation, and repeating 6.7 GB of hashing per seed buys nothing.
    for flag, value in (("--subject_map_path", args.subject_map_path),
                        ("--bridge_report", args.bridge_report),
                        ("--senet_report", args.senet_report),
                        ("--feature_report", args.feature_report)):
        if not value:
            raise EveEvalError(
                "{} is required on the EVE branch (FR6.3): the artefact handshakes "
                "are what stop F2, F3 and F4 from silently discussing different "
                "builds.".format(flag))

    preflight = check_eval(
        bridge_dir=os.path.dirname(os.path.abspath(args.fix_dir)),
        senet_dir=os.path.dirname(os.path.abspath(args.user_emb_path)),
        feature_dir=os.path.dirname(os.path.abspath(args.feature_report)),
        weights_dir=args.evaluation_dir,
        subject_num=args.subject_num, num_fewshot=args.num_fewshot, fast=True,
        fixations_path=args.fix_dir,
        heatmaps_path=(args.heatmap_dir or
                       join(os.path.dirname(os.path.abspath(args.fix_dir)),
                            "gt_heatmaps.h5")),
        subject_map_path=args.subject_map_path,
        bridge_report_path=args.bridge_report,
        senet_report_path=args.senet_report,
        feature_report_path=args.feature_report,
        user_emb_path=args.user_emb_path,
        emb_npy_path=args.emb_dir,
        feat_dir=args.feat_dir,
        checkpoint_path=os.path.join(checkpoints_dir, "checkpoint_best.pth"),
        split="test")
    if not preflight["ok"]:
        for i, msg in enumerate(preflight["failures"], 1):
            logger.info("PREFLIGHT FAILURE [{}] {}".format(i, msg))
        raise EveEvalError(
            "{} preflight failure(s) -- see the log above (FR3, D7)".format(
                len(preflight["failures"])))
    logger.info("preflight OK: {}".format(json.dumps(preflight["counts"])))

    with open(args.subject_map_path) as fh:
        to_eve = json.load(fh)["to_eve"]
    with open(args.bridge_report) as fh:
        bridge_report = json.load(fh)

    # ---- FR4.3: the (name, subject) -> exp_key map, built ONCE ------------------
    # load_trial_exp_keys is torch-free and reads only trials/{trial_key,exp_key};
    # the ~88 MB trials/heatmaps is never touched here. Its fixations_sha256 gate
    # fires before any mapping is returned.
    exp_key_map = load_trial_exp_keys(
        args.heatmap_dir or join(os.path.dirname(os.path.abspath(args.fix_dir)),
                                 "gt_heatmaps.h5"),
        fixations_path=args.fix_dir)

    test_dataset = EVE_evaluation(
        args, args.img_dir, args.feat_dir, args.fix_dir, args.emb_dir,
        heatmaps_dir=args.heatmap_dir, exp_key_map=exp_key_map,
        action_map=(args.im_h, args.im_w),
        # FR6.2 -- EXPLICIT. OSIE's test.py omits this and relies on the (600, 800)
        # default while parsing --origin_width/--origin_height into unused variables.
        # Omitting it here mis-scales every coordinate by 2.4x / 1.8x and every metric
        # still returns a number (D3).
        origin_size=(args.origin_height, args.origin_width),
        resize=(args.height, args.width), max_length=args.max_length,
        type="test", transform=transform)

    if (test_dataset.resizescale_x, test_dataset.resizescale_y) != (3.75, 2.8125):
        raise EveEvalError(
            "resize scales are ({}, {}), expected (3.75, 2.8125) -- origin_size did "
            "not take effect and every coordinate would be mis-scaled (D3, "
            "FR6.2)".format(test_dataset.resizescale_x, test_dataset.resizescale_y))

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_func
    )

    # ---- FR8.1: the ground-truth heatmap store, loaded once ---------------------
    # GtHeatmapStore.load()'s fixations_sha256 gate is the safety net for its
    # POSITIONAL row addressing: row i is record i of fixations.json.
    store = (GtHeatmapStore.load(args.heatmap_dir, fixations_path=args.fix_dir)
             if args.heatmap_dir else None)

    device = torch.device('cuda')

    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead=args.nhead,
                              subject_feature_dim=args.subject_feature_dim, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, encoder_dropout=args.encoder_dropout,
                              decoder_dropout=args.decoder_dropout, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, args=args).cuda()

    model = gazeformer(transformer, spatial_dim=(args.im_h, args.im_w), args=args,
                       subject_num=args.subject_num, subject_feature_dim=args.subject_feature_dim,
                       action_map_num=args.action_map_num,
                       dropout=args.cls_dropout, max_len=args.max_length).cuda()


    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length,
                        map_width=args.im_w, map_height=args.im_h,
                        width=args.width, height=args.height)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"), map_location=device)
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            print(f'loading checkpoint from {checkpoints_dir}')
            model.load_state_dict(test_checkpoint[key])


    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)


    model.eval()
    repeat_num = args.eval_repeat_num
    all_gt_fix_vectors = []
    all_predict_fix_vectors = []
    predict_results = []
    n_batches = 0
    hm_sum = {"NSS": 0.0, "CC": 0.0, "KLD": 0.0}
    hm_M = 0
    with tqdm(total=len(test_loader) * repeat_num) as pbar_test:
        for i_batch, batch in enumerate(test_loader):
            # FR6.4 -- upstream's `if i_batch > 100: break` is inert for OSIE (70
            # batches) and LIVE for EVE (354): it would silently stop at 101 images,
            # 29% of the test set, and report a plausible wrong number. Parameterised
            # rather than deleted so the resolved value lands in the logged arg
            # namespace and the log itself is the evidence nothing was truncated (D5).
            if args.max_batches > 0 and i_batch >= args.max_batches:
                break
            n_batches += 1
            tmp = [batch["images"], batch["fix_vectors"], batch["task_embeddings"], batch["subjects"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            # merge the first two dim
            tmp = [_.view(-1, *_.shape[2:]) if torch.is_tensor(_) else _ for _ in tmp]
            images, gt_fix_vectors, task_embeddings, subjects = tmp
            # task = images.new_zeros((images.shape[0], args.lm_hidden_dim))

            N, _, C = images.shape

            # Flattened in EXACTLY the order the tensors were: image-major, then
            # subject. Row k of these lists belongs to row k of `images` and of
            # `all_actions_prob` (FR8.2, FR7.2).
            flat_keys = [k for per_image in batch["trial_keys"] for k in per_image]
            flat_len = batch["lengths"].reshape(-1).tolist()
            flat_subjects = batch["subjects"].reshape(-1).tolist()

            with torch.no_grad():
                predict = model(src=images, subjects=subjects, task=task_embeddings)

            log_normal_mu = predict["log_normal_mu"]
            log_normal_sigma2 = predict["log_normal_sigma2"]
            all_actions_prob = predict["all_actions_prob"]

            # ---- FR8: the supplementary NSS / CC / KLD block --------------------
            if store is not None:
                if len(flat_keys) != all_actions_prob.shape[0]:
                    raise ValueError(
                        "heatmap batch misalignment: {} trial keys vs {} predictions "
                        "(FR8.2)".format(len(flat_keys), all_actions_prob.shape[0]))
                gt = torch.from_numpy(store.get_batch(flat_keys)).to(
                    all_actions_prob.device)
                if gt.shape[0] != all_actions_prob.shape[0]:
                    raise ValueError(
                        "heatmap batch misalignment: {} gt rows vs {} predictions "
                        "(FR8.2)".format(gt.shape[0], all_actions_prob.shape[0]))
                # Argument order is PREDICTION FIRST, ground truth second. KLD is not
                # symmetric and reversing it yields a plausible wrong number (FR8.3).
                s = score_step_heatmaps(all_actions_prob, gt, flat_len,
                                        max_length=args.max_length)
                # FR8.4 -- weight each batch by its valid-timestep count so the run's
                # value is the mean over all valid timesteps, not a mean of batch means.
                m = sum(min(int(v), args.max_length) for v in flat_len)
                for k in hm_sum:
                    hm_sum[k] += s[k] * m
                hm_M += m

            image_prediction_dict = {_: [] for _ in range(len(batch["img_names"]))}
            all_gt_fix_vectors.extend(gt_fix_vectors)
            for trial in range(repeat_num):
                samples = sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)
                prob_sample_actions = samples["selected_actions_probs"]
                durations = samples["durations"]
                sample_actions = samples["selected_actions"]
                sampling_random_predict_fix_vectors, _, _ = sampling.generate_scanpath(
                    images, prob_sample_actions, durations, sample_actions)
                for idx in range(len(batch["img_names"])):
                    image_prediction_dict[idx].extend(
                        sampling_random_predict_fix_vectors[idx * args.subject_num:(idx + 1) * args.subject_num])
                    for subject_idx in range(args.subject_num):
                        slot = idx * args.subject_num + subject_idx
                        pred = sampling_random_predict_fix_vectors[slot]
                        # ---- FR7.2 (D4) -- the subject id comes from the BATCH ----
                        # get_prediction_list() writes args.fewshot_subject[subject_idx],
                        # which is correct for OSIE only by accident (5 entries, five
                        # subjects). Here fewshot_subject has 38 entries and
                        # subject_num is 3, so it would stamp participant 0/1/2 onto
                        # every one of the 354 images while the actual trio varies per
                        # image. The metrics are unharmed -- the evaluator's diagonal
                        # is positional -- but prediction.json would be a lie, F6 would
                        # re-score against the wrong ground truth, and D4's question
                        # "which of my real subjects is row 3?" would get a confident
                        # wrong answer.
                        dense = int(flat_subjects[slot])
                        if str(dense) not in to_eve:
                            raise KeyError(
                                "dense subject id {} is absent from "
                                "subject_id_map['to_eve'] (FR7.5)".format(dense))
                        predict_results.append({
                            "name": batch["img_names"][idx],
                            "subject": dense,
                            "subject_eve": to_eve[str(dense)],
                            "X": [int(item[0]) for item in pred],
                            "Y": [int(item[1]) for item in pred],
                            "T": [int(round(item[2] * 1000, 3)) for item in pred],
                        })
                pbar_test.update(1)

            all_predict_fix_vectors.extend(list(image_prediction_dict.values()))

    # FR6.4 -- the belt for the cap. A run that silently covered part of the split
    # must not report (D7).
    if args.max_batches <= 0 and n_batches != len(test_loader):
        raise RuntimeError(
            "ran {} batches of {} -- the split was truncated (FR6.4)".format(
                n_batches, len(test_loader)))

    cur_metrics, cur_metrics_std, score_details = comprehensive_evaluation_by_subject(all_gt_fix_vectors,
                                                                                      all_predict_fix_vectors,
                                                                                      args)

    # Print and log all evaluation metrics to tensorboard.
    logger.info("The metrics for best model performance are: ")
    for metrics_key in cur_metrics.keys():
        for (metric_name, metric_value) in cur_metrics[metrics_key].items():
            logger.info("{metrics_key:10}-{metric_name:15}: {metric_value:.4f}".format
                        (metrics_key=metrics_key, metric_name=metric_name, metric_value=metric_value))

    # ---- FR7.4: prediction.json must cover the scored key set EXACTLY -----------
    with open(args.fix_dir) as fh:
        scored = [r for r in json.load(fh) if r["split"] == "test"]
    if args.max_batches <= 0:
        gt_keys = {(r["name"], int(r["subject"])) for r in scored}
        pred_keys = [(r["name"], int(r["subject"])) for r in predict_results]
        dupes = [k for k, c in collections.Counter(pred_keys).items() if c > 1]
        missing = sorted(gt_keys - set(pred_keys))
        extra = sorted(set(pred_keys) - gt_keys)
        if dupes or missing or extra:
            raise RuntimeError(
                "prediction.json key set != the scored split's (FR7.4): "
                "{} duplicated {}, {} missing {}, {} unexpected {}".format(
                    len(dupes), dupes[:10], len(missing), missing[:10],
                    len(extra), extra[:10]))

    if len(predict_results):
        with open(os.path.join(log_info_folder, "prediction.json"), 'w') as f:
            json.dump(predict_results, f, indent=4)

    SM = scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))
    MM = np.mean(list(cur_metrics["MultiMatch"].values()))
    SED = cur_metrics["VAME"]["SED"]

    # ---- FR9: metrics.json, written unconditionally ----------------------------
    nan_drops = multimatch_nan_drops(score_details, args.subject_num)
    # hm_M == 0 can only happen on an empty loader; dividing then would turn a
    # "nothing ran" into a ZeroDivisionError stack trace instead of a stated absence.
    heatmap_block = None if (store is None or hm_M == 0) else {
        "NSS": hm_sum["NSS"] / hm_M, "CC": hm_sum["CC"] / hm_M,
        "KLD": hm_sum["KLD"] / hm_M, "M": hm_M,
        "denominator": ("NSS/CC/KLD average over valid timesteps (M = {}), not over "
                        "(image, subject) cells ({}) like the scanpath metrics. "
                        "Reported separately; never enters the paper-comparable row "
                        "and never influences SM or MM (D6, FR8.5).".format(
                            hm_M, len(predict_results))),
        "nss_caveat": ("NSS standardises by (x - mean) / (std + 1e-7); on a near-flat "
                       "action map it is O(1) noise. An NSS near zero is NOT 'chance "
                       "level' (FR8.6). CC near 0 is the honest floor."),
    }

    metrics_record = {
        "seed": args.seed,
        "args": {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in vars(args).items()},
        "versions": resolved_versions(),
        "sha256": {
            "fixations": sha256_file(args.fix_dir),
            "user_embedding": sha256_file(args.user_emb_path),
            "checkpoint": sha256_file(os.path.join(checkpoints_dir,
                                                   "checkpoint_best.pth")),
            "embeddings_npy": sha256_file(args.emb_dir),
            # D1/D5 -- the frozen files as they were ON THE MACHINE THAT PRODUCED
            # THESE NUMBERS. The repo test proves the copy; this proves the run.
            "evaluation_py": sha256_file(join(os.path.dirname(os.path.abspath(__file__)),
                                              "utils", "evaluation.py")),
            "scanmatch_py": sha256_file(join(os.path.dirname(os.path.abspath(__file__)),
                                             "utils", "evaltools", "scanmatch.py")),
            "visual_attention_metrics_py": sha256_file(
                join(os.path.dirname(os.path.abspath(__file__)), "utils", "evaltools",
                     "visual_attention_metrics.py")),
        },
        "counts": {
            "n_images": len(test_dataset),
            "n_cells": len(scored),
            "n_subjects": preflight["counts"]["n_subjects"],
            "subject_num": args.subject_num,
            "n_batches_run": n_batches,
            "n_batches_expected": len(test_loader),
            "n_predictions": len(predict_results),
        },
        "headline": {"SM": float(SM), "MM": float(MM), "SED": float(SED)},
        "metrics": {k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in cur_metrics.items()},
        "per_cell_std": None,
        "per_cell_std_reason": ("cur_metrics_std deferred to F6 (TechStack section "
                                "3.5b), which recomputes it from prediction.json on "
                                "CPU. null is written rather than omitted so absence "
                                "cannot be mistaken for zero (FR9.2)."),
        "heatmap": heatmap_block,
        "heatmap_reason": (None if heatmap_block is not None else
                           ("--heatmap_dir not set" if store is None else
                            "no valid timesteps were scored (M = 0)")),
        "multimatch_nan_drops": nan_drops,
        "bridge_counters": bridge_report.get("counters"),
        "preflight": {k: v for k, v in preflight.items() if k != "paths"},
        "notes": reporting_notes(args, nan_drops,
                                 bridge_report.get("min_support_per_subject")),
        "record_kind": ("a RECORD of what the frozen evaluator returned; it "
                        "recomputes nothing. F6 is the re-scorer (FR9.4)."),
    }
    with open(os.path.join(log_info_folder, args.metrics_json), 'w') as f:
        json.dump(metrics_record, f, indent=2, sort_keys=True)

    logger.info(metrics_record["notes"]["retrieval"])
    logger.info(metrics_record["notes"]["cohort"])
    logger.info(metrics_record["notes"]["squashes"])
    logger.info(metrics_record["notes"]["bridge_counters"])
    logger.info(metrics_record["notes"]["checkpoint_provenance"])
    logger.info(metrics_record["notes"]["multimatch_nan_drops"])
    logger.info(metrics_record["notes"]["sed_stde_aliases"])
    if heatmap_block is not None:
        logger.info(heatmap_block["denominator"])
        logger.info("heatmap NSS {:.4f} CC {:.4f} KLD {:.4f} over M = {} timesteps"
                    .format(heatmap_block["NSS"], heatmap_block["CC"],
                            heatmap_block["KLD"], heatmap_block["M"]))

    # FR6.7 -- left exactly as upstream has it; the run script tees stdout.
    print('SM: {}, MM: {}, SED: {}'.format(round(SM, 3), round(MM, 3), round(SED, 3)))

if __name__ == "__main__":
    main()
