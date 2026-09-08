# F1 — Reproduce the COCO-FreeView eval baseline on the cluster

> Spec folder: `spec/2026-09-08-cocofv-baseline-on-cluster/`
> Supersedes the OSIE formulation of Roadmap F1. Read with
> [Mission.md](../constitution/Mission.md) · [TechStack.md](../constitution/TechStack.md) ·
> [Roadmap.md](../constitution/Roadmap.md).

---

## Goal

Prove that the released ISP-SENet checkpoint, the `isp` conda environment, the frozen metric code and
the GPU cluster all work together — **before** our own data enters the picture — by reproducing the
paper's **COCO-FreeView few-shot query-set** numbers end to end on the cluster and recording the exact
command, environment and artefacts that produced them. Until this run exists, every later failure on
our data has at least two candidate causes (our bridge, or the environment); afterwards it has one.
This replaces OSIE as the baseline branch: COCO-FreeView is free-viewing like our data, its stimuli are
photographic scenes rather than the OSIE 800×600 set, and its released `fewshot_user_embedding_10.pt`
covers a 3-subject query set (subjects 0, 1, 2) — the same order of magnitude as the cohort
Roadmap OPEN-5 currently permits on the EVE bundle.

**Constitution amendment.** [Roadmap.md](../constitution/Roadmap.md) §4 currently lists
"The **COCO-FreeView** and **COCO-Search18** branches, and anything task/search-conditioned" as
explicitly out of scope. This spec strikes **COCO-FreeView** from that list; COCO-Search18 and
everything task/search-conditioned stay out of scope. The amendment is recorded in FR16 and must be
applied to `Roadmap.md` as part of this feature.

---

## Scope

### In scope

- Building/verifying the `isp` conda environment on the cluster and recording deviations from the
  [TechStack.md](../constitution/TechStack.md) §1 pin list.
- Wiring the **already-staged** COCO-FreeView data (category-subfoldered stimulus images +
  the authors' free-viewing fixation label JSON) into the paths
  `ISP/COCO_FV/GazeformerISP/src/test.py` actually reads.
- Reconciling the four path mismatches between `preprocess_fixations.py`,
  `preprocess/feature_extractor.py` and `src/test.py` (FR3, FR4, FR5) — **by argument, not by editing
  defaults**, per TechStack §6.2.
- Stage B: producing `image_features/<task>/<name>.pth` and `embeddings.npy`.
- Placing `checkpoint_best.pth` and `fewshot_user_embedding_10.pt` at the paths the branch expects,
  with on-disk case verified.
- A single SLURM `sbatch` script, modelled on
  `From-Noisy-Eye-Tracking-To-Scanpath/train_ms.sh`, that runs the query-set evaluation.
- One query-set run: `--fewshot_subject 0 1 2`, `--subject_num 3`.
- Recording SM / MM / SED / STDE / the full ScanMatch + MultiMatch + retrieval block and comparing
  against the paper's COCO-FreeView row (D6).
- Written documentation of every place the COCO_FV branch's `utils/evaluation.py` diverges from the
  OSIE one (FR10) — documented, **never** edited (D1).

### Out of scope

- **The base-set (seen-subject) run** (`--ex_subject 0 1 2 --user_emb_path .../train_user_embedding.pt`,
  `--subject_num 7`). Deferred; not needed to prove the pipeline.
- **Averaging over multiple `--random_support` seeds.** On the `test` split `select_fewshot_subject()`
  returns before the support-sampling branch (FR9.5), so `--random_support` and `--num_fewshot` have
  **no effect on the query-set score**. A single run is the whole answer. They still appear in the log
  filename, which FR15 pins.
- COCO-Search18, and anything task- or search-conditioned.
- Training, fine-tuning, and the RL / policy-gradient path (D8).
- Generating our own subject embeddings (SE-Net, Detectron2, MSDeformAttn). The released
  `fewshot_user_embedding_10.pt` is used as shipped.
- Any edit to `src/utils/evaluation.py` or `src/utils/evaltools/*` in **any** branch (D1).
- The offline re-scorer — that stays Roadmap F6, and this run's artefacts become its fixture.
- EVE data, the bridge, and the heatmap-metric block.
- Downloading the data (it is already on the cluster) and making anything run on Windows GPUs.

---

## Functional Requirements

### FR1 — Environment

FR1.1 The run executes inside a conda environment created from `ISP/environment.yml`, activated by
name. The environment name is a variable in the run script (`ISP_ENV`, default `isp`).

FR1.2 Before the evaluation command, the script prints the resolved versions of `python`, `torch`,
`numpy`, `scipy`, `scikit-image`, `multimatch-gaze` and `opencv-python` to the SLURM stdout log.

FR1.3 `multimatch-gaze` **must** resolve to exactly `0.1.3`. Any other version is a hard failure: the
script exits non-zero with an explicit message rather than producing incomparable numbers
(TechStack §1, "metric-critical — pin exactly").

FR1.4 Every deviation from the TechStack §1 pin list observed on the cluster is recorded in
`spec/2026-09-08-cocofv-baseline-on-cluster/notes.md` under a heading `## Environment as built`,
with the deviating package, the pinned version, the actual version, and one sentence on whether it can
touch metric arithmetic.

### FR2 — Data layout on the cluster

FR2.1 The stimulus images are addressed through one variable, `FV_IMAGE_ROOT`, pointing at a
directory whose immediate children are COCO category directories, each containing `*.jpg`:

```
$FV_IMAGE_ROOT/
├── bottle/       *.jpg
├── bowl/         *.jpg
├── ...           (one directory per COCO category present in the labels)
```

FR2.2 The fixation labels are addressed through one variable, `FV_FIX_JSON`, pointing at the authors'
COCO-FreeView label file (`coco_fv_fixations_update_duration.json` per
`SE-Net/configs/coco_freeview_useremb.json`, or `coco_freeview_fixations.json` per
`src/preprocess/preprocess_fixations.py` — whichever is the file actually staged; FR3.1 pins which).

FR2.3 No absolute cluster path appears in any `.py` file. All of them live in the run script or are
passed as CLI arguments (TechStack §6.4).

FR2.4 The script asserts that `$FV_IMAGE_ROOT` exists and contains ≥ 1 subdirectory holding ≥ 1
`.jpg`, and that `$FV_FIX_JSON` exists and parses as a JSON list, before doing anything expensive.
Failure exits non-zero with the offending path in the message (D7).

### FR3 — `fixations.json` wiring

FR3.1 `src/preprocess/preprocess_fixations.py` reads a hardcoded relative path
`../../../PHAT/data/coco_freeview_fixations.json` and writes a hardcoded `src/data/FV/fixations.json`,
while `src/test.py --fix_dir` defaults to `src/data/fixations.json`. **The mismatch is resolved by
passing `--fix_dir` explicitly**, never by editing either default.

FR3.2 The canonical fixation file used by the run is produced at
`$FV_WORK/data/FV/fixations.json` and passed as `--fix_dir $FV_WORK/data/FV/fixations.json`.

FR3.3 If the staged label file already satisfies FR3.4, it is copied to that path verbatim and
`preprocess_fixations.py` is **not** run. If it does not, a thin wrapper
(`tools/cocofv_prep/normalize_fixations.py`) applies exactly the three transformations
`preprocess_fixations.py` performs and nothing else:
`split == "val"` → `"validation"`; `task == "potted plant"` → `"potted_plant"`;
`task == "stop sign"` → `"stop_sign"`. The wrapper takes `--in`/`--out` arguments so no path is
hardcoded (FR2.3).

FR3.4 The file passed as `--fix_dir` must be a JSON **list** whose every element has the keys
`name` (str, ends `.jpg`), `subject` (int), `X` (list[float]), `Y` (list[float]), `T` (list[float|int]),
`length` (int), `split` (str), `task` (str). Additional keys are permitted and ignored.

FR3.5 The following must hold on the records with `split == "test"`, and are checked by
`tools/cocofv_prep/check_fixations.py` before the GPU run:

| # | invariant | on failure |
|---|---|---|
| a | `len(X) == len(Y) == len(T) == length` for every record | raise, listing the first 5 offenders |
| b | the set of `subject` values contains `{0, 1, 2}` | raise |
| c | every distinct `(task, name)` pair has **exactly** `subject_num` records once filtered to `--fewshot_subject` | raise, listing offending pairs and their counts |
| d | no `task` value contains the substring `jpg` | raise |
| e | `0 <= X < 512` and `0 <= Y < 320` for every fixation | count and report; do not silently clamp |
| f | every `(task, name)` has a readable `$FV_IMAGE_ROOT/<task>/<name>` | raise, listing missing files |

Invariant (c) is the equal-subject requirement: `COCOSearch_evaluation` groups by `"{task}/{name}"` and
`test.py` slices predictions with `index // args.subject_num`, so an image with the wrong number of
subjects silently misaligns every subsequent image's predictions against the wrong ground truth.
This is the single highest-consequence check in the spec (D4, D7).

### FR4 — Image features (Stage B)

FR4.1 Features are produced by `src/preprocess/feature_extractor.py` invoked with
`--dataset_path $FV_IMAGE_ROOT --output_path $FV_WORK/data/FV`.

FR4.2 `image_data()` iterates `os.listdir(dataset_path)` treating each entry as a **task** directory
and writes `<output_path>/image_features/<task>/<name>.pth`. The resulting layout is therefore
per-category, and `--feat_dir` must point at `$FV_WORK/data/FV/image_features` — **not** at the
`src/data/image_features` default, which has no category level.

FR4.3 Each `.pth` is a `torch.Tensor` of shape `(768, 2048)` = `(24*32, 2048)`, dtype `float32`,
on CPU. Produced from a `(768, 1024)` resize with ImageNet normalisation through the
`maskrcnn_resnet50_fpn(COCO_V1)` backbone body.

FR4.4 `COCOSearch_evaluation.__getitem__` builds the feature path as
`join(feat_dir, img_name.replace('jpg', 'pth'))` where `img_name == "{task}/{name}.jpg"`. Because
`str.replace` is unanchored, a **task or filename containing the substring `jpg` corrupts the path**.
FR3.5(d) rejects that case up front.

FR4.5 Feature extraction covers **at minimum** every `(task, name)` appearing in the `test` split of
`--fix_dir`. Extracting the full image root is permitted; extracting less than the test split is a
hard failure surfaced by FR4.6.

FR4.6 Before the GPU evaluation, `tools/cocofv_prep/check_features.py` asserts that for every
`(task, name)` in the test split, `$FV_WORK/data/FV/image_features/<task>/<name>.pth` exists, loads,
and has shape `(768, 2048)`. Missing or misshapen files raise, listing the first 10 (D7).

FR4.7 The feature cache is written to and read from **BeeGFS**, not `$LOCAL_SCRATCH`. Sizing: one
`.pth` is 768 × 2048 × 4 B ≈ 6.3 MB, so a few thousand stimuli reach tens of GB — the same failure mode
the reference `train_ms.sh` documents ("previous run silently failed to rsync the 22 GB .h5"). Only the
label JSON and the checkpoint are copied to scratch.

### FR5 — Task embeddings

FR5.1 `feature_extractor.text_data()` writes `<output_path>/embeddings.npy`, i.e.
`$FV_WORK/data/FV/embeddings.npy`, while `test.py --emb_dir` defaults to `src/data/embeddings.npy`
and a third file is checked in at `src/embeddings.npy`. Resolved by passing
`--emb_dir $FV_WORK/data/FV/embeddings.npy`.

FR5.2 The file loads as `np.load(path, allow_pickle=True).item()` → `dict[str, np.ndarray]`
containing the key `"free-viewing"` with a `float32` vector of shape `(768,)`
(`args.lm_hidden_dim = 768`). It is generated from the sentence
`"You are allowed to view visual stimuli or scenes without any specific task or instruction."`
through `sentence-transformers/stsb-roberta-base-v2`.

FR5.3 If `sentence-transformers` cannot reach the hub from the compute node, the model is pre-fetched
on the login node into `$HF_HOME` and `HF_HUB_OFFLINE=1` is exported in the run script. The run must
not depend on compute-node internet access.

FR5.4 The checked-in `ISP/COCO_FV/GazeformerISP/src/embeddings.npy` is left untouched. If it satisfies
FR5.2 it may be **copied** to `$FV_WORK/data/FV/embeddings.npy` in place of running `text_data()`; which
of the two was used is recorded in `notes.md`.

### FR6 — Checkpoint and subject embedding placement

FR6.1 `test.py` loads `os.path.join(args.evaluation_dir, "checkpoints", "checkpoint_best.pth")`.
With `--evaluation_dir src/assets/FV-ex-012` (the default) the required on-disk path is
`ISP/COCO_FV/GazeformerISP/src/assets/FV-ex-012/checkpoints/checkpoint_best.pth`.

FR6.2 The source file is `weights/COCO_FV-20260904T121424Z-1-001/COCO_FV/checkpoint_best.pth`. It is
**symlinked or copied**, never moved, and never committed (`.gitignore` excludes `*.pth`).

FR6.3 `gazeformer.__init__` executes `self.subject_embed = torch.load(args.user_emb_path)` — a raw
tensor, not an `nn.Embedding`, indexed as `self.subject_embed[subjects]`. The default
`--user_emb_path ../../../SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt` resolves, from the
`ISP/COCO_FV/GazeformerISP/` working directory, to `SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt`.
Source: `weights/COCO_FV-20260904T121424Z-1-001/COCO_FV/fewshot_user_embedding_10.pt`.

FR6.4 The loaded `subject_embed` tensor must have shape `(S, 384)` with `S >= 3` and dtype `float32`
(`args.subject_feature_dim = 384`). Row `i` must correspond to the `i`-th entry of
`--fewshot_subject`, i.e. to dense subject `i` after `select_fewshot_subject()`'s remap (D4).

FR6.5 **On-disk case must be verified before wiring** (TechStack §3.4): `FV-ex-012` vs `fv-ex-012`,
`fewshot_user_embedding_10.pt` vs `fewshot_subject_embedding.pt` (the `SE-Net/README.md` spelling).
The cluster filesystem is case-sensitive. The verified names go in `notes.md`.

FR6.6 `model.load_state_dict(test_checkpoint[key])` is called for every non-`optimizer` key of the
checkpoint, with default `strict=True`. If the checkpoint carries a `subject_embed.weight` entry
(the commented-out `nn.Embedding` path in `gazeformer.py:100`) this raises. If it does, the failure and
its resolution are recorded in `notes.md`; the resolution must not be an edit to `evaluation.py` or
`evaltools/*`.

### FR7 — The evaluation invocation

FR7.1 The command is executed with the **current working directory set to
`ISP/COCO_FV/GazeformerISP/`** — every relative default in `test.py` is anchored there
(TechStack §1, "Command conventions").

FR7.2 The exact invocation:

```
CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject 0 1 2 \
  --subject_num 3 \
  --fix_dir      "$FV_WORK/data/FV/fixations.json" \
  --feat_dir     "$FV_WORK/data/FV/image_features" \
  --emb_dir      "$FV_WORK/data/FV/embeddings.npy" \
  --img_dir      "$FV_IMAGE_ROOT" \
  --evaluation_dir src/assets/FV-ex-012 \
  --user_emb_path ../../../SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt \
  --seed 0 \
  --eval_repeat_num 1
```

FR7.3 `--img_dir` is passed for the log record only. `COCOSearch_evaluation` stores `stimuli_dir` but
its `__getitem__` loads **features only** — the `Image.open` path is commented out. Nothing in the eval
path reads a raw image. This must be stated in `notes.md` so nobody debugs a stimulus path that is
never touched.

FR7.4 Values that are **not** overridden, and must be verified as the effective defaults rather than
assumed: `--width 512`, `--height 320`, `--im_h 24`, `--im_w 32`, `--max_length 20`, `--min_length 1`,
`--batch 1`, `--action_map_num 4`, `--subject_feature_dim 384`, `--lm_hidden_dim 768`.

FR7.5 `--eval_repeat_num 1` is the reported configuration. Inference is stochastic
(`Sampling.random_sample`), so the seed (`--seed 0`) is part of the reported result (D5).

### FR8 — Cluster run script

FR8.1 A single script `bash/test_cocofv.sh` (new, at the repo root under `bash/`) is submittable with
`sbatch bash/test_cocofv.sh` and needs no arguments for the canonical run.

FR8.2 It carries a SLURM header mirroring `train_ms.sh`: `--job-name`, `--output=logs/cocofv_out_%j.log`,
`--error=logs/cocofv_err_%j.log`, `--cpus-per-task=4`, `--mem=16G`, `--time`, `--gres=gpu:1`,
`--mail-type=BEGIN,END,FAIL`, `--mail-user`.

FR8.3 It reproduces the reference script's cluster idioms in order:
timestamp echo → `$SLURM_NODELIST` echo → `cd` to BeeGFS home and capture `HOME_DIR` →
`sudo mount_image.py my_env.ext4 --rw` → stage small inputs to `$LOCAL_SCRATCH` →
`source .../miniconda3/etc/profile.d/conda.sh` → `conda activate "$ISP_ENV"` → `cd` to the project →
run → closing timestamp echo.

FR8.4 All tunables are shell variables in one block at the top of the script, each with a default:
`HOME_DIR`, `PROJECT_DIR`, `FV_IMAGE_ROOT`, `FV_FIX_JSON`, `FV_WORK`, `ISP_ENV`, `FEWSHOT_SUBJECTS`,
`SUBJECT_NUM`, `SEED`. Each is overridable from the environment
(`FV_WORK="${FV_WORK:-...}"`), so a variant run needs no edit to the script body.

FR8.5 `set -euo pipefail` is in effect, and the script exits non-zero on any failed precondition
(FR1.3, FR2.4, FR3.5, FR4.6). A partial run that produces a log with fewer images than expected must
**not** be reported as a result (D7).

FR8.6 Stage B (feature extraction) is guarded: it runs only when
`$FV_WORK/data/FV/image_features` is missing or `check_features.py` fails, controlled by a
`FORCE_FEATURES=0|1` variable. Re-running the evaluation must not re-extract tens of GB of features.

FR8.7 The script does **not** rsync `image_features/` into `$LOCAL_SCRATCH` (FR4.7). This is called out
in a comment, with the reason, exactly as `train_ms.sh` does for its 22 GB `.h5`.

### FR9 — Metric configuration and loader semantics to verify, not assume

FR9.1 `COCOSearch_evaluation` is constructed with `resize=(args.height, args.width) = (320, 512)` and
**`origin_size` left at its class default `(320, 512)`**, because `test.py` does not pass it. The
resulting `resizescale_x = resizescale_y = 1.0`. This is correct for COCO-FreeView, whose labels are
natively in 512×320 — unlike the EVE case (D3), no rescaling occurs and none should.

FR9.2 The evaluator is therefore fed coordinates in 512×320 and durations in **seconds**
(`__getitem__` divides `T` by 1000.0), matching TechStack §4's contract with `args.width=512`,
`args.height=320`.

FR9.3 ScanMatch is constructed as `ScanMatch(Xres=512, Yres=320, Xbin=16, Ybin=12, Offset=(0,0),
TempBin=50, Threshold=3.5)` — the OSIE binning applied to a 512×320 screen, giving 32 × 26.67 px bins.
MultiMatch receives `screensize=[512, 320]`.

FR9.4 `args.max_length = 20` for COCO_FV (OSIE uses 16). Ground-truth scanpaths are **not** truncated
in `COCOSearch_evaluation.__getitem__` — it iterates `range(fixation["length"])` in full. Only the
generated scanpaths are capped at 20 by `Sampling`. Asymmetric lengths are expected and are not a bug.

FR9.5 On the `test` split, `select_fewshot_subject()` filters to `--fewshot_subject`, remaps those ids
to dense `0..N-1` in **argument order**, and returns immediately — the support-sampling and
`sample_{k}.json` writing branches are `split == 'train'` only. Consequences: `--num_fewshot` and
`--random_support` do not affect the score, and `log_dir=None` being passed is harmless.

FR9.6 `test.py` for COCO_FV contains **no `if i_batch > 100: break`** cap — unlike the OSIE branch
(TechStack §5). The full test split is evaluated. This must be confirmed by comparing the number of
images in the test split against `len(test_loader)` in the log.

### FR10 — The COCO_FV `evaluation.py` divergences (frozen, documented)

`ISP/COCO_FV/GazeformerISP/src/utils/evaluation.py` is **not** byte-identical to the OSIE one. It is
frozen under D1 exactly as it stands; the differences are recorded here so the numbers are read
correctly, and **no** attempt is made to reconcile the two files.

| # | divergence | consequence |
|---|---|---|
| FR10.1 | No `if len(vector) >= 3` guard around MultiMatch / SED / STDE. Padding to length 3 still occurs, then all metrics run unconditionally. | Every cell is scored; the OSIE branch leaves some at `-1`. |
| FR10.2 | No NaN → 1 coercion when the two quantised ScanMatch sequences are identical. | A NaN ScanMatch score stays NaN instead of becoming 1. |
| FR10.3 | Diagonal arrays are filtered with `!= -1` (and `(...==-1).sum(-1)==0` for MultiMatch) before the mean; `SED`/`STDE` likewise. | Uninitialised cells are dropped rather than polluting the mean. The dropped count is not printed — FR14.2 requires it be derived. |
| FR10.4 | `p2g()` computes **`r3 = ... rank < 2`**, not `rank < 3`. | The value logged as `R@3` is in fact Recall@2. It must be labelled that way in every write-up. Not a defect to fix (D1). |
| FR10.5 | `p2g()` skips rows whose scores are all `-1` (`ranks = -1`, filtered out afterwards). | The retrieval denominator is the number of scored rows, not `len(gt) * subject_num`. |
| FR10.6 | `human_evaluation_by_subject()` uses `stimulus = zeros((320, 512, 3))` but `screensize=[320, 240]` — internally inconsistent. | Irrelevant: the function is commented out in `test.py` and is not called. Do not call it. |

FR10.7 `src/utils/evaltools/scanmatch.py` is byte-identical between the OSIE and COCO_FV branches
(verified). `visual_attention_metrics.py` must be diffed likewise and the result recorded.

FR10.8 `git status --porcelain ISP/COCO_FV/GazeformerISP/src/utils/` must be empty at the end of the
feature. A non-empty result is a D1 violation and blocks the result from being reported.

### FR11 — Subject identity (D4)

FR11.1 `--fewshot_subject 0 1 2` maps original subject `0 → 0`, `1 → 1`, `2 → 2`. The identity map is
a coincidence of this particular query set, not a general property; the run must still record it
explicitly rather than rely on it.

FR11.2 The COCO_FV branch has **no `utils/data_postprocess.py`** and therefore no
`recover_subject_ids()` / `get_prediction_list()`. `test.py` writes `subject_idx = index %
args.subject_num` — the **dense** index — directly into `prediction.json`.

FR11.3 Because of FR11.1 the dense index equals the real COCO-FreeView subject id for this run, so
`prediction.json` is correct as written and **no code change is made**. `notes.md` must state that
this holds only because `--fewshot_subject` is `0 1 2` in ascending order, and that any other query set
would require the inverse map. This is the D4 gap the EVE branch (F5) must close for itself.

FR11.4 The evaluator's row index is the predicted subject and the column index the ground-truth
subject, with the diagonal reported (D4). `test.py` builds `image_prediction_dict[idx]` by slicing
`[idx * subject_num : (idx+1) * subject_num]`, which preserves the per-image subject order produced by
`COCOSearch_evaluation.__getitem__`. FR3.5(c) is what makes that slicing valid.

### FR12 — `prediction.json` schema (differs from TechStack §3.5)

FR12.1 `test.py` writes `result/FV-ex-012/log/prediction.json` — a JSON list of
`{"name", "task", "subject", "X", "Y", "T", "length"}`.

FR12.2 Divergences from the OSIE schema documented in TechStack §3.5, which the F6 re-scorer must
handle: `name` here is the **bare filename** while the loader keys on `"{task}/{name}"`; `task` is
present; `length` is present; and `T` is `fix_vector_array[:, 2] * 1000` as a **float**, not
`int(round(t * 1000, 3))`. `X`/`Y` are floats, not cast to `int`.

FR12.3 `X`, `Y`, `T` are `numpy.float32` scalars at the point of assignment; `json.dump` must succeed.
If it raises `TypeError: Object of type float32 is not JSON serializable` on the cluster's numpy, the
failure is recorded and worked around in the run script (post-hoc conversion), **not** by editing
`evaluation.py` or `evaltools/*`.

FR12.4 `prediction.json` must be re-scorable offline: `(name, task, subject)` must uniquely identify a
row, and the number of rows must equal `len(test_images) * subject_num * eval_repeat_num`.

### FR13 — Reported numbers (D6)

FR13.1 The following are extracted from `result/FV-ex-012/log/log_test_subject_10_0.txt` into a table
in `notes.md`:

- `MultiMatch`: vector, direction, length, position, duration (5 values) and their mean `MM`.
- `ScanMatch`: with-duration, without-duration, and `SM = scipy.stats.hmean` of the two.
- `VAME`: `SED`, `STDE`.
- Retrieval on the ScanMatch-with-duration matrix: `MRR`, `R@1`, **`R@2` (logged as `R@3` — FR10.4)**,
  `R@5`.

FR13.2 Each is reported against the paper's COCO-FreeView row, with the absolute difference. The run
is a **success** when `SM`, `MM` and `SED` each land within sampling noise of the published values;
"sampling noise" is quantified by FR13.3.

FR13.3 To quantify it, the query-set run is repeated at `--seed 0 1 2` (three runs, otherwise
identical). The spread across seeds is the noise band. This is the **only** repetition in scope; it
varies the sampling RNG, not the support set (FR9.5).

FR13.4 If a metric falls outside the band, the feature is **not** done. The discrepancy is
investigated and recorded before anything downstream proceeds — that is the entire point of a baseline.

### FR14 — Fail loudly (D7)

FR14.1 Preconditions FR2.4, FR3.5, FR4.6, FR6.4 all run **before** the GPU work and exit non-zero on
failure with the offending items enumerated.

FR14.2 After the run, `notes.md` records: the number of test images, the number of `(image, subject)`
cells, the number of cells dropped by FR10.3's `-1` filtering, the number of rows dropped by
`is_eliminating_nan=True`, and the number of retrieval rows skipped by FR10.5. Where the frozen code
does not print a count, it is derived from `score_details` or from the artefacts — never estimated.

FR14.3 A count of ground-truth scanpaths shorter than 3 fixations in the test split is reported,
since those are padded with `(1., 1., 1e-3)` inside the frozen evaluator and the padded array then
replaces the original for all subsequent metrics in that cell (TechStack §4).

### FR15 — Artefacts

FR15.1 The run produces, relative to `ISP/COCO_FV/GazeformerISP/`:

| path | content |
|---|---|
| `result/FV-ex-012/log/log_test_subject_10_0.txt` | full resolved arg namespace + every metric (D5). Filename is `log_test_subject_{num_fewshot}_{random_support}.txt` — `10_0` for the defaults, despite neither affecting the score (FR9.5). |
| `result/FV-ex-012/log/prediction.json` | FR12. |
| `logs/cocofv_out_%j.log`, `logs/cocofv_err_%j.log` | SLURM stdout/stderr, incl. FR1.2's version dump. |

FR15.2 Each of the three seeds overwrites the same log filename. The run script copies each into
`result/FV-ex-012/log/seed{SEED}/` immediately after the run so FR13.3's comparison is possible.

FR15.3 None of these are committed (`.gitignore` excludes `*.pth`, `*.pt`, `*.h5`; `result/` content is
run output). Only `spec/`, the run script, and `tools/cocofv_prep/` are committed (TechStack §6.5).

FR15.4 Any `.pyc` file this feature's runs drop into upstream directories is deleted before commit
(TechStack §6.6); the 89 already-tracked `__pycache__` directories are left alone.

### FR16 — Constitution amendment

FR16.1 `spec/constitution/Roadmap.md` §4 loses "COCO-FreeView" from the out-of-scope bullet. The bullet
becomes: "✗ The **COCO-Search18** branch, and anything task/search-conditioned."

FR16.2 Roadmap §1's F1 row is retitled "Reproduce the **COCO-FreeView** eval baseline on the cluster"
and its §3 body replaced by a pointer to this spec folder, with the OSIE checklist preserved as a
struck-through or clearly-marked superseded note (the OSIE run is dropped, not deferred).

FR16.3 Roadmap §2's dependency graph and §6's external-dependency table are updated: OSIE stimulus
images are no longer needed for F1; COCO-FreeView images + labels are (already staged).

FR16.4 `TechStack.md` §2's repository map annotation `# ◀ FREE-VIEWING BRANCH — our base to adapt` on
`ISP/OSIE/GazeformerISP/` is left as-is: **the EVE branch (F5) still mirrors OSIE**, because OSIE
carries `data_postprocess.py`, a checked-in example `fixations.json`, and a shipped `prediction.json`
that COCO_FV lacks. A sentence is added stating that F1's baseline branch and F5's template branch are
deliberately different, and why.

FR16.5 `TechStack.md` gains a short subsection recording FR10's divergence table, so the next reader
does not assume `evaluation.py` is one file replicated across branches.

---

## Public API Summary

No new runtime classes. The feature's interface is one SLURM script, three preflight tools, and one
documented invocation.

```bash
# bash/test_cocofv.sh — submitted with: sbatch bash/test_cocofv.sh
# All variables overridable from the environment.
HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
FV_IMAGE_ROOT="${FV_IMAGE_ROOT:-$PROJECT_DIR/data/COCO_FV}"
FV_FIX_JSON="${FV_FIX_JSON:-$PROJECT_DIR/data/coco_fv_fixations_update_duration.json}"
FV_WORK="${FV_WORK:-$PROJECT_DIR/work/cocofv}"      # BeeGFS: features live here (FR4.7)
ISP_ENV="${ISP_ENV:-isp}"
FEWSHOT_SUBJECTS="${FEWSHOT_SUBJECTS:-0 1 2}"
SUBJECT_NUM="${SUBJECT_NUM:-3}"
SEED="${SEED:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
```

```python
# tools/cocofv_prep/normalize_fixations.py            (FR3.3)
def normalize(records: list[dict]) -> list[dict]: ...
# CLI: py tools/cocofv_prep/normalize_fixations.py --in PATH --out PATH

# tools/cocofv_prep/check_fixations.py                (FR3.4, FR3.5)
def check_fixations(
    fix_path: str,
    image_root: str,
    fewshot_subjects: list[int],
    split: str = "test",
) -> dict:                      # counters; raises CocoFvPreflightError on any hard invariant
    ...
# CLI: py tools/cocofv_prep/check_fixations.py --fix PATH --images DIR \
#          --fewshot-subject 0 1 2 [--split test]

# tools/cocofv_prep/check_features.py                 (FR4.6)
def check_features(
    fix_path: str,
    feat_dir: str,
    split: str = "test",
    expected_shape: tuple[int, int] = (768, 2048),
) -> dict:                      # {"n_checked": int, "missing": list[str], "bad_shape": list[str]}
    ...
# CLI: py tools/cocofv_prep/check_features.py --fix PATH --feat-dir DIR

class CocoFvPreflightError(RuntimeError): ...
```

```bash
# The evaluation invocation, from ISP/COCO_FV/GazeformerISP/  (FR7.2)
CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject 0 1 2 --subject_num 3 --seed 0 --eval_repeat_num 1 \
  --fix_dir  "$FV_WORK/data/FV/fixations.json" \
  --feat_dir "$FV_WORK/data/FV/image_features" \
  --emb_dir  "$FV_WORK/data/FV/embeddings.npy" \
  --img_dir  "$FV_IMAGE_ROOT" \
  --evaluation_dir src/assets/FV-ex-012 \
  --user_emb_path ../../../SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt
```

---

## Dependencies

| direction | item | contract |
|---|---|---|
| reads | `$FV_IMAGE_ROOT/<task>/*.jpg` | staged COCO-FreeView stimuli, category subdirectories (FR2.1) |
| reads | `$FV_FIX_JSON` | authors' COCO-FreeView fixation labels (FR2.2, FR3.4) |
| reads | `weights/COCO_FV-.../COCO_FV/checkpoint_best.pth` | ISP checkpoint (FR6.1–6.2) |
| reads | `weights/COCO_FV-.../COCO_FV/fewshot_user_embedding_10.pt` | `(S, 384)` float32 subject embeddings (FR6.3–6.4) |
| reads | `ISP/environment.yml` | `isp` conda env (FR1.1) |
| reads | `sentence-transformers/stsb-roberta-base-v2` | task embedding (FR5.2), pre-fetched (FR5.3) |
| reads | `MaskRCNN_ResNet50_FPN_Weights.COCO_V1` | feature backbone (FR4.3), pre-fetched |
| reads (frozen) | `ISP/COCO_FV/.../src/utils/evaluation.py`, `evaltools/*` | D1 — read, never edit (FR10.8) |
| writes | `$FV_WORK/data/FV/fixations.json` | FR3.2 |
| writes | `$FV_WORK/data/FV/image_features/<task>/<name>.pth` | `(768, 2048)` float32 (FR4.2–4.3) |
| writes | `$FV_WORK/data/FV/embeddings.npy` | `{"free-viewing": (768,)}` (FR5.2) |
| writes | `ISP/COCO_FV/.../src/assets/FV-ex-012/checkpoints/checkpoint_best.pth` | symlink/copy (FR6.2) |
| writes | `SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt` | symlink/copy (FR6.3) |
| writes | `ISP/COCO_FV/.../result/FV-ex-012/log/{log_test_*.txt, prediction.json}` | FR15.1 |
| writes | `bash/test_cocofv.sh`, `tools/cocofv_prep/*.py` | committed (FR15.3) |
| writes | `spec/2026-09-08-cocofv-baseline-on-cluster/notes.md` | FR1.4, FR13, FR14.2 |
| writes | `spec/constitution/{Roadmap,TechStack}.md` | FR16 |
| feeds | Roadmap **F6** (offline re-scorer) | this run's `prediction.json` + `fixations.json` are F6's fixture; F6 must handle FR12.2's schema divergences |
| feeds | Roadmap **F7** | FR13's table is the "ours vs paper" reference row |
