# Tech Stack

> Constitution file 2 of 3. Read together with [Mission.md](Mission.md) and [Roadmap.md](Roadmap.md).
> Last updated: 2026-09-07

---

## 1. Execution environment

| | |
|---|---|
| **Target runtime** | Linux GPU server / cluster, NVIDIA CUDA |
| **Dev machine** | Windows 11 (this checkout) — **editing and artefact inspection only** |
| **Never run on Windows** | anything importing `torch.cuda`, MSDeformAttn, or Detectron2 |
| **Runnable on Windows/CPU** | the data bridge (Stage A), JSON schema validation, and offline re-scoring of `prediction.json` (pure numpy/scipy) |

The repo requires **two separate conda environments** — they are not compatible and must not be merged:

### `isp` — ISP / Gazeformer (inference, Stage D + E)
Created from `ISP/environment.yml`. CUDA 11.6 toolchain pinned into the env.

### `senet` — SE-Net (subject embeddings, Stage C)
Created from `SE-Net/environment.yml`.

| package | pin | notes |
|---|---|---|
| python | 3.8.0 | do not upgrade; `scikit-learn==0.22.2` will not build on 3.10+ |
| pytorch | 1.11.0 (cu113) | + torchvision 0.12.0, torchaudio 0.11.0 |
| numpy | 1.23.5 | structured-array dtype behaviour matters to the metrics |
| scipy | 1.10.0 | `scipy.stats.hmean` for the SM headline number |
| scikit-image | 0.19.2 | |
| **multimatch-gaze** | **0.1.3** | **metric-critical — pin exactly** |
| timm | 0.6.13 | SE-Net Swin backbone |
| sentence-transformers | 2.2.2 | task-text embeddings (`stsb-roberta-base-v2`) |
| opencv-python | 4.7.0.72 | |
| deepgaze-pytorch | 0.2.0 | |

Two extra install steps for `senet` only, both of which need a working `nvcc` and matching GCC:

1. **Detectron2** — installed from source, per the HAT repo instructions.
2. **MSDeformAttn** — `cd SE-Net/src/pixel_decoder/ops && sh make.sh`. On a GCC version error:
   `conda install -c conda-forge gxx=9`.

> These two steps are the single biggest install risk in the project. In eval-only mode they are needed
> **only if we generate our own subject embeddings** (Stage C). If we reuse a released
> `*_user_embedding.pt`, the `isp` env alone is sufficient. See [Roadmap.md](Roadmap.md) F3.

### Command conventions
- On the cluster: `CUDA_VISIBLE_DEVICES=0 python src/...`, always invoked **from the
  `ISP/<DATASET>/GazeformerISP/` directory** — every default path in `opts.py` / `test.py` is relative to
  it (`src/data/...`, `src/assets/...`, `../../../SE-Net/...`).
- On Windows, for the CPU-only bridge/validation work: `py -3.8 ...` (the `py` launcher), never `python`.

---

## 2. Repository map

```
few-shot-scanpath/
├── SE-Net/                         # Stage C — Subject-Embedding Network (built on HAT)
│   ├── train.py                    #   entry point; --eval-only emits subject embeddings
│   ├── configs/*.json              #   per-dataset hparams (osie_useremb.json, ...)
│   ├── common/{data,dataset,metrics,losses}.py
│   └── src/{builder,models}.py, backbone/swin.py, pixel_decoder/ (MSDeformAttn)
│
├── ISP/                            # Stages B, D, E — personalized scanpath prediction
│   ├── environment.yml
│   ├── OSIE/GazeformerISP/         # ◀ FREE-VIEWING BRANCH — our base to adapt
│   │   ├── bash/train.sh
│   │   └── src/
│   │       ├── test.py             #   Stage D+E entry point (eval-only)
│   │       ├── train.py, opts.py
│   │       ├── preprocess/
│   │       │   ├── feature_extractor.py   # Stage B: images → .pth, tasks → embeddings.npy
│   │       │   └── preprocess_fixations.py
│   │       ├── dataset/dataset.py  #   OSIE, OSIE_rl, OSIE_evaluation + subject-selection helpers
│   │       ├── models/             #   gazeformer.py, models.py (Transformer), sampling.py, loss.py
│   │       ├── data/fixations.json #   ◀ reference example of the canonical schema
│   │       └── utils/
│   │           ├── evaluation.py           # ▲ FROZEN (D1)
│   │           ├── evaltools/scanmatch.py  # ▲ FROZEN (D1)
│   │           ├── evaltools/visual_attention_metrics.py  # ▲ FROZEN (D1)
│   │           ├── data_postprocess.py     #   subject-id recovery, prediction.json writer
│   │           ├── logger.py, checkpointing.py, recording.py, config.py
│   ├── COCO_FV/GazeformerISP/      # free-viewing on COCO images (category subfolders)
│   └── COCO_Search18/GazeformerISP/# target-present visual search (per-trial task labels)
│
├── weights/                        # released checkpoints (git-ignored: *.pt, *.pth)
│   ├── OSIE-.../OSIE/{checkpoint_best.pth, ckp_11999.pt,
│   │                  train_user_embedding.pt, fewshot_user_embedding_10.pt}
│   ├── COCO_FV-.../COCO_FV/{...,  ckp_27999.pt, ...}
│   └── COCO_Search18-.../COCO_Search18/{..., ckp_27999.pt, ...}
│
└── spec/                           # ◀ our spec-driven-development workspace
    ├── constitution/{Mission,TechStack,Roadmap}.md
    └── YYYY-MM-DD-<slug>/{requirements,plan,validation}.md
```

**We adapt the `ISP/OSIE/GazeformerISP` branch.** Rationale: our dataset is free-viewing; OSIE is the only
free-viewing branch with a single fixed task (`"free-viewing"`), a flat image directory, a checked-in
example `fixations.json`, a `data_postprocess.py`, and a shipped example `prediction.json` to diff against.

---

## 3. Data contracts

Everything below is a hard interface. A spec that changes one of these must say so in its Requirements.

### 3.1 `fixations.json` (canonical scanpath form — Stage A output)
A flat JSON **list**; one object per (image, subject) trial.

```json
{
  "name": "1001.jpg",          // stimulus filename, must exist in img_dir
  "subject": 0,                // int, 0-indexed, dense over the dataset
  "X": [395.5, 390.6, ...],    // float px, ORIGINAL stimulus space, top-left origin, 1-indexed
  "Y": [265.7, 326.4, ...],    // float px, same space; len(Y) == len(X)
  "T": [246, 136, ...],        // int MILLISECONDS, len(T) == len(X)
  "length": 12,                // == len(X)
  "split": "train",            // one of "train" | "validation" | "test"
  "condition": "freeview",     // OSIE constant
  "task": "none"               // OSIE constant; the loader hardcodes task="free-viewing"
}
```

Loader behaviour to be aware of:
- Filtered by `split` at construction: `[_ for _ in fixations if _["split"] == type]`.
- Grouped by `name` into `imgid_to_sub`; `__len__` is the number of **images**, and one batch item yields
  *all subjects* for that image. Every image must therefore have the **same number of subjects**, or the
  `(subject × subject)` score matrix in `evaluation.py` will be ragged.
- Truncated to `args.max_length` (default 16) fixations.

### 3.2 Image features (Stage B output)
- `src/data/image_features/<name>.pth` — one file per stimulus, `<name>` is the image filename with
  `.jpg` → `.pth` (a literal `str.replace('jpg','pth')`, so **avoid "jpg" elsewhere in filenames**).
- Produced by `preprocess/feature_extractor.py`: Mask R-CNN ResNet-50-FPN backbone body
  (`MaskRCNN_ResNet50_FPN_Weights.COCO_V1`), input resized to `(768, 1024)` = `(384*2, 512*2)`,
  ImageNet normalisation, output flattened to `(H*W, 2048)` = `(24*32, 2048)` = `(768, 2048)`.
- `feature_extractor.image_data()` reads from `<dataset_path>/train/` — a hardcoded subdirectory; adapt or
  symlink rather than editing if avoidable.

### 3.3 Task embeddings
- `src/data/embeddings.npy` — a pickled `dict[str, np.ndarray]`, loaded with `allow_pickle=True).item()`.
- Free-viewing needs exactly one key: `"free-viewing"`, dim 768 (`args.lm_hidden_dim`), from
  `sentence-transformers/stsb-roberta-base-v2`.

### 3.4 Checkpoints and embeddings (in `weights/`, git-ignored)
| file | consumer | note |
|---|---|---|
| `checkpoint_best.pth` | `ISP/.../src/test.py` | loaded into `gazeformer`; must live at `<evaluation_dir>/checkpoints/checkpoint_best.pth` |
| `ckp_11999.pt` (OSIE) / `ckp_27999.pt` | `SE-Net/train.py` | SE-Net weights |
| `train_user_embedding.pt` | ISP `--user_emb_path` | embeddings for **seen** (base-set) subjects |
| `fewshot_user_embedding_10.pt` | ISP `--user_emb_path` | embeddings for **unseen** subjects, 10-shot |

Note the case inconsistency between docs and code: READMEs write `assets/osie-ex-10to15/` and
`train_user_embedding.pt`, while `SE-Net/README.md` also writes `train_subject_embedding.pt` and
`ISP/OSIE/.../README.md` writes `OSIE-ex-10to15`. On a case-sensitive Linux filesystem this matters —
verify the actual filenames on disk before wiring paths.

### 3.5 `prediction.json` (Stage D artefact)
Written by `get_prediction_list()` to `result/<evaluation_dir basename>/log/prediction.json`:
`{"name", "subject", "X", "Y", "T"}` with `X`/`Y` cast to `int` and `T` back to **milliseconds**
(`int(round(t * 1000, 3))`). Same schema as `fixations.json` minus `length`/`split`/`condition`/`task`,
which makes it re-scorable offline.

---

## 4. Metric contracts (FROZEN — see D1)

Entry point: `comprehensive_evaluation_by_subject(gt_fix_vectors, predict_fix_vectors, args)` in
`src/utils/evaluation.py`.

**Inputs.** Two nested lists indexed `[image][subject]`, each leaf a numpy **structured array** with
`dtype = {'names': ('start_x','start_y','duration'), 'formats': ('f8','f8','f8')}`, coordinates in the
resized 512×384 space, duration in **seconds**.

**Reads from `args`:** `args.width` (512), `args.height` (384), `args.subject_num`.

| metric | implementation | parameters | computed on |
|---|---|---|---|
| MultiMatch (vector, direction, length, position, duration) | `multimatch_gaze.docomparison` | `screensize=[args.width, args.height]` | diagonal only |
| ScanMatch **with** duration | `evaltools.scanmatch.ScanMatch` | `Xres=512, Yres=384, Xbin=16, Ybin=12, Offset=(0,0), TempBin=50, Threshold=3.5` | **full matrix** (feeds retrieval) |
| ScanMatch **without** duration | `evaltools.scanmatch.ScanMatch` | same minus `TempBin` | diagonal only |
| SED (string edit distance) | `evaltools.visual_attention_metrics.string_edit_distance` | `stimulus = zeros((height, width, 3), float32)` | diagonal only |
| STDE | `...scaled_time_delay_embedding_similarity` | same stimulus | diagonal only |
| retrieval | `p2g(...)` on the ScanMatch-with-duration matrix | `mode="max"` | full matrix |

Non-obvious behaviour that must be preserved:
- Scanpaths shorter than **3** fixations are padded with `(1., 1., 1e-3)` tuples before MultiMatch, and
  the padded array then replaces the original for *all* subsequent metrics in that cell.
- ScanMatch input is built by `np.array([list(_) for _ in list(vec)])` then `[:, -1] *= 1000`
  (seconds → ms) — so the **caller** does the unit conversion, not the metric.
- A NaN ScanMatch score is coerced to `1` when the two quantised sequences are identical.
- `is_eliminating_nan=True` (default) drops MultiMatch rows containing NaN before the mean; SED/STDE means
  do **not** get the same treatment.
- Uninitialised cells are `-1`, not NaN — an all-`-1` result means the loop never ran, not a bad score.

**Headline numbers** (as printed by `test.py`):
`SM = scipy.stats.hmean([scanmatch_wo_dur, scanmatch_w_dur])`, `MM = mean(5 MultiMatch dims)`, `SED`.

---

## 5. Model and inference contract

- `models/models.py::Transformer` + `models/gazeformer.py::gazeformer`, conditioned on a subject embedding
  of dim `args.subject_feature_dim = 384` and a task embedding of dim `args.lm_hidden_dim = 768`.
- Spatial action map `(args.im_h, args.im_w) = (24, 32)`; `args.action_map_num = 4`;
  `max_length = 16`, `min_length = 1`.
- The model emits `all_actions_prob` plus log-normal duration parameters
  (`log_normal_mu`, `log_normal_sigma2`); `models/sampling.py::Sampling` turns these into fixation vectors
  via `random_sample()` → `generate_scanpath()`. Inference is therefore **stochastic** — `--eval_repeat_num`
  controls how many samples per trial.
- `test.py` currently hard-stops at `if i_batch > 100: break`. This caps evaluation at 101 images and
  **must be handled explicitly** in any spec that reports full-dataset numbers.

---

## 6. Working conventions

1. **Frozen files** (D1): `utils/evaluation.py`, `utils/evaltools/*`. Read them; never edit them.
2. **Additive over invasive.** Prefer a new dataset branch / new script over modifying a shared file.
   When a shared file must change, the change must be argument-gated so the OSIE default path is
   bit-identical to before.
3. **New dataset branch layout.** If we create `ISP/<OurDataset>/GazeformerISP/`, mirror the OSIE tree
   exactly — same filenames, same relative paths — so upstream diffs stay readable.
4. **Paths.** Never hardcode an absolute path. Cluster paths go in the run script / CLI args, not in `.py`.
5. **Git hygiene.** `.gitignore` already excludes `*.pt` and `*.pth`. Never commit weights, features,
   stimulus images, or subject-level gaze data. Add `spec/` artefacts, converters, and run scripts.
6. **`__pycache__` directories are checked in upstream.** Ignore them; do not "clean up".
7. **No new heavy dependencies.** The env pins are fragile (python 3.8, torch 1.11). Anything new must be
   pure-python and justified.
8. **Every run is logged.** Use `utils/logger.Logger`; it already dumps the full arg namespace (D5).
