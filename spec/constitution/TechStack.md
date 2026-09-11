# Tech Stack

> Constitution file 2 of 3. Read together with [Mission.md](Mission.md) and [Roadmap.md](Roadmap.md).
> Last updated: 2026-09-10

---

## 1. Execution environment

| | |
|---|---|
| **Target runtime** | Linux GPU server / cluster, NVIDIA CUDA |
| **Dev machine** | Windows 11 (this checkout) — **editing and artefact inspection only** |
| **Never run on Windows** | anything importing `torch.cuda`, MSDeformAttn, or Detectron2 |
| **Runnable on Windows/CPU** | the data bridge (Stage A), JSON schema validation, and offline re-scoring of `prediction.json` (pure numpy/scipy) |

> **Dev-machine reality check (2026-09-08).** The Windows checkout has **Python 3.12 only** — `py -3.8`
> does not resolve, so CPU-side work runs on `py` (3.12) with numpy 2.1.2, scipy 1.14.1, h5py 3.12.1,
> torch 2.7.0+cpu, pandas 2.2.3. That is *not* the pin list below. It is fine for the bridge, whose
> arithmetic is version-agnostic, but anything whose output the cluster later trusts bit-for-bit — in
> particular the ground-truth heatmap cache — must be re-validated under the `isp` env's numpy 1.23.5
> before F5 relies on it, since float32 promotion and structured-array behaviour is exactly what the
> pin exists for.

The repo requires **two separate conda environments** — they are not compatible and must not be merged:

> **Corrected 2026-09-09.** Earlier revisions of this section carried **one** pin table between the two
> environment headings, implying it described both. It does not — those values are
> `SE-Net/environment.yml`'s. The two environments differ on nearly every version, which is precisely
> why §1 says they "must not be merged". The tables are now separate and each is labelled with the file
> it was read from. **F1's FR1 records deviations against the `isp` column**, not the `senet` one.
> The mislabelling was found on 2026-09-09 while diagnosing an `EnvironmentNameNotFound: isp`.

### `isp` — ISP / Gazeformer (inference, Stage D + E) ◀ the env F1 and F5 need
Created from `ISP/environment.yml`. CUDA 11.6 toolchain pinned into the env.

| package | pin (`ISP/environment.yml`) | notes |
|---|---|---|
| python | 3.8.18 | |
| pytorch | 1.13.1 (cu116) | conda, `py3.8_cuda11.6_cudnn8.3.2_0`; + torchaudio 0.13.1 |
| torchvision | **0.14.1**, *not* the file's `0.16.2` | see the warning below — this is a defect in the file |
| numpy | 1.24.3 | structured-array dtype behaviour matters to the metrics |
| scipy | 1.9.3 | `scipy.stats.hmean` for the SM headline number |
| scikit-image | 0.19.3 | |
| **multimatch-gaze** | **0.1.3** | **metric-critical — pin exactly.** Identical in both envs |
| sentence-transformers | 3.0.1 | imported at module scope by `feature_extractor.py` (Stage B) |
| opencv-python | 4.9.0.80 | |
| timm | 0.9.16 | not used on the ISP eval path |

> **`ISP/environment.yml` is a dirty `conda env export`, not a curated spec.** 518 conda packages with
> exact build strings — the full CUDA 11.6 toolkit, the `anaconda` metapackage, Spyder, Jupyter, Scrapy —
> plus 117 pip entries. Two consequences:
>
> 1. **It is internally inconsistent.** Conda pins `pytorch=1.13.1`; pip pins `torchvision==0.16.2`,
>    which requires torch 2.1.2. Creating the env from the file as written lets pip pull torch 2.1.2 over
>    the conda PyTorch. The correct partner for torch 1.13.1 is **torchvision 0.14.1**.
> 2. **Solving it is slow and often unsatisfiable** on a base image other than the authors'. Building a
>    minimal env from the ISP column above is the supported path; record it as an FR1 deviation with the
>    resolved `conda list` / `pip freeze` attached, per D5.
>
> The ISP **eval** path (`src/test.py` and everything it reaches) imports only: `torch`, `torchvision`,
> `numpy`, `scipy`, `skimage`, `cv2`, `multimatch_gaze`, `PIL`, `pandas`, `matplotlib`, `tqdm`.
> Stage B adds `sentence_transformers` — needed at *import* time even though only `image_data()` is
> called, because `feature_extractor.py` imports it at module scope. Nothing else in the 635 entries is
> reached. (`GazeParser` appears in `evaltools/scanmatch.py` but only inside a docstring; the module
> imports `numpy` alone. It is not a dependency.)

### 1.1 What F1 actually ran on — the pin list is softer than it looks ◀ 2026-09-09

**F1's green run did not use `isp` at all.** It used a pre-existing cluster env (`scanpath`) that is
several major versions newer than anything above, and it reproduced the published OSIE numbers anyway.

| | pinned (`ISP/environment.yml`) | F1's actual run | |
|---|---|---|---|
| python | 3.8.18 | **3.11.14** | |
| torch | 1.13.1+cu116 | **2.10.0+cu126** | |
| torchvision | 0.14.1 | **0.25.0+cu126** | |
| numpy | 1.24.3 | **2.1.2** | major version jump |
| scipy | 1.9.3 | **1.14.1** | |
| scikit-image | 0.19.3 | **0.26.0** | corrected 2026-09-09 — see below |
| opencv | 4.9.0.80 | **4.11.0** | corrected 2026-09-09 — see below |
| **multimatch-gaze** | **0.1.3** | **0.1.3** | the one pin that was held |

> **The scikit-image and opencv rows were wrong until 2026-09-09, and the generated run record is what
> caught it.** They previously read `0.19.3 / 4.9.0.80` — the *pinned* values — in the column labelled
> "F1's actual run", with the gloss "installed to satisfy imports". The sweep's `versions.txt`, read
> back by `aggregate_seeds.py`, says **skimage 0.26.0** and **cv2 4.11.0**. The deviation from the pin
> list is therefore larger than this section claimed. Neither library is on the metric path — the
> frozen suite imports `skimage` and `cv2` only in `visual_attention_metrics.py`, and the SED/STDE call
> sites do not reach the functions that use them (see the `scipy.misc.imresize` dead path above) — so
> nothing about the published-row agreement changes. What changes is the record's honesty. This is the
> concrete argument for D5 and for generating the record from artefacts rather than transcribing it: a
> hand-written table asserted versions nobody had checked, and it stayed wrong until a tool read the
> file.

Why this did not break, established by reading and then by testing the frozen code directly under
numpy 2.1.2 on the dev machine before the run:

- **No removed numpy aliases anywhere in the tree** — no `np.float`, `np.int`, `np.bool`, `np.object`,
  `np.bool8`, `np.NaN`, `np.product`. The 1.20→1.24→2.0 removals have nothing to bite on.
- **All five metrics execute and return sane values under numpy 2.1.2 / scipy 1.14.1**: MultiMatch
  `docomparison` (including the pad-to-length-3 path), ScanMatch with and without duration, SED, STDE,
  and `scipy.stats.hmean`.
- **`from scipy.misc import imresize` at `visual_attention_metrics.py:57` is a dead path.** It is inside
  a function the SED/STDE call sites never reach. `scipy.misc.imresize` has been gone since scipy 1.3,
  so this would have broken the *pinned* env too — it is not a modern-scipy problem.
- **`torch.load`'s `weights_only=True` default (torch ≥ 2.6) is harmless here.** `checkpoint_best.pth`
  is `{"model": OrderedDict}` of plain tensors — no argparse `Namespace`, no optimizer state — and the
  `*_user_embedding.pt` files are bare tensors. Nothing needs an allowlist. The call sites in
  `dataset.py`, `gazeformer.py` and `test.py` pass no `weights_only=` and do not need to.

> **What this licenses, and what it does not.** It is now established that the frozen metric suite
> *runs* on a modern stack and lands on the published OSIE row. It is **not** established that it is
> bit-identical to the pinned env — there is no python 3.8 on the dev machine to diff against, and
> "consistent with the published numbers" is a coarser test than bitwise equality. ScanMatch and SED are
> integer/string operations and barely exposed; MultiMatch and STDE are float paths and are the place
> any drift would show. **Every run must therefore record its resolved versions** (D5) so a future
> discrepancy can be attributed. Do not quietly assume two runs on different stacks are comparable
> because F1 was fine.

Practical consequences for F5 and anyone reusing an env:

- `ISP_ENV` in `bash/test_osie.sh` is a tunable precisely so an existing env can be used. Install into
  it with `--no-deps` — an unconditional `pip install` lets pip resolve multimatch-gaze's numpy/scipy/
  pandas requirements and move versions other work depends on. The script does this conditionally and
  then *verifies* the 0.1.3 pin, which is what actually enforces it.
- **`sentence-transformers` need not be installed.** `preprocess/feature_extractor.py` imports
  `SentenceTransformer` at module scope but uses it only at line 63, inside `text_data()` — never called,
  because every branch ships its `embeddings.npy`. `bash/test_osie.sh` registers a stub module in
  `sys.modules` before importing `image_data`, which avoids pulling `transformers` + `tokenizers` (and
  pip's opinion about torch) into a shared env. Call-site only; the upstream file is untouched.
- Building `isp` from `ISP/environment.yml` remains unattempted, and on this evidence unnecessary for
  eval-only work.

### 1.2 `SE-Net/environment.yml`'s pip section is unusable — build `senet` from the conda half ◀ 2026-09-11

**An earlier revision of this section said the `senet` file, unlike `ISP/environment.yml`, "is not
known to be internally inconsistent". That was wrong**, and it was wrong in the way that matters. The
two files share the same pathology — both are dirty `conda env export` dumps — and the check that
missed it compared only the torch *version numbers*, which do agree (conda `pytorch=1.11.0`, pip
`torch==1.11.0`). The pip section's 180 entries are what the authors happened to have installed, and
**at least three of them cannot be satisfied from PyPI at all**:

| entry | why it can never install |
|---|---|
| `deepgaze-pytorch==0.2.0` | not published to PyPI (GitHub-only). **Imported nowhere in this repository** — `grep -rn deepgaze SE-Net/ ISP/ tools/` is empty. |
| `multiscaledeformableattention==1.0` | this **is Step 3's own build output** — `ops/setup.py` declares `name="MultiScaleDeformableAttention", version="1.0"`. It is compiled by `make.sh`, never downloaded. |
| `ace-tools==0.0` | a stub version of a package that is not on PyPI. |

`conda env create -f SE-Net/environment.yml` therefore **fails at the pip stage every time**, reporting
whichever of the three pip resolves first (observed: `deepgaze-pytorch`). Because pip resolves the whole
set before installing anything, **no pip package is installed and the conda half is left intact** — so
the failure is recoverable without recreating the env.

Two further hazards in the same section, which is why the fix is to skip it rather than to prune the
three:

- **`torch==1.11.0` from PyPI is a cu102 build.** Installing it over conda's
  `pytorch=1.11.0=py3.8_cuda11.3_cudnn8.2.0_0` would replace the CUDA build that MSDeformAttn must
  compile against, while leaving the version string looking correct.
- **The `nvidia-*-cu12` wheels and `triton==2.1.0` belong to a torch 2.x / CUDA 12 stack**, not to this
  env's cu113. They are export noise from another environment on the same machine.

**The supported path (FR1.1's "minimal env" branch, recorded per D5):** create from the conda section
alone, then add by hand only what F3's import path actually reaches. That path is
`tools/eve_senet/{dataset,embed}.py` → `SE-Net/common/{config,utils,data}.py` and `SE-Net/src/models.py`,
and it needs exactly:

| package | source | why |
|---|---|---|
| pytorch 1.11.0 cu113, torchvision 0.12.0, numpy 1.23.5, scipy 1.10.0, pillow, scikit-image | **conda section** | already there |
| `opencv-python` | pip, `--no-deps` | `common/utils.py` does `import re, cv2` at module scope |
| `scikit-learn` | pip, `--no-deps` | `common/utils.py` does `from sklearn.model_selection import KFold` at module scope. **Version is unconstrained**: `KFold` is used only by `fold_split()`, which F3 never calls — the import must merely succeed. Do not fight the `0.22.2` pin. |
| `timm` 0.6.13 | pip, `--no-deps` | `src/backbone/swin.py` |
| `detectron2` + `fvcore` | source install (§1 Step 2) | `src/models.py`, both pixel decoders, the transformer decoder |
| `MultiScaleDeformableAttention` | `make.sh` (§1 Step 3) | compiled in place, never installed |

Install the pip four with `--no-deps`, the lesson §1.1 records from F1: an unconditional `pip install`
lets pip resolve their numpy/scipy requirements and move versions the conda solve just pinned.

Everything else in those 180 entries — Jupyter, JupyterLab, mmcv, diffusers, transformers,
sentence-transformers, seaborn, openxlab, the aliyun SDK — is unreached. (`sentence-transformers` is
ISP Stage B's, and even there it is stubbed rather than installed; see §1.1.)

**`check_env.py` probes `cv2` and `sklearn` as of 2026-09-11**, added after this failure. Both are
module-scope imports of `common/utils.py`, so a missing one is an `ImportError` before any F3 code
runs; neither was covered by the original nine probes, and `environment.yml` puts both *only* in the
unusable pip section — precisely the combination that produces a confusing failure three steps later.

### `senet` — SE-Net (subject embeddings, Stage C)
Created from `SE-Net/environment.yml` — but **its pip section cannot be installed; see §1.2**.
OPEN-2 resolved toward generating our own embeddings on 2026-09-10, so this env **is** needed (F3).
`deepgaze-pytorch` appears in the table below for fidelity to the file; it is imported nowhere in
the repository and must not be installed.

| package | pin (`SE-Net/environment.yml`) | notes |
|---|---|---|
| python | 3.8.0 | do not upgrade; `scikit-learn==0.22.2` will not build on 3.10+ |
| pytorch | 1.11.0 (cu113) | + torchvision 0.12.0, torchaudio 0.11.0 |
| numpy | 1.23.5 | |
| scipy | 1.10.0 | |
| scikit-image | 0.19.2 | |
| **multimatch-gaze** | **0.1.3** | same pin as `isp` |
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
- On Windows, for the CPU-only bridge/validation work: `py ...` (the `py` launcher), never `python`.
  Earlier revisions said `py -3.8`; 3.8 is not installed on this machine — see the dev-machine note in §1.
- The bridge CLI, from the repo root:
  `py tools/eve_bridge/build.py --bundle-dir DIR --out-dir DIR [--unseen-subjects ...]
  [--support-pool-size 20] [--seed 0] [--skip-stimuli] [--skip-heatmaps]`
- Its tests: `py -m pytest tests/eve_bridge -q`. The `[bundle]` group is skipped unless
  `--bundle-dir` is passed; `--bridge-subjects a,b` and `--bridge-support-pool-size N` override the
  configuration those checks run under.
- The OSIE baseline run (F1), from the repo root on the cluster. **The primary path is interactive,
  under `salloc`** — chosen for debuggability, and the way the F1 sweep is actually run:
  ```
  salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=08:00:00
  bash bash/test_osie.sh            # SEED=0
  SEED=1 bash bash/test_osie.sh
  SEED=2 bash bash/test_osie.sh
  ```
  **`bash script`, never `source script`:** the script sets `-euo pipefail`, which under `source`
  applies to the calling shell — one failed preflight exits the login shell and drops the allocation.

  `sbatch bash/test_osie.sh` (with `--export=ALL,SEED=1`) still works unchanged; the `#SBATCH` block is
  retained. Nothing in the body depends on which path is used — the directives are inert comments when
  the script is executed directly, `SLURM_NODELIST` has a fallback, and `SEED` is read from the
  environment either way. Every tunable in the top block is overridable from the environment.

  Two consequences of the interactive path, both handled in the script:
  - **There is no `logs/osie_out_<jobid>.log`** — everything lands in the terminal. This is why
    `test.py`'s stdout is tee'd into `log/seed$SEED/stdout.txt`: the headline `SM / MM / SED` is a bare
    `print()` (§3.5a) and scrollback is not an artefact.
  - **The image mount is non-fatal.** Several invocations share one allocation, so `mount_image.py`
    finds the image already mounted from the first run and exits non-zero; under `set -e` that would
    abort before any precondition ran. `conda activate "$ISP_ENV"` is the real check and still fails
    loudly. `MOUNT_IMAGE=0` skips the attempt.

  Re-runs are cheap: Stage B is guarded by `check_features.py`'s exit code, so feature extraction is
  skipped once the cache is complete (`FORCE_FEATURES=1` overrides).

  *(Retargeted 2026-09-09 from `bash/test_cocofv.sh`; note the `py -m pip` → `python -m pip` fix —
  the `py` launcher is Windows-only and does not exist on the cluster.)*
- Its CPU preflight tools, runnable from the repo root on Windows or a login node:
  - `py tools/osie_prep/check_fixations.py --fix PATH --images DIR --fewshot-subject 10 11 12 13 14
    [--split test] [--origin-width 800] [--origin-height 600]`
    — prints the counter dict as JSON on **stdout**, commentary on stderr, so it can be teed.
  - `py tools/osie_prep/check_features.py --fix PATH --feat-dir DIR` — the only torch importer of
    the two; its **exit code** drives the run script's skip-extraction guard.
  - There is no `normalize_fixations.py`: the OSIE branch's shipped
    `ISP/OSIE/GazeformerISP/src/data/fixations.json` is already canonical (§3.1), so there is no
    transform to apply. See §3.7 for the *other* OSIE label file and why it must not be used.
- The seed-sweep aggregator / **run-record generator** (FR13.3), run **once, after all three seeds have
  landed**, **from the repo root** on a login node — stdlib only, no GPU, no torch:
  ```
  python tools/osie_prep/aggregate_seeds.py \
    --log-dir   ISP/OSIE/GazeformerISP/result/OSIE-ex-10to15/log \
    --reference spec/2026-09-08-osie-eval-baseline/paper_reference.json \
    --out       ISP/OSIE/GazeformerISP/result/OSIE-ex-10to15/log/metrics_sweep.json \
    --report    spec/2026-09-08-osie-eval-baseline/run_record.md
  ```
  JSON on **stdout**; the markdown table on stderr unless `--markdown` names a file. `--report` writes
  the generated run record — environment, denominators, the D6 table, the paper comparison. See §3.5b.

  **Mind the working directory.** `result/<eval>/log` is anchored to `ISP/<DATASET>/GazeformerISP/` by
  `test.py`, *not* to the repo root, so a bare `--log-dir result/OSIE-ex-10to15/log` resolves only from
  inside the branch directory — from the repo root it looks like the sweep never ran. Running
  everything from the repo root with the branch prefix (above) keeps all four paths in one frame.
  The tool detects this specific mix-up and names it rather than reporting the sweep as missing.
- Its tests: `py -m pytest tests/osie_prep -q` (74 tests). No fixtures or cluster data needed.
- **The F3 subject-embedding run (Stage C)**, from the repo root on the cluster, in the `senet` env.
  Interactive under `salloc`, like F1's; `sbatch` still works and the `#SBATCH` block is retained:
  ```
  salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00
  bash bash/embed_eve_subjects.sh            # SEED=0
  SEED=1 bash bash/embed_eve_subjects.sh     # a second draw from the same support pools
  ```
  **`bash script`, never `source script`** — same reason as F1's: `-euo pipefail` under `source`
  applies to the login shell. Every tunable is overridable from the environment: `SENET_ENV`, `SEED`,
  `NUM_FEWSHOT`, `BRIDGE_DIR`, `OUT_DIR`, `CKPT`, `CONFIG`, `DATA_ROOT`, `DEVICE`. Artefacts land under
  `$OUT_DIR/seed$SEED/`, so a second seed cannot overwrite the first (§3.5a's lesson, applied up front).

  The script's order is preflight → embed → verify → `.pyc` cleanup, and the first and third of those
  gate on **exit codes**:
  - `py tools/eve_senet/check_env.py [--senet-dir DIR] [--versions PATH]` — probes `torch`,
    `torch.cuda.is_available()`, `torchvision`, `numpy`, `scipy`, `timm`, `detectron2` and the
    MSDeformAttn extension, one `try` each so it runs to completion in a *broken* env (the only env
    where it is interesting). JSON on **stdout**, a table on stderr, non-zero exit on any failure.
    `--versions` writes FR1.5's `versions.txt` **from the same resolution code**, so the recorded stack
    cannot disagree with the checked one — §1.1's lesson made structural rather than restated.
  - `py tools/eve_senet/verify_embedding.py --embedding PATH --subject-map PATH [--report PATH]` —
    CPU-only and CUDA-free, so it runs on a login node or on Windows. Shape / dtype / finiteness /
    no-zero-row / no-duplicate-row, the `to_dense`↔`to_eve` roundtrip for all 38 ids, and with
    `--report` the filename-vs-report seed agreement. Prints `dense_id → eve_id → ||row||₂` for every
    row, which is where D4's "which of my real subjects is row 3?" gets a literal answer. It shares no
    code with `embed.py`'s own assertions on purpose: a checker built from the writer's helpers can
    only confirm the writer was self-consistent.
- `py tools/eve_senet/embed.py --checkpoint PATH [--fixations ...] [--num-fewshot 10] [--seed 0]
  [--device cuda] [--raw-durations] [--allow-missing-key PREFIX]`, if invoked directly. `--device cpu`
  is **refused** unless `--allow-cpu` is also passed, so a silently CPU-bound run cannot be mistaken
  for a normal one. `--raw-durations` is validation Group 5's control arm and nothing else.
- Its tests: `py -m pytest tests/eve_senet -q` (60 tests). The `bridge`-marked ones read the real
  `data/eve_bridge/` artefacts and **skip** when absent — that directory is git-ignored, so a fresh
  checkout has none.

  > **Run each test directory in its own invocation.** `tests/eve_bridge`, `tests/osie_prep` and
  > `tests/eve_senet` each carry a `conftest.py` that its own modules import by bare name
  > (`from conftest import ...`) with no `__init__.py`, so collecting two of them in one pytest run
  > makes the first `conftest` shadow the others and the imports fail. Pre-existing, not new with F3.

---

## 2. Repository map

```
few-shot-scanpath/
├── SE-Net/                         # Stage C — Subject-Embedding Network (built on HAT)
│   ├── train.py                    #   entry point; --eval-only emits subject embeddings
│   ├── configs/*.json              #   per-dataset hparams (osie_useremb.json, ...)
│   │                               #   + eve_useremb.json — OURS, added 2026-09-10 (F3).
│   │                               #   A NEW file; no shipped config was edited
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
├── tools/eve_bridge/               # ◀ Stage A — OUR code. CPU/Windows. Added 2026-09-08 (F2)
│   ├── heatmaps.py                 #   build_step_heatmaps / to_target_scanpath (numpy+scipy)
│   ├── convert.py                  #   build_fixations / export_stimuli (numpy+PIL, imports evedataset)
│   ├── store.py                    #   GtHeatmapStore — the gt_heatmaps.h5 cache (numpy+h5py)
│   ├── validate.py                 #   validate() / BridgeValidationError
│   ├── heatmap_metrics.py          #   score_step_heatmaps() — THE ONLY torch importer in the bridge
│   └── build.py                    #   CLI
│
├── tools/osie_prep/                # ◀ OUR code. CPU preflight for F1. Retargeted 2026-09-09
│   ├── __init__.py                 #   OsiePreflightError
│   ├── check_fixations.py          #   FR3.5 invariants; equal-subject (c) and duration-bin (g)
│   │                               #   above all (stdlib only)
│   ├── check_features.py           #   FR4.6; THE ONLY torch importer here; exit code drives the guard
│   └── aggregate_seeds.py          #   FR13.3 seed-sweep pooling + GENERATED run record
│                                   #   (--report). stdlib only. Added 2026-09-09; see §3.5b
│
├── tools/eve_senet/                # ◀ Stage C — OUR code. CPU dev / GPU run. Added 2026-09-10 (F3)
│   ├── __init__.py                 #   EveSenetError
│   ├── check_env.py                #   FR1.4 preflight AND FR1.5 versions.txt, one source
│   ├── durations.py                #   EVE's own decile duration bins (numpy only, Windows)
│   ├── dataset.py                  #   EveSupportDataset (anchor only) + build_fix_labels
│   ├── embed.py                    #   selection, model, forward pass, pipeline, CLI
│   └── verify_embedding.py         #   FR9 structural verifier (CPU; exit code drives the guard)
│
├── bash/test_osie.sh               # ◀ OUR code. The single sbatch script for F1
├── bash/embed_eve_subjects.sh      # ◀ OUR code. The single run script for F3
│
├── tests/eve_bridge/               # pytest, CPU. `[bundle]`-marked tests need --bundle-dir
├── tests/osie_prep/                # pytest, CPU. Self-contained fixtures, no cluster data
├── tests/eve_senet/               # pytest, CPU. `bridge`-marked tests need data/eve_bridge/
│
├── weights/                        # released checkpoints (git-ignored: *.pt, *.pth)
│   ├── OSIE-.../OSIE/{checkpoint_best.pth, ckp_11999.pt,
│   │                  train_user_embedding.pt, fewshot_user_embedding_10.pt}
│   ├── COCO_FV-.../COCO_FV/{...,  ckp_27999.pt, ...}
│   └── COCO_Search18-.../COCO_Search18/{..., ckp_27999.pt, ...}
│
└── spec/                           # ◀ our spec-driven-development workspace
    ├── constitution/{Mission,TechStack,Roadmap}.md
    └── YYYY-MM-DD-<slug>/{requirements,plan,validation}.md  (+ notes.md when a spec
                                                              lands findings worth keeping;
                                                              F1 instead carries a GENERATED
                                                              run_record.md + a hand-transcribed
                                                              paper_reference.json — §3.5b)
```

**We adapt the `ISP/OSIE/GazeformerISP` branch.** Rationale: our dataset is free-viewing; OSIE is the only
free-viewing branch with a single fixed task (`"free-viewing"`), a flat image directory, a checked-in
example `fixations.json`, a `data_postprocess.py`, and a shipped example `prediction.json` to diff against.

> **F1's baseline branch and F5's template branch are the same branch again, as of 2026-09-09.**
> Both are `ISP/OSIE/GazeformerISP/`. F1 was briefly re-pointed at `ISP/COCO_FV/GazeformerISP/` on the
> argument that its 3-subject query set matched what OPEN-5 permits on EVE; that is withdrawn, because
> the COCO-FreeView **test split is a held-out challenge benchmark** and a baseline nobody outside the
> challenge can reproduce cannot serve as F1's environment proof (Roadmap §4). The collapse is a real
> benefit: F1 and F5 now exercise the same loader, the same `data_postprocess.py`, and the same
> `evaluation.py`, so an F5 discrepancy cannot be blamed on branch drift. The COCO_FV branch stays on
> disk and its `evaluation.py` divergences stay documented in §4.2 — they are still true, and F6/F7
> may need them.

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

### 3.5a Where the metrics actually land (Stage E) ◀ verified on F1's run, 2026-09-09

Under `result/<evaluation_dir basename>/log/` — for F1, `result/OSIE-ex-10to15/log/`:

| artefact | contents |
|---|---|
| `log_test_subject_{num_fewshot}_{random_support}.txt` | the full resolved arg namespace (D5), then every **mean** in `cur_metrics`: MultiMatch's 5 dims, ScanMatch w/ and w/o duration, SED, STDE, and the retrieval block (MRR, R@1/3/5) |
| `prediction.json` | §3.5 schema; for F1, 350 records = 70 images × 5 subjects |

`bash/test_osie.sh` copies both into `log/seed$SEED/` before the next seed overwrites them — the filename
tracks `--num_fewshot`/`--random_support`, neither of which varies across a seed sweep, so without the
copy every seed would land on the same path.

Three properties of this that are easy to get wrong:

1. **The headline `SM / MM / SED` line is a bare `print()`, not `logger.info()`** (`test.py`, last line
   of `main()`). It goes to **stdout** and *not* into `log_test_subject_*.txt`.
   *(Fixed at the call site 2026-09-09: `bash/test_osie.sh` pipes `test.py` through `tee` into
   `log/seed$SEED/stdout.txt`, so the headline is now a per-seed artefact the sweep owns. Before that
   its only home was `logs/osie_out_<jobid>.log`, named by **job id**, not by seed — three seeds meant
   three logs distinguishable only by submission order. The logger's `StreamHandler` writes to stderr,
   so the metric lines and tqdm stay out of the tee'd file. Keep the SLURM logs anyway.)*
2. **`cur_metrics_std` is computed and discarded.** `comprehensive_evaluation_by_subject()` returns
   means *and* standard deviations, but `test.py` logs only the means. **D6 asks for both.**
   *(Decided 2026-09-09: `test.py` stays **unmodified** for F1, as Roadmap F1 already records. D6's
   std is supplied for now by the **across-seed** spread from the sweep, and the per-cell
   `cur_metrics_std` is **deferred to F6**, which recomputes everything from `prediction.json` on CPU
   and can emit both. The two are different quantities — see §3.5b — and must never be presented as
   one.)* Should that decision ever be revisited, the edit is additive and print-only, and note the
   asymmetry when writing it: `cur_metrics_std` is populated for `MultiMatch`, `ScanMatch` and `VAME`
   **only** — `retrieval scanmatch w/ duration` exists in `cur_metrics` alone, so a parallel-structure
   loop `KeyError`s.
3. **The metrics live only in a log file.** There is no machine-readable metrics artefact *from
   `test.py`* — recovering a number means parsing formatted text. `tools/osie_prep/aggregate_seeds.py`
   (§3.5b) now does that parsing once and emits JSON, but it can only report what was logged; it is a
   reader, not a re-scorer. The full argument for **F6** stands: F6 re-derives every metric from
   `prediction.json` + `fixations.json` on CPU. Treat the log files as primary and do not delete them.

### 3.5b The seed sweep and its two different "std" ◀ 2026-09-09

Inference is stochastic (`Sampling.random_sample()`), so one seed is a point, not a band. FR13.3's
sweep is `--seed 0 1 2`; `bash/test_osie.sh` preserves four artefacts per seed under
`result/<eval>/log/seed$SEED/` — `log_test_subject_*.txt`, `prediction.json`, `stdout.txt` (the
headline), and `versions.txt` (the resolved stack, D5, and the thing §1.1 says every run must record).

`tools/osie_prep/aggregate_seeds.py` pools them into `metrics_sweep.json` + `metrics_sweep.md`. It is
stdlib-only, imports no frozen code (D1), and recomputes no metric from scanpaths — it only re-derives
the two published *composites* (`SM` = harmonic mean of the two ScanMatch variants, `MM` = mean of the
five MultiMatch dimensions, §4) from means the authors' evaluator already produced, then cross-checks
them against each seed's printed headline.

> **The two standard deviations are not interchangeable.**
> **Across-seed** (what the sweep reports): spread over **runs**, n = 3 — "how much does this number
> move if I re-sample?" Sample std, ddof = 1, and at n = 3 a noisy estimate, so min/max ride alongside
> it and the range should be reported too.
> **Per-cell** (`cur_metrics_std`, discarded by `test.py`, deferred to F6): spread over
> **(image, subject) cells** within one run — "how much does this vary across the test set?"
> A write-up that labels one as the other is reporting a quantity it did not measure.

Two guards exist because pooling the wrong runs yields a plausible, wrong band — the D7 failure mode:
the `seed` in each log's arg namespace must equal its directory's seed (catching a mis-targeted copy
step), and every argument that changes *what is measured* must be identical across seeds (catching a
sweep that silently mixes two `--subject_num` values or two `--fix_dir` files, including the §3.7
duration-bin trap). Both raise rather than warn.

The tool also annotates three things D6 would otherwise report misleadingly: `pr5` is **structurally
saturated** at `100.0` whenever `subject_num <= 5`; OSIE's `p2g()` computes `r3` as `rank < 3`, so
`pr3` really is R@3 here (the `rank < 2` defect in §4.2 is COCO_FV's, not this branch's); and
`SED_best`/`STDE_best` are aliases rather than a best-of-N (§4).

#### The run record is generated, not written ◀ decided 2026-09-09

There is deliberately **no hand-written `notes.md`** for F1. A file of copy-pasted numbers drifts from
the artefacts the moment anything is re-run, and D5's claim is precisely that every reported number be
re-derivable from artefacts alone. `aggregate_seeds.py --report` therefore generates the whole record
on each invocation, reading all four stored outputs per seed — the metric log, `stdout.txt`,
`versions.txt`, and `preflight_fixations.json` — and emitting the resolved environment, the
denominators (including the D7 soft counters and the pad-to-3 consequence), the D6 table, and the
comparison. Regenerate it; never hand-edit it.

Two guards match the ones on the metric block, for the same reason: pooled seeds must share **one
resolved stack** (§1.1 — otherwise the spread mixes sampling noise with a version change) and **one
preflight fingerprint** (same split, same image/subject counts). Both raise. Absent artefacts are
reported as absent rather than skipped, so a record that does not state its environment says so.

Exactly two inputs are not derivable and are named as such in the output:

| input | why it cannot be generated | where it lives |
|---|---|---|
| the paper's published OSIE row | **nothing in this repo contains it** — `result-images/main-result.png` is qualitative scanpath figures, the READMEs carry no table | `spec/2026-09-08-osie-eval-baseline/paper_reference.json`, transcribed once by hand, with `source` / `transcribed_by` / `transcribed_utc` provenance fields |
| the verdict | what the numbers license, given the checkpoint trained on OSIE subjects 0–9 and scored on 10–14, is an argument rather than a measurement | the spec, and F7 |

In the reference file a `null` means **not transcribed** and renders as `--`; it is never treated as
zero and never as agreement. Do not guess a value to fill a hole — a wrong reference silently converts
a real discrepancy into an apparent match, which is D7's failure mode applied to the comparison itself.
The report's "within seed spread" column is descriptive only: at n = 3 the spread is a noisy estimate,
and what counts as reproducing the row is F7's judgement, not the tool's.

### 3.6 EVE bridge artefacts (Stage A output, added 2026-09-08)

`tools/eve_bridge/build.py` writes five things into `--out-dir`. Downstream features consume these
artefacts, never the bridge's code — the one exception is `score_step_heatmaps()`, which F5 imports.

| artefact | consumer | contract |
|---|---|---|
| `fixations.json` | F4, F5 | §3.1 exactly. EVE specifics: `X`/`Y` in native **1920×1080**, 1-indexed (the bridge adds `+1` once, to `bundle.get_scanpath()`'s 0-indexed pixels); `T` = `round(duration_ms)`, floored at 1; `split ∈ {train, test}` only — never `validation`; `condition="freeview"`, `task="none"`. Records sorted by `(name, subject)`, and **that order is the index space everything else keys against**. |
| `subject_id_map.json` | F5 | `{"to_dense": {"train02": 0, …}, "to_eve": {"0": "train02", …}}`. Dense ids are the index into the *sorted* participant list, so argument order cannot change them. This is the sole authority for D4's "which of my real subjects is row 3?". |
| `stimuli/<name>.jpg` | F4 | Native 1920×1080, `quality=95, subsampling=0`. One per `stimulus_name`; on a rendering collision the first exp_key in sorted order wins and the conflict is *counted*, not resolved (Roadmap OPEN-6). |
| `gt_heatmaps.h5` | F5 | Layout below. |
| `bridge_report.json` | F5, F7 | Resolved args, `origin_size`, `support_pool_size`, `support_per_subject`, `min_support_per_subject`, subject/stimulus/trial counts (incl. `num_trials_train` / `num_trials_test`), `fixations_sha256`, and every drop counter — always present, `0` when nothing fired (D7). |

**The `train` and `test` splits carry different contracts** *(corrected 2026-09-10)*. The §3.1
equal-subject invariant binds the **scored** split only, and even there it demands a uniform
**count**, not a common cohort: `evaluation.py` iterates each image's *actual* subject list and its
diagonal is **positional**, so the subject identities may differ from image to image. Every `test`
name must yield exactly `args.subject_num` records because the collectors are `-1`-initialised and
reduced by a bare `np.mean()` with no `!= -1` filter on this branch — a short image folds `-1` into
every metric. `args.subject_num` does **not** size the subject embedding (`gazeformer.py` L100 is
commented out), so a 38-participant cohort scores correctly at `subject_num = 3`. Only the retrieval
block needs several subjects per image; MultiMatch, ScanMatch-w/o-duration, SED and STDE are each
guarded by `if row_idx == col_idx`. The `train` split is deliberately **ragged** — a support stimulus is
subject-private and carries one record. Nothing enforces otherwise, because
`ISP/OSIE/GazeformerISP/src/test.py` constructs `OSIE_evaluation(..., type="test")` and
**never builds a train-split loader at all** (the subject embedding arrives precomputed via
`--user_emb_path`), and `SE-Net/common/utils.py::select_fewshot_subject()` keeps whichever
subjects have each drawn image (`if subject in img_name_groups[img_name]`).

Two consequences downstream:

- **F5 asserts `num_fewshot <= min_support_per_subject`**, not `<= support_pool_size`. The
  pools share no image names, so the thinnest pool binds.
- **F3 must invoke SE-Net once per subject.** `select_fewshot_subject()` draws `num_fewshot`
  names from the **union** across the fewshot subjects; with disjoint pools one draw of 10
  names gives each subject ≈ 10/N scanpaths, unequally.

**`origin_size` is `(1080, 1920)` as `(H, W)`** and is written into both `bridge_report.json` and the
HDF5 root attrs, precisely so F5 passes it explicitly rather than inheriting the `OSIE_evaluation`
default `(600, 800)`. The resulting `resizescale_x = 3.75`, `resizescale_y = 2.8125` are non-uniform —
the deliberate **squash** (Roadmap OPEN-4).

#### `gt_heatmaps.h5`

EVE ships no saliency maps, so the ground-truth counterpart of the model's `all_actions_prob` is the
per-timestep blurred fixation map that `OSIE.__getitem__` already builds — one-hot placement at
`((pos-1)/downscale).astype(int32)` (truncating, not rounding), then `gaussian_filter(σ=1)`, then
divide by the sum. `tools/eve_bridge/heatmaps.py` re-derives that rather than importing `dataset.py`
(which would drag in torch/torchvision/skimage/matplotlib), and a parity test pins the two **bitwise**
over 200 seeded cases so the copy cannot drift.

```
/                       attrs: created_utc (ISO-8601 Z), bundle_dir,
                               origin_size (2,) int32 = [1080, 1920],
                               action_map  (2,) int32 = [24, 32],
                               max_length  int32 = 16, blur_sigma float32 = 1.0,
                               fixations_sha256, n_trials int32,
                               subject_ids_dense (N_subj,) int32,
                               subject_ids_eve   (N_subj,) vlen utf-8
/trials/trial_key       (N,)             vlen utf-8   "{name}|{dense subject}"
/trials/exp_key         (N,)             vlen utf-8   originating EVE exp_key (D4 provenance)
/trials/subject         (N,)             int32        dense subject id
/trials/length          (N,)             int32        min(len(X), max_length)
/trials/action_mask     (N, 16)          float32
/trials/heatmaps        (N, 16, 24, 32)  float32, gzip=4, chunks=(1, 16, 24, 32)
```

Two invariants worth knowing before touching it:

- **Row `i` is record `i` of `fixations.json`.** The store addresses rows *positionally*, so
  `fixations_sha256` is the whole safety net: `GtHeatmapStore.load(path, fixations_path=…)` raises on
  mismatch. A merely *reordered* JSON changes the hash and is correctly rejected — an order-insensitive
  check would let a mis-indexed cache through.
- **`trial_key` carries the dense subject, `exp_key` rides alongside.** `exp_key_of()` and
  `trial_key_of()` are inverses with no fallback that invents a key, which is what makes any row
  traceable back to the real EVE recording.

---

### 3.7 The two OSIE label files — one is NOT a fixations.json

There are two OSIE label files in this checkout. They have the **same 10,500 records, the same
`(name, subject, split)` key set, the same nine keys, and byte-identical `X`/`Y`**. Only `T` differs,
and only one of them is a `fixations.json` in the §3.1 sense:

| file | `T` | use |
|---|---|---|
| `ISP/OSIE/GazeformerISP/src/data/fixations.json` (6.06 MB) | duration in **ms**, 20 – 1975 (test split: 20 – 1033) | ✅ **canonical.** `--fix_dir`'s default; what F1 evaluates |
| `data/osie_fixations_update_duration.json` (7.75 MB) | **decile bin index, 0 – 9** | ✗ a duration-head training target. Never `--fix_dir` |

Verified 2026-09-09 by joining the two on `(name, subject)`: the bins are ten equal-count buckets of
the true duration — bin 0 = 20–96 ms, bin 1 = 97–125, … bin 9 = 358–1975, ~9,800 fixations each.

**Why this is dangerous rather than merely wrong.** Feeding the binned file to `test.py` raises nothing.
Every §3.1 invariant holds — schema, lengths, equal subjects, coordinate ranges, splits. The damage is
entirely downstream and entirely silent: `evaluation.py` multiplies `duration` by 1000 for ScanMatch,
whose `TempBin=50` then quantises a *bin index* as though it were milliseconds, and MultiMatch's
duration dimension compares bins to bins. `SM` (the harmonic mean of the two ScanMatch variants) and
`MM` (the mean of five dimensions, one of them duration) would both be wrong, plausible, and
uncomparable to the published table — the precise failure D7 exists to prevent.

`tools/osie_prep/check_fixations.py` invariant **(g)** is the guard: it raises when `max(T)` over the
evaluated split is ≤ 20, which no real fixation-duration distribution can be. `bash/test_osie.sh`
defaults `FIX_JSON` to the canonical file, and `data/` is git-ignored (working convention 5).

The name is the trap: `osie_fixations_update_duration.json` reads like the file where durations were
*fixed up*. It is the file where they were *replaced*.

### 3.8 EVE subject-embedding artefacts (Stage C output, added 2026-09-10)

`tools/eve_senet/embed.py` writes three things into `--out-dir` (`data/eve_senet/seed$SEED/` under the
run script). F5 consumes the artefacts, never the code.

| artefact | consumer | contract |
|---|---|---|
| `eve_fewshot_user_embedding_{num_fewshot}_seed{seed}.pt` | **F5's `--user_emb_path`** | `torch.float32` tensor `(38, 384)`, `torch.save`d. Row *i* is dense subject *i*, i.e. `subject_id_map.json`'s `to_eve[str(i)]` — rows are stacked in ascending dense-id order by construction, never by dict iteration order (D4). No row is all-zero, and none is a duplicate of another. |
| `senet_report.json` | F5, F7 | Resolved args, `duration_bin_edges` (11 floats) + `duration_bin_occupancy`, the realised `support_selection` (38 × 10 stimulus names), `per_subject` (`eve_id`, `n_forward`, `row_norm`), `senet_input_size` / `senet_rescale`, `task_emb_source` + key, `missing_keys` / `unexpected_keys` in full, `fixations_sha256`, `checkpoint_sha256`, `embedding_sha256`, and every D7 counter — always present, `0` when nothing fired. |
| `versions.txt` | D5 | The resolved stack, written by `check_env.py`'s own resolution code (§1.1). |

Five properties that matter downstream:

- **The seed is in the filename *and* in the report, and `verify_embedding.py --report` asserts they
  agree.** This is the guard `aggregate_seeds.py` needed for F1 — a mis-targeted copy step pooling
  non-replicates — installed up front rather than after the fact.
- **`embedding_sha256` is the handshake with F5.** F5 asserts it before consuming the tensor, so the
  two features cannot silently be discussing different artefacts. `checkpoint_sha256` likewise pins
  *which* released SE-Net checkpoint produced them, giving OPEN-7's "the released checkpoint may differ
  from the one behind the table" question a concrete handle on the SE-Net side too.
- **`fixations.json` is sha256-gated on the way in**, against `bridge_report["fixations_sha256"]`. A
  merely **reordered** JSON is correctly rejected: F2's artefacts are addressed positionally (working
  convention 10), so an order-insensitive check would let a mis-indexed run through.
- **`senet_rescale = [0.266667, 0.296296]` is a SECOND squash, not the one in OPEN-4.** SE-Net's input
  is 512×**320** (8:5); F5's metric screen is 512×**384** (4:3, `resizescale = 3.75 / 2.8125`). The two
  distortions are independent and **both** must reach F7. The report records the SE-Net one explicitly
  so it cannot be inferred away later.
- **`num_fewshot` is exactly 10 per subject, never "whatever the pool holds".** F2 dropped `train23`
  specifically to raise `min_support_per_subject` from 9 to 10 and buy the paper's n = 10 row; using
  each subject's whole pool would give some participants 20 shots and others 10, leaving the cohort's
  rows unequally informed and the comparison to that row unsound.

#### Two upstream behaviours F3 works around, and one it pins

Recorded because each is a plausible-but-wrong number waiting to happen, and none is a defect to fix
upstream (working convention 2 — additive over invasive; `SE-Net/` is untouched):

1. **`Siamese_Triplet_Gaze.__getitem__` cannot run on a single-subject dataset.** It builds the full
   training triplet regardless of eval mode and draws the negative from a *different* subject, so the
   candidate comprehension is empty and `random.choice` raises `IndexError`. The `num_fewshot == 1`
   short-circuit would dodge it but is gated on the shot count, not on eval mode. F3 must invoke
   SE-Net **per subject** (the union draw in `select_fewshot_subject()` would otherwise hand each
   subject ≈ 10/38 scanpaths from our stimulus-disjoint pools), so it *subclasses* the class and
   returns the anchor alone — which is all `evaluate_user_siamese` ever reads (`batch = batch['anchor']`).
2. **`builder.py` builds the eval loader with `drop_last=True` at `batch_size // 2 = 8`.** A 10-scanpath
   support set would yield one batch of 8 and silently discard 20 % of the evidence. F3 uses
   `drop_last=False` and asserts the forward-passed count equals `num_fewshot` for every subject.
   (`evaluate_user_siamese` has a second defect F3 also avoids: its `num_fewshot == 1` branch saves an
   **un-normalised sum** rather than a mean.)
3. **The duration channel is dead.** `SE-Net/src/models.py` does `ventral_embs += ventral_pos`, then
   computes `duration_encoding`, adds it into `ventral_pos`, and calls `ventral_pos.fill_(0)` on the
   next line; `ventral_pos` is then reused as an accumulator for the indicator embeddings. Nothing
   reads it in between, so the duration never reaches the network. F3 still feeds EVE's own **decile
   bins** — the input contract the released checkpoint was trained under (§3.7), one numpy call, and
   correct if a future variant consumes the channel — and **pins the deadness with a bitwise-identity
   check** (`--raw-durations` is the control arm) rather than assuming it. If that check ever fails,
   the channel has come alive and every embedding produced under the assumption is invalid.

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
- **`SED_best` and `STDE_best` are aliases, not a best-of-N.** `evaluation.py` does
  `SED_best_metrics = SED_metrics_rlts` (and the same for STDE) with no selection step, so
  `SED_best ≡ SED` and `STDE_best ≡ STDE` **identically, in every configuration and at any
  `--eval_repeat_num`**. Confirmed on F1's seed 0: `SED 7.3000 / SED_best 7.3000`,
  `STDE 0.8460 / STDE_best 0.8460`. The name promises a selection the code never performs. They are
  **not** a second, corroborating result and must never be reported as one — D6 names only `SED` and
  `STDE`, and `aggregate_seeds.py` excludes the `_best` pair from its table for this reason (it emits
  a note instead). Frozen under D1: documented, not fixed.

**Headline numbers** (as printed by `test.py`):
`SM = scipy.stats.hmean([scanmatch_wo_dur, scanmatch_w_dur])`, `MM = mean(5 MultiMatch dims)`, `SED`.

### 4.1 Heatmap metrics — NSS / CC / KLD (borrowed, not frozen)

`models/loss.py` is **not** on the D1 frozen list, but it already contains these three functions with
the authors' epsilon handling and normalisation, and the COCO branches' `test.py` imports them for the
same purpose. `tools/eve_bridge/heatmap_metrics.py` therefore loads that file **by path via
`importlib`** and calls into it — never copies, never reimplements. Reimplementing would produce
numbers that are ours, not theirs.

- Argument order is **prediction first, ground truth second** in all three: `NSS(input, fixation)`,
  `CC(input, salmap)`, `KLD(input, salmap)`. KLD is not symmetric, so getting this backwards yields a
  plausible but wrong number; a test pins it with an asymmetry check.
- The wrapper adds exactly one thing: masking of padding steps. Only `(b, t)` with
  `t < min(length[b], 16)` is scored, because padding steps carry an all-zero ground-truth map on
  which CC and NSS are undefined and would silently deflate every number.
- **Denominator asymmetry — must be stated wherever these are reported.** The three functions reduce
  with `.mean()` over dim 0, so the returned value is a mean over valid **timesteps**
  (`M = Σ_b min(length[b], 16)`). The scanpath metrics average over **(image, subject) cells**. They
  are not two views of the same mean.
- **NSS is ill-conditioned on a near-flat prediction.** It standardises by `(x - mean) / (std + 1e-7)`;
  on a constant float32 map the numerator is pure rounding residual (~1e-8) against a bare-epsilon
  denominator, so the result is O(1) noise whose value depends on the constant (−1.79 at 0.37, −0.60 at
  1.0, 0.0 at 2.0). CC correctly returns 0. The meaningful floor is a *random* prediction, where both
  sit at ≈ 0. An NSS near zero on a nearly-uniform action map is therefore not informative — do not
  read it as "chance level".

### 4.2 Per-branch metric drift — `evaluation.py` is NOT one file replicated across branches

The single most expensive wrong assumption available in this repository is that
`src/utils/evaluation.py` is the same file in every branch. It is not.
`ISP/COCO_FV/GazeformerISP/src/utils/evaluation.py` has drifted from the OSIE one in six ways. **All
versions are frozen under D1 regardless** — each produced the published numbers for its own dataset, so
reconciling them would make our numbers incomparable to the table we are trying to reproduce, which
defeats the mission. Documented, never edited.

Verified 2026-09-08: `evaltools/scanmatch.py` and `evaltools/visual_attention_metrics.py` are both
**byte-identical** between the OSIE and COCO_FV branches. Only `evaluation.py` differs.

| # | COCO_FV divergence from OSIE | consequence |
|---|---|---|
| 1 | No `if len(vector) >= 3` guard around MultiMatch / SED / STDE. Padding to length 3 still occurs, then all metrics run unconditionally. | Every cell is scored; the OSIE branch leaves some at `-1`. |
| 2 | No NaN → 1 coercion when the two quantised ScanMatch sequences are identical. | A NaN ScanMatch score stays NaN instead of becoming 1. |
| 3 | Diagonal arrays are filtered with `!= -1` (and `(...==-1).sum(-1)==0` for MultiMatch) before the mean; `SED`/`STDE` likewise. | Uninitialised cells are dropped rather than polluting the mean. The dropped count is **not printed** and must be derived from `score_details`. |
| 4 | `p2g()` computes **`r3 = ... rank < 2`**, not `rank < 3`. | The field logged as `R@3` is in fact **Recall@2**, and must be labelled `R@2` in every write-up. The published COCO-FreeView column came from this same code. Not a defect to fix (D1). |
| 5 | `p2g()` skips rows whose scores are all `-1`. | The retrieval denominator is the number of scored rows, not `len(gt) * subject_num`. |
| 6 | `human_evaluation_by_subject()` uses `stimulus = zeros((320, 512, 3))` but `screensize=[320, 240]` — internally inconsistent. | Irrelevant: the function is commented out in `test.py`. Do not call it. |

Two more COCO_FV-specific contracts, for the same reason:

- **The metric screen is 512×320, not 512×384.** `args.width=512`, `args.height=320`, and
  `COCOSearch_evaluation`'s `origin_size` default is `(320, 512)` — which `test.py` does not override,
  so `resizescale_x == resizescale_y == 1.0` and **no rescaling occurs**. That is correct: the labels
  are natively 512×320. ScanMatch is therefore `Xres=512, Yres=320, Xbin=16, Ybin=12` → 32 × 26.67 px
  bins. Contrast D3's EVE case, where rescaling is mandatory.
- **`args.max_length = 20`** for COCO_FV (OSIE uses 16), and ground-truth scanpaths are **not**
  truncated — `__getitem__` iterates `range(fixation["length"])` in full. Only generated scanpaths are
  capped. The asymmetry is expected and is not a bug.

---

## 5. Model and inference contract

- `models/models.py::Transformer` + `models/gazeformer.py::gazeformer`, conditioned on a subject embedding
  of dim `args.subject_feature_dim = 384` and a task embedding of dim `args.lm_hidden_dim = 768`.
- Spatial action map `(args.im_h, args.im_w) = (24, 32)`; `args.action_map_num = 4`;
  `max_length = 16`, `min_length = 1`.
- **`--eval_repeat_num` is effectively pinned to 1 on the eval path** *(established 2026-09-09)*.
  `test.py` extends each image's prediction list once per repeat, so `len(predict_fix_vector)` becomes
  `eval_repeat_num * subject_num`; but `evaluation.py`'s collectors are shaped
  `(n_images, subject_num)` and `scores_of_each_images` is `(n_images, subject_num, subject_num, 9)`,
  indexed `[index, row_idx, col_idx]`. Any `eval_repeat_num > 1` therefore drives `row_idx` past
  `subject_num` and **raises IndexError**. The argument is real but belongs to another path; do not
  reach for it to explain a discrepancy against the published row, and do not spend GPU time testing
  it. F1 passes `--eval_repeat_num 1` explicitly.
- The model emits `all_actions_prob` plus log-normal duration parameters
  (`log_normal_mu`, `log_normal_sigma2`); `models/sampling.py::Sampling` turns these into fixation vectors
  via `random_sample()` → `generate_scanpath()`. Inference is therefore **stochastic** — `--eval_repeat_num`
  controls how many samples per trial.
- `test.py` hard-stops at `if i_batch > 100: break`. It caps evaluation at 101 **batches** and
  **must be handled explicitly** in any spec that reports full-dataset numbers.
  **It is inert for OSIE's own test split** (settled 2026-09-09): `OSIE_evaluation.__len__` returns
  `len(self.imgid)` — the number of **images**, not records — and one batch item yields *all* subjects
  for its image. The test split is 70 images × 15 subjects = 1050 records, so the loader yields 70
  batches at `--batch 1` and 18 at the default `--batch 4`; the cap never fires. The shipped
  `result/git-osie-useremb-ex-10to15/log/prediction.json` holds 350 records = 70 × 5, independently
  confirming the published number covers the full split. F1 therefore leaves `test.py` unmodified.
  **The cap becomes live in F5**, whose stimulus count is not bounded the same way — remove or
  parameterise it there, and never assume "101 images": the unit is batches, hence images.

---

## 6. Working conventions

1. **Frozen files** (D1): `utils/evaluation.py`, `utils/evaltools/*`. Read them; never edit them.
2. **Additive over invasive.** Prefer a new dataset branch / new script over modifying a shared file.
   When a shared file must change, the change must be argument-gated so the OSIE default path is
   bit-identical to before.
3. **New dataset branch layout.** If we create `ISP/<OurDataset>/GazeformerISP/`, mirror the OSIE tree
   exactly — same filenames, same relative paths — so upstream diffs stay readable.
4. **Paths.** Never hardcode an absolute path. Cluster paths go in the run script / CLI args, not in `.py`.
5. **Git hygiene.** `.gitignore` excludes `*.pt`, `*.pth`, `*.h5`, `*.npy`, `*.json` and
   `__pycache__/`, plus `data/`, `work/` and `logs/`. Never commit weights, features, stimulus images,
   or subject-level gaze data — and note that a bridge `--out-dir` contains all three, so it belongs
   outside the repo or under an ignored path. Add `spec/` artefacts, converters, and run scripts.

   > **The `*.json` rule is negated for `spec/`** (`!spec/**/*.json`), added 2026-09-09. As a blanket
   > rule it silently swallowed `spec/2026-09-08-osie-eval-baseline/paper_reference.json` — the
   > hand-transcribed published row the generated run record depends on. An ignored file does not
   > appear as untracked, so nothing flagged it; it was simply absent on the next checkout, and the
   > aggregator failed on the cluster with a missing `--reference`. When adding a broad ignore, check
   > it against `git status --ignored` for artefacts that are part of the *record* rather than the
   > *data*.
   >
   > **Negated again for `SE-Net/configs/` on 2026-09-10** (`!SE-Net/configs/*.json`), for the same
   > reason and caught the same way. Those files are hparam **source**, not data; the shipped
   > OSIE/COCO ones only survive because `.gitignore` does not untrack. F3's new
   > `eve_useremb.json` would have been invisible — not untracked, simply absent on the next
   > checkout, with the run failing on a missing `--config`. Same failure, second instance:
   > check any broad ignore against `git status --ignored` before trusting it.
   >
   > `*.so`, `**/ops/build/` and `*.egg-info/` were added at the same time for MSDeformAttn's
   > compiler output. `sh SE-Net/src/pixel_decoder/ops/make.sh` builds **in place inside the repo** —
   > the one step in the project that writes into `SE-Net/`, and it must read as a build artefact
   > rather than an edit to a tracked file.
6. **`__pycache__` directories are checked in upstream.** Ignore them; do not "clean up" — the 89
   tracked ones stay tracked (`.gitignore` does not untrack). Do delete any *new* `.pyc` your own runs
   drop into upstream directories: loading `loss.py` or `dataset.py` via `importlib` writes them.
7. **No new heavy dependencies.** The env pins are fragile (python 3.8, torch 1.11). Anything new must be
   pure-python and justified.
8. **Every run is logged.** Use `utils/logger.Logger`; it already dumps the full arg namespace (D5).
9. **The bridge never imports frozen code.** `tools/eve_bridge/` must not import anything under
   `ISP/*/GazeformerISP/src/utils/`. Its only ISP dependency is `models/loss.py`, loaded by path.
   Keep torch confined to `heatmap_metrics.py` so the rest stays runnable on the Windows dev machine.

   > **`tools/eve_senet/` is stricter still** (added 2026-09-10): it does not touch the ISP tree *at
   > all*, and it does not import `tools/eve_bridge/` either — it reads the bridge's **artefacts**
   > (D2), so the two packages share no module. What it does import is `SE-Net/`, which it may read
   > and subclass but never edit (convention 2). Keep the heavy imports deferred: `check_env.py`,
   > `durations.py` and `verify_embedding.py` must stay Windows-runnable, and `embed.py` reaches
   > Detectron2/MSDeformAttn only inside `load_model()`.
10. **Bridge artefacts are addressed positionally.** `fixations.json` record order *is* the key space
    for `gt_heatmaps.h5`. If you regenerate one, regenerate the other; the `fixations_sha256` check
    exists to make the mistake loud rather than silent.
