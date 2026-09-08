# F1 — Implementation plan

> Companion to [requirements.md](requirements.md) and [validation.md](validation.md).
> Roadmap: **F1**. Constitution: [Mission](../constitution/Mission.md) ·
> [TechStack](../constitution/TechStack.md) · [Roadmap](../constitution/Roadmap.md).

---

## Context and Design Decisions

### Why F1 exists at all, and what that implies about how it is built

F1 produces no reusable code and no data of our own. Its entire output is *evidence* — evidence that the
`isp` environment, the released checkpoint, and the frozen metric suite compose into the numbers the paper
reports. Everything in this plan is therefore optimised for one property: **that a later failure in F5 can
be attributed unambiguously**. The corollary is that F1 must change as little as possible. If F1 both
reproduces the baseline *and* carries three convenience refactors, then when F5 disagrees with the paper
we have four suspects instead of one.

This is why the plan below stages files rather than moving them, adapts the filesystem to a hardcoded
`train/` subdirectory rather than fixing the hardcode, calls `image_data()` as a function rather than
repairing `feature_extractor.py`'s broken `__main__`, and leaves an inert `i_batch > 100` cap in place.
Each of those is a place where the "obviously better" change would have been a deviation from the
configuration that produced the published number.

### The one edit, and why it is allowed

D1 freezes `utils/evaluation.py` and `utils/evaltools/*`. `test.py` is **not** on that list, and D6
requires means *and* standard deviations. The evaluator already computes `cur_metrics_std` and returns
it — `test.py` simply binds it and never reads it. So the edit is not adding a computation; it is
un-discarding one that already happened. It is confined to the logging block, touches no argument, no
tensor and no file write, and leaves the `SM` / `MM` / `SED` expressions byte-identical. Under Working
convention 2 ("additive over invasive", and when a shared file must change the default path stays
bit-identical) this is the minimal form of the change.

The alternative considered and rejected was forking `test.py` into `test_baseline.py`. That keeps the
upstream diff at zero, but it clones ~200 lines including the model construction and the sampling loop,
and every one of those lines then becomes a place where F1's run can silently differ from the shipped
configuration. A four-line logging change in the original file is the smaller risk.

### What is computed on the cluster and what is checked on the laptop

Stages B, D and E need CUDA and run on the cluster (TechStack §1: never run anything importing
`torch.cuda` on the Windows dev machine). But every *verification* in [validation.md](validation.md) that
does not need a GPU is deliberately expressed over the written artefacts — `prediction.json`,
`fixations.json`, the `.pth` shapes, the log text — so it can run on Windows CPU with `py`. That split is
what makes F6 (the offline re-scorer) a natural next step rather than a new capability: F1 already
establishes that every reported number is re-derivable from artefacts alone (D5).

### Why the seed sweep, and why not `--eval_repeat_num`

Inference is stochastic: `Sampling.random_sample()` draws from the action-probability map and the
log-normal duration parameters (TechStack §5). A single run therefore produces one draw from a
distribution, and "reproduced the paper's number" is not a statement one draw can support. Two ways to get
a spread were available.

`--eval_repeat_num > 1` averages several samples per trial inside one run — but `predict_results` is
appended to inside the repeat loop, so the resulting `prediction.json` would contain `repeat_num`
records per `(name, subject)` pair. That breaks the FR14 uniqueness contract and, downstream, F6's ability
to index predictions by trial. It also changes the evaluator's input structure relative to the shipped
run.

Sweeping `--seed` across five whole runs keeps every artefact in the shipped shape and measures exactly
the quantity we need: run-to-run variation of the reported headline numbers under an otherwise identical
configuration. The cost is five times the GPU time on a 70-image test split, which is cheap.

One wrinkle the plan must handle: the log filename is
`log_test_subject_{num_fewshot}_{random_support}.txt` and contains **no seed**, while `prediction.json`
has a fixed name. Five seeds would overwrite each other. Rather than change `test.py`'s path construction
(which would be a second, non-print-only edit), the run script varies `--random_support` in lockstep with
`--seed` — legitimate precisely because Finding E establishes it is inert on the test path — and copies
`prediction.json` aside after each run.

### Ordering

The dependency chain is strict and each link is a place where a wrong assumption would otherwise surface
much later as a confusing error:

- The env must exist before anything can be shape-checked, and `multimatch-gaze==0.1.3` must be confirmed
  before any GPU time is spent, since a wrong version invalidates the run retroactively.
- Assets must be *located* before they are staged, because their on-disk case is genuinely unknown
  (TechStack §3.4 records the docs contradicting each other).
- Stage B must complete and be verified before the first inference run, because `OSIE_evaluation`
  discovers a missing `.pth` as a mid-run `torch.load` failure after the model is already on the GPU.
- The `test.py` edit lands before the sweep, not after, so all five runs share one logging format.

---

## Step 1 — Build the `isp` environment and verify the metric-critical pins

**Where:** cluster shell. **Writes:** `spec/2026-09-08-osie-eval-baseline/environment.md`.

Create the environment from `ISP/environment.yml`, then capture and audit it *before* touching the GPU.

```bash
conda env create -f ISP/environment.yml -n isp
conda activate isp
conda list  > /tmp/isp_conda_list.txt
pip freeze  > /tmp/isp_pip_freeze.txt
```

Audit against the TechStack §1 pin table. The check is not "did it install" but "did it install the
version the metric contract assumes":

```python
# Run under the isp env. Hard-stops before any GPU time is committed.
import importlib.metadata as md
HARD = {"multimatch-gaze": "0.1.3"}                      # wrong version => numbers not comparable
SOFT = {"numpy": "1.23.5", "scipy": "1.10.0", "torch": "1.11.0",
        "scikit-image": "0.19.2", "scikit-learn": "0.22.2"}
for pkg, want in HARD.items():
    got = md.version(pkg)
    assert got == want, f"HARD STOP: {pkg} pinned {want}, got {got} (FR1)"
for pkg, want in SOFT.items():
    got = md.version(pkg)
    if got != want:
        print(f"DEVIATION: {pkg} pinned {want}, got {got} -- record in environment.md")
```

Write `environment.md` with: the conda/pip captures (or their relevant excerpts), one row per pin-table
package marked `match` / `deviation: pinned X, got Y`, the CUDA driver and GPU model, and the resolved
python version. Satisfies **FR1**.

**Do not proceed** if the `multimatch-gaze` assertion fires. A different MultiMatch version is not a
deviation to note — it is a different metric, and D1's whole purpose is that our MM numbers mean the same
thing as the paper's.

---

## Step 2 — Locate and stage the checkpoint and the user embedding

**Where:** cluster shell, from the repo root. **Writes:** staged files under
`ISP/OSIE/GazeformerISP/src/assets/` and `SE-Net/assets/` (both git-ignored content).

First **observe**, do not assume — TechStack §3.4 records the READMEs contradicting each other on
`OSIE-` vs `osie-` and `user` vs `subject`, and the cluster filesystem is case-sensitive:

```bash
ls -la weights/OSIE-*/OSIE/
ls -la ISP/OSIE/GazeformerISP/src/assets/          # currently: OSIE-ex-10to15/ (README only)
ls -la SE-Net/assets/ 2>/dev/null || echo "SE-Net/assets does not exist -- will create"
```

Record the exact names observed. Then stage, creating the two target directories:

```bash
CKPT_SRC=$(ls weights/OSIE-*/OSIE/checkpoint_best.pth)
EMB_SRC=$(ls  weights/OSIE-*/OSIE/fewshot_user_embedding_10.pt)

mkdir -p ISP/OSIE/GazeformerISP/src/assets/OSIE-ex-10to15/checkpoints
mkdir -p SE-Net/assets/OSIE-ex-10to15

ln -sf "$(realpath "$CKPT_SRC")" \
   ISP/OSIE/GazeformerISP/src/assets/OSIE-ex-10to15/checkpoints/checkpoint_best.pth
ln -sf "$(realpath "$EMB_SRC")" \
   SE-Net/assets/OSIE-ex-10to15/fewshot_user_embedding_10.pt
```

Symlinks over copies: `weights/` stays the single source of truth and nothing large is duplicated. If the
cluster filesystem does not support symlinks, copy instead and say so in `environment.md`.

Note the asymmetry in how the two paths are resolved, because it is a common source of confusion:
`--evaluation_dir` is relative to the **CWD** (`ISP/OSIE/GazeformerISP/`) and `checkpoints/` is appended
by `test.py`; `--user_emb_path` is a path with `../../../` baked into the default, resolving out of
`ISP/OSIE/GazeformerISP/` and back down into `SE-Net/`. Both only work when the CWD is
`ISP/OSIE/GazeformerISP/`. Satisfies **FR2**.

Then check the embedding is what the model expects, before it reaches the model:

```python
import torch
emb = torch.load("../../../SE-Net/assets/OSIE-ex-10to15/fewshot_user_embedding_10.pt",
                 map_location="cpu")
t = emb if torch.is_tensor(emb) else next(iter(emb.values()))   # released files vary in wrapping
assert t.shape[-1] == 384, f"subject_feature_dim mismatch: got {t.shape} (FR3)"
assert t.shape[0] >= 5,    f"need >= subject_num=5 rows, got {t.shape} (FR3)"
print("user embedding:", tuple(t.shape), t.dtype)   # -> environment.md
```

Satisfies **FR3**. If `emb` is a dict, record its key structure in `environment.md` — F3 will need to know
the container format when it decides OPEN-2.

---

## Step 3 — Stage the OSIE stimuli into the hardcoded `train/` subdirectory

**Where:** cluster. **Writes:** `<STAGE_B_ROOT>/train/*.jpg`.

Obtain the OSIE 800×600 stimuli (NUS-VIP `predicting-human-gaze-beyond-pixels`, Roadmap §6). They are
**not** in this repo and are not committed (Working convention 5).

`image_data()` reads `join(dataset_path, 'train/')` — hardcoded (TechStack §3.2). Adapt the filesystem,
not the code:

```bash
STAGE_B_ROOT=<a scratch path outside the repo>
mkdir -p "$STAGE_B_ROOT/train"
# symlink or copy every OSIE .jpg into $STAGE_B_ROOT/train/
ls "$STAGE_B_ROOT/train" | wc -l
```

Cross-check the staged set against the names the loader will actually ask for, so a shortfall is found
now rather than as a `torch.load` failure in Step 5:

```python
import json, os
names = {r["name"] for r in json.load(open("ISP/OSIE/GazeformerISP/src/data/fixations.json"))}
have  = set(os.listdir(f"{STAGE_B_ROOT}/train"))
missing = names - have
assert not missing, f"{len(missing)} stimuli missing from train/, e.g. {sorted(missing)[:5]} (FR4)"
print(f"{len(names)} distinct stimuli required, all present")
```

Satisfies **FR4**.

---

## Step 4 — Run Stage B and verify the feature tensors

**Where:** cluster, from `ISP/OSIE/GazeformerISP/`. **Writes:** `src/data/image_features/*.pth`.

Create the output directory first — `image_data()` never does (FR5), and without it every `torch.save`
raises `FileNotFoundError`:

```bash
cd ISP/OSIE/GazeformerISP
mkdir -p src/data/image_features
```

Invoke `image_data()` as a **function**, not the module: `feature_extractor.py`'s `__main__` calls
`text_data(dataset_path=args.p, ...)` and `args.p` does not exist (Finding D), so running the module
raises `AttributeError` after the image pass — and would in any case regenerate the shipped
`embeddings.npy` (Finding C, FR6).

```python
# from ISP/OSIE/GazeformerISP/, isp env active
import sys, torch
sys.path.insert(0, "src")
from preprocess.feature_extractor import image_data

image_data(dataset_path="<STAGE_B_ROOT>",
           output_path="src/data",
           device=torch.device("cuda:0"),
           overwrite=True)          # True: a partial earlier run must not pass as complete (FR7)
```

This downloads `MaskRCNN_ResNet50_FPN_Weights.COCO_V1` on first use (Roadmap §6) — expect a one-time
torchvision fetch. Then verify every tensor before any inference runs (**FR7**, D7):

```python
import json, os, torch
names = {r["name"] for r in json.load(open("src/data/fixations.json"))}
bad = []
for n in sorted(names):
    p = os.path.join("src/data/image_features", n.replace("jpg", "pth"))   # the loader's literal replace
    if not os.path.exists(p):
        bad.append((n, "missing")); continue
    t = torch.load(p, map_location="cpu")
    if tuple(t.shape) != (768, 2048): bad.append((n, f"shape {tuple(t.shape)}"))
    elif t.dtype != torch.float32:    bad.append((n, f"dtype {t.dtype}"))
    elif not torch.isfinite(t).all(): bad.append((n, "non-finite"))
assert not bad, f"{len(bad)} bad feature files, e.g. {bad[:5]} (FR7)"
print(f"{len(names)} feature tensors verified (768, 2048) float32")
```

`(768, 2048)` is `(24*32, 2048)` — the `(384*2, 512*2)` resize through the ResNet-50 backbone body,
flattened (TechStack §3.2). A different first dimension means the resize or the backbone differs and
every downstream number would be wrong.

---

## Step 5 — The print-only `test.py` edit

**Modifies:** `ISP/OSIE/GazeformerISP/src/test.py`, one region only.

The current block, near the end of `main()`, discards `cur_metrics_std` and routes the headline through
bare `print` (so it never reaches the log file). Replace **only** the logging, keeping the `SM` / `MM` /
`SED` expressions character-identical so an edited run and an unedited run produce the same numbers:

```python
# BEFORE
logger.info("The metrics for best model performance are: ")
for metrics_key in cur_metrics.keys():
    for (metric_name, metric_value) in cur_metrics[metrics_key].items():
        logger.info("{metrics_key:10}-{metric_name:15}: {metric_value:.4f}".format(...))
...
print('SM: {}, MM: {}, SED: {}'.format(round(SM, 3), round(MM, 3), round(SED, 3)))

# AFTER
logger.info("The metrics for best model performance are: ")
for metrics_key in cur_metrics.keys():
    for (metric_name, metric_value) in cur_metrics[metrics_key].items():
        std = cur_metrics_std.get(metrics_key, {}).get(metric_name)   # retrieval has no std (Finding G)
        if std is None:
            logger.info("{:10}-{:15}: {:.4f}".format(metrics_key, metric_name, metric_value))
        else:
            logger.info("{:10}-{:15}: {:.4f} +/- {:.4f}".format(
                metrics_key, metric_name, metric_value, std))

SM  = scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))   # unchanged
MM  = np.mean(list(cur_metrics["MultiMatch"].values()))            # unchanged
SED = cur_metrics["VAME"]["SED"]                                   # unchanged
STDE = cur_metrics["VAME"]["STDE"]                                 # newly surfaced
logger.info("HEADLINE seed={} SM={:.4f} MM={:.4f} SED={:.4f} STDE={:.4f}".format(
    args.seed, SM, MM, SED, STDE))
print('SM: {}, MM: {}, SED: {}'.format(round(SM, 3), round(MM, 3), round(SED, 3)))
```

Three constraints on this edit:

- `cur_metrics_std.get(...)` — never `cur_metrics_std[...]`. The `retrieval scanmatch w/ duration` group
  exists in `cur_metrics` only, and indexing it would raise **after** the whole evaluation has run,
  discarding an hour of work at the last line.
- The `HEADLINE` line embeds `args.seed`, which is what makes the Step 7 sweep collectable by grepping
  the logs rather than by hand-transcribing.
- The `i_batch > 100` cap above is **not touched** (FR9). It is inert here — 70 test images at
  `--batch 1` is 70 batches — and removing it would make F1's configuration differ from the one behind
  the shipped `prediction.json`.

Satisfies **FR8**, **FR9**.

---

## Step 6 — Write `bash/test.sh`

**Creates:** `ISP/OSIE/GazeformerISP/bash/test.sh` (committed — the one new repo file in F1).

Mirrors the style of the existing `bash/train.sh`. Every path is passed explicitly rather than inherited
from an argparse default, so the resolved configuration is readable from the script alone (D5), and no
absolute path appears (Working convention 4).

```bash
# Run from ISP/OSIE/GazeformerISP/ -- every path below is relative to that directory.
#   SEED=0 sh bash/test.sh
#
# F1 query-set baseline: released checkpoint, released 10-shot user embedding,
# unseen OSIE subjects 10-14. Eval-only; nothing here trains.

SEED=${SEED:-0}
GPU=${GPU:-0}
EVAL_DIR=src/assets/OSIE-ex-10to15
USER_EMB=../../../SE-Net/assets/OSIE-ex-10to15/fewshot_user_embedding_10.pt
RESULT_DIR=result/$(basename $EVAL_DIR)/log

# --random_support is inert on the test path (select_fewshot_subject returns early when
# split != 'train'); it is varied with SEED purely so the per-seed logs do not collide.
CUDA_VISIBLE_DEVICES=$GPU python src/test.py \
  --fewshot_subject 10 11 12 13 14 \
  --subject_num 5 \
  --eval_repeat_num 1 \
  --batch 1 \
  --seed $SEED \
  --random_support $SEED \
  --num_fewshot 10 \
  --evaluation_dir $EVAL_DIR \
  --user_emb_path $USER_EMB \
  --fix_dir src/data/fixations.json \
  --feat_dir src/data/image_features \
  --emb_dir src/data/embeddings.npy

# prediction.json has a fixed name and would be overwritten by the next seed.
cp $RESULT_DIR/prediction.json $RESULT_DIR/prediction_seed${SEED}.json
```

Notes carried in the script's comments, so the next reader does not rediscover them:

- `--img_dir` is deliberately **absent**. `OSIE_evaluation` never opens a stimulus (Finding B); passing it
  would imply inference depends on the images, which it does not.
- `--random_support $SEED` is safe *only because* Finding E established it is inert on the test path. It
  must not be copied into F5 without re-checking that, since F5 may score a split where it is live.
- `--num_fewshot 10` is likewise inert here; it is pinned rather than left default so the log filename is
  predictable.

Satisfies **FR10**, **FR11**, **FR13**.

---

## Step 7 — The reference run and the seed sweep

**Where:** cluster, from `ISP/OSIE/GazeformerISP/`.
**Writes:** `result/OSIE-ex-10to15/log/{log_test_subject_10_<seed>.txt, prediction_seed<seed>.json}`.

Run seed 0 first and verify it fully (Steps 8–9) **before** spending time on the sweep — a structural
problem such as an all-`-1` score matrix or a subject-axis off-by-one will show up on the first run and
there is no point repeating it four more times.

```bash
cd ISP/OSIE/GazeformerISP
SEED=0 sh bash/test.sh                      # reference run
# ... verify per Steps 8-9 ...
for S in 1 2 3 4; do SEED=$S sh bash/test.sh; done
```

The per-seed log lands at `log_test_subject_10_<seed>.txt` (from
`log_test_subject_{num_fewshot}_{random_support}.txt` with `--random_support` tracking the seed), so the
five runs do not overwrite each other; `prediction.json` is copied aside by the script. Record this naming
convention in `results.md` — it is not obvious from `test.py`, which never sees the seed in the filename.

Expected shape of a healthy run: 70 progress steps in the inference bar (`len(test_loader) * repeat_num`
with 70 images and `repeat_num=1`), then 70 in the evaluator's bar. Satisfies **FR12**, **FR13**.

---

## Step 8 — Verify coverage and the score matrix (D7)

**Where:** may run on the cluster or on Windows CPU (`py`) against the artefacts.

Before any number is written down, confirm the evaluator was fed the structure it assumes. The
equal-subject invariant is what keeps its `(subject × subject)` matrix square (TechStack §3.1); a
violation does not raise, it produces a ragged matrix and plausible nonsense.

```python
import json, collections
fx = json.load(open("src/data/fixations.json"))
test = [r for r in fx if r["split"] == "test"]
per_img = collections.Counter(r["name"] for r in test)
assert set(per_img.values()) == {15}, f"unequal subjects per image: {set(per_img.values())} (FR15)"
assert len(per_img) == 70, f"expected 70 test images, got {len(per_img)} (FR15)"
q = [r for r in test if r["subject"] in {10, 11, 12, 13, 14}]
per_img_q = collections.Counter(r["name"] for r in q)
assert set(per_img_q.values()) == {5}, f"query cohort ragged: {set(per_img_q.values())} (FR15)"
```

Then the failure guard and the NaN-drop count, both derived from what the run returned rather than by
editing the frozen evaluator (**FR15**, **FR16**):

- **Guard:** if any diagonal mean in the log is exactly `-1.0000`, the run **failed** — `-1` is the
  evaluator's uninitialised sentinel, not a bad score (TechStack §4). Record nothing and diagnose.
- **Dropped rows:** `is_eliminating_nan=True` silently removes NaN MultiMatch rows before the mean. The
  denominator is `70 × 5 = 350` diagonal rows; the count actually averaged is
  `collect_multimatch_diag_rlts.shape[0]` inside the frozen function. Re-derive the difference from the
  returned `score_details` — its MultiMatch slots hold `NaN` where the metric failed and `-1` where the
  cell never ran — and report `dropped = 350 - kept` alongside every MultiMatch mean. A MultiMatch number
  published without this count violates D7.

The shipped `fixations.json` contains **11 records with `length < 3`** across all splits; any of these on
the test split are padded to length 3 with `(1., 1., 1e-3)` inside the frozen evaluator, and the padded
array then replaces the original for **all** subsequent metrics in that cell (TechStack §4). Count how
many fall in the query cohort and record it — it is the same counter F7 will want from
`bridge_report.json`'s `short_scanpath` for our data.

---

## Step 9 — Verify `prediction.json` against the F6 contract

**Where:** Windows CPU (`py`) is sufficient — this is exactly the offline-artefact check F6 generalises.

The subject-axis assertion is the important one. `select_fewshot_subject()` remaps 10–14 to a dense 0–4
for the model, and `get_prediction_list()` must map them back via `args.fewshot_subject[subject_idx]`
before writing. A dense `0..4` in the output means D4 was violated and every diagonal score is scoring the
wrong person against the wrong person.

```python
import json, collections
pred = json.load(open("result/OSIE-ex-10to15/log/prediction_seed0.json"))
ship = json.load(open("result/git-osie-useremb-ex-10to15/log/prediction.json"))

assert {k for r in pred for k in r} == {"name", "subject", "X", "Y", "T"}      # FR14
assert len(pred) == 350 == len(ship)
assert sorted({r["subject"] for r in pred}) == [10, 11, 12, 13, 14]            # D4 roundtrip
assert {r["name"] for r in pred} == {r["name"] for r in ship}
assert len({(r["name"], r["subject"]) for r in pred}) == 350                   # no repeat-loop dupes
for r in pred:
    assert len(r["X"]) == len(r["Y"]) == len(r["T"])
    assert 1 <= len(r["X"]) <= 16
    assert all(isinstance(v, int) for v in r["X"] + r["Y"] + r["T"])
```

Coordinates are ints in the **resized 512×384** space and `T` is int milliseconds — `get_prediction_list()`
applies `int(round(t * 1000, 3))` to seconds (TechStack §3.5). Sanity-bound them against the shipped file
rather than against an assumed range. Satisfies **FR14**.

---

## Step 10 — Record the baseline and decide

**Writes:** `spec/2026-09-08-osie-eval-baseline/results.md`.

Assemble from the five logs:

- **Reference table (seed 0), mean ± std** for all five MultiMatch dimensions, ScanMatch with/without
  duration, `SED`, `STDE`, `SED_best`, `STDE_best`, and the retrieval block (`pmrr`, `pr1`, `pr3`, `pr5`,
  `rsum`) — the retrieval row without a std, since the evaluator does not produce one (Finding G).
- **Seed spread:** mean and min–max across seeds 0–4 for `SM`, `MM`, `SED`, `STDE`, `pmrr`, collected by
  grepping the `HEADLINE` lines.
- **The paper's OSIE row** alongside, with the comparison judged against the measured spread — not
  against an eyeballed tolerance.
- **Annotations:** `pr5` is structurally saturated at 100.0 with 5 subjects (Finding F) and carries no
  information; the NaN-dropped MultiMatch row count from Step 8; the short-scanpath padding count; and the
  note that the `i_batch > 100` cap never fired because the test split is 70 images (Finding A).
- **The exact command**, the resolved env from `environment.md`, and the resolved asset paths from Step 2.

Then decide (**FR17**, **FR18**). If the numbers sit inside the seed spread of the paper's row, F1 is done:
tick the Roadmap F1 checklist, note that F6 is now unblocked, and record that Findings A and E close the
roadmap's "decide and document" items. If they do not, F1 is **not** done — write the discrepancy up and
escalate it as a new roadmap open decision. Do not reach for the metric parameters, the resize, or the
evaluator; a setup that cannot reproduce OSIE cannot license any statement about our data (D8), and
tuning it into agreement would destroy exactly the diagnostic value F1 exists to provide.

---

## Implementation Order

1. **Step 1** — build `isp`, hard-stop on `multimatch-gaze != 0.1.3`, write `environment.md`. *(FR1)*
2. **Step 2** — `ls` the real asset names, stage checkpoint + user embedding, assert the 384-dim shape. *(FR2, FR3)*
3. **Step 3** — stage OSIE stimuli into `<STAGE_B_ROOT>/train/`, cross-check against `fixations.json` names. *(FR4)*
4. **Step 4** — `mkdir src/data/image_features`, call `image_data()` as a function, verify `(768, 2048)` float32. *(FR5, FR6, FR7)*
5. **Step 5** — print-only `test.py` logging edit; leave the `i_batch` cap alone. *(FR8, FR9)*
6. **Step 6** — write `bash/test.sh`. *(FR10, FR11, FR13)*
7. **Step 7** — seed-0 reference run, verify, then sweep seeds 1–4. *(FR12, FR13)*
8. **Step 8** — coverage assertions, `-1` failure guard, NaN-drop and short-scanpath counts. *(FR15, FR16)*
9. **Step 9** — `prediction.json` schema + D4 subject-roundtrip check. *(FR14)*
10. **Step 10** — write `results.md`; declare done or escalate. *(FR17, FR18)*

Steps 8 and 9 gate Step 7's sweep: run them against seed 0 before committing GPU time to seeds 1–4.
