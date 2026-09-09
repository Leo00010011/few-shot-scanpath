# Implementation Plan — COCO-FreeView baseline on the cluster  *(superseded)*

> # ⚠ SUPERSEDED — 2026-09-09
>
> **This spec was never executed and F1 is not this.** F1 is back on OSIE:
> [`spec/2026-09-08-osie-eval-baseline/`](../2026-09-08-osie-eval-baseline/).
>
> **Why it was withdrawn.** The COCO-FreeView **test split is a held-out challenge
> benchmark** — the public `COCOFreeView_fixations_trainval.json` carries `train`/`valid`
> only, with no public test labels. A baseline computed on it is either not the published
> number or not reproducible by anyone outside the challenge, and F1's entire purpose is a
> **reproducible** environment proof. That reason is recorded in Roadmap §4 as a permanent
> out-of-scope entry rather than a preference, so it is not re-litigated later (D8).
>
> **Retained deliberately, not archived by accident.** §FR10 documents six real divergences
> between `ISP/COCO_FV/.../evaluation.py` and the OSIE one — most consequentially **FR10.4**,
> where `p2g()` computes `rank < 2` while logging the field as `R@3`, so the published
> COCO-FreeView `R@3` column is in fact **Recall@2**. Those are true facts about this
> codebase (mirrored in [TechStack.md](../constitution/TechStack.md) §4.2) and F6/F7 may need
> them. Nothing here is to be *implemented*; it is reference.
>
> **What moved.** `tools/cocofv_prep/` → `tools/osie_prep/` and `bash/test_cocofv.sh` →
> `bash/test_osie.sh`, both retargeted (Roadmap F1). Paths named below no longer exist.


> Companion to [requirements.md](requirements.md) and [validation.md](validation.md).

---

## Context and Design Decisions

### Why COCO-FreeView replaces OSIE as the baseline branch

The purpose of F1 has never been the OSIE numbers themselves — it is to prove the environment, the
checkpoints and the frozen metric code work *before* our data is in the picture (Roadmap §3 F1,
"every later failure then has an unambiguous cause"). Any released checkpoint serves that purpose.
COCO-FreeView serves it slightly better than OSIE for three reasons:

1. Its query set is **3 unseen subjects**, the same order of magnitude as the cohort Roadmap OPEN-5
   currently permits on the EVE bundle (2). The OSIE query set is 5. If the retrieval block degenerates
   at small `subject_num`, we learn that here, on data whose correct answer is published.
2. Its stimuli are photographic COCO scenes, closer to EVE's than the OSIE set.
3. Its `test.py` has **no `if i_batch > 100: break` cap** (FR9.6), so the "is the published number over
   101 images or the full split?" question that hung over the OSIE F1 checklist simply does not arise.

The cost is that COCO_FV is a less complete branch than OSIE — no `data_postprocess.py`, no checked-in
`fixations.json`, no shipped `prediction.json`, and an `evaluation.py` that has drifted from the OSIE
one. Those are the risks this plan spends most of its steps on. **F5's EVE branch still mirrors OSIE**
(FR16.4): OSIE is the better *template*, COCO_FV is the better *baseline*. Keeping those two roles
separate is deliberate and must survive into the constitution.

### Why the path mismatches are fixed by argument, never by editing defaults

`preprocess_fixations.py` writes `src/data/FV/fixations.json`; `test.py --fix_dir` defaults to
`src/data/fixations.json`. `feature_extractor.py` writes `<output>/image_features/<task>/`;
`--feat_dir` defaults to `src/data/image_features` (no task level). `text_data()` writes
`<output>/embeddings.npy`; `--emb_dir` defaults to `src/data/embeddings.npy`; a third `embeddings.npy`
sits checked-in at `src/`. Four inconsistencies, all in upstream code.

TechStack §6.2 ("additive over invasive") and §6.4 ("never hardcode an absolute path") settle it:
every one is resolved by passing an explicit CLI argument from the run script. The upstream files stay
byte-identical, so an `ISP/COCO_FV` diff against the authors' release stays readable, and the run
script becomes the single place that documents the real layout. Editing the defaults would hide the
mismatch in a place nobody reads and would make the branch diverge from upstream for no gain.

### Why the feature cache lives on BeeGFS and is never rsynced to scratch

Directly transplanted from the reference `train_ms.sh`, which carries the scar in a comment: *"previous
run silently failed to rsync the 22 GB .h5 and training then couldn't find it."* A COCO-FreeView
feature cache is one 6.3 MB `.pth` per stimulus (768 × 2048 × 4 B); a few thousand stimuli is tens of
GB, the same order as the cache that failed. Only the label JSON and the checkpoint — megabytes — go to
`$LOCAL_SCRATCH`. FR8.7 requires the exclusion be commented, with the reason, so the next person does
not "optimise" it back in.

### Why the preflight checks are a separate CPU tool, not code inside `test.py`

D7 demands loud failure on shape or coverage mismatch, and D1 forbids touching the frozen evaluator.
The equal-subject invariant (FR3.5c) is the one that matters most: `COCOSearch_evaluation` groups by
`"{task}/{name}"` and `test.py` slices with `index // args.subject_num`, so an image carrying 2 or 4
subjects instead of 3 does not crash — it shifts every subsequent image's predictions against the wrong
ground truth and produces a plausible, wrong number. That is exactly Mission P2. Catching it needs to
happen *before* a GPU allocation is spent, and it is pure JSON arithmetic, so it belongs in a small
CPU tool that also runs on the Windows dev machine (TechStack §1) without importing torch.

`tools/cocofv_prep/` mirrors the `tools/eve_bridge/` precedent: our code, outside the ISP tree, so
`ISP/COCO_FV/GazeformerISP/`'s import surface stays identical to upstream (TechStack §6.9's spirit).
It must not import anything under `ISP/*/GazeformerISP/src/utils/`.

### Why `evaluation.py`'s divergences are documented and not reconciled

D1 is unconditional: the metric code is frozen. The COCO_FV `evaluation.py` differs from the OSIE one
in six ways (FR10) — most consequentially `p2g()` computing `r3 = rank < 2`, so the field logged as
`R@3` is really Recall@2. That is what produced the paper's published COCO-FreeView retrieval column.
"Fixing" it would make our number incomparable to the table we are trying to reproduce, which defeats
the entire mission (Mission §1). So it is documented in three places — the FR10 table, `notes.md`, and
a new TechStack subsection — and the write-up must label the column `R@2`.

### Why three seeds and no support-set repeats

`select_fewshot_subject()` returns before its support-sampling branch when `split != 'train'`
(FR9.5), so on the query set `--random_support` and `--num_fewshot` change nothing but the log
filename. Repeating over support seeds would produce three identical numbers and prove nothing.
What *is* stochastic is `Sampling.random_sample` (TechStack §5), so the meaningful repetition is over
`--seed`. Three runs give the noise band FR13.2's success criterion needs; without it "within sampling
noise" is unfalsifiable.

### Why the constitution is amended in the same feature

Roadmap §4 lists COCO-FreeView as out of scope and §4's preamble says reopening any of those "is a
constitution change". Doing the work without the amendment would leave the constitution contradicting
the repository, which is the failure mode the whole spec workflow exists to prevent. The amendment is
Step 9, done last, so it records what was actually built rather than what was intended.

---

## Implementation Steps

### Step 1 — Verify the ground truth before writing anything

No files created. This step exists because five later steps depend on facts that must be *observed*
on the cluster, not assumed.

On the cluster, record into a scratch file for Step 8:

1. `ls -la $FV_IMAGE_ROOT | head` — confirm category subdirectories (FR2.1).
2. `python -c "import json;d=json.load(open('$FV_FIX_JSON'));print(len(d), sorted(d[0].keys()))"` —
   confirm FR3.4's key set and which label file is staged (FR2.2).
3. `ls -la weights/COCO_FV-*/COCO_FV/` — exact on-disk filenames (FR6.5).
4. `python -c "import torch;t=torch.load('<...>/fewshot_user_embedding_10.pt');print(type(t), getattr(t,'shape',None), getattr(t,'dtype',None))"`
   — FR6.4.
5. `python -c "import torch;c=torch.load('<...>/checkpoint_best.pth',map_location='cpu');print(list(c.keys()));
   print([k for k in c[[x for x in c if x!='optimizer'][0]] if 'subject_embed' in k])"`
   — FR6.6, does the state dict carry `subject_embed.weight`?
6. `diff ISP/OSIE/.../src/utils/evaltools/visual_attention_metrics.py ISP/COCO_FV/.../src/utils/evaltools/visual_attention_metrics.py`
   — FR10.7.

If (5) shows a `subject_embed` key, resolve it in Step 6 *before* burning a GPU allocation.

### Step 2 — `tools/cocofv_prep/__init__.py` and the error type

New file `tools/cocofv_prep/__init__.py`:

```python
class CocoFvPreflightError(RuntimeError):
    """Raised when a precondition for the COCO-FreeView evaluation run is violated."""
```

Nothing else. No torch import anywhere in this package — it must run on the Windows dev machine
(TechStack §1) and on the cluster login node without a GPU.

### Step 3 — `tools/cocofv_prep/normalize_fixations.py` (FR3.3)

Only needed if Step 1(2) shows the staged label file is *not* already normalized. Reproduces exactly
the three transformations `src/preprocess/preprocess_fixations.py` performs, and nothing more:

```python
_TASK_FIXES = {"potted plant": "potted_plant", "stop sign": "stop_sign"}

def normalize(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        r = dict(r)
        if r.get("split") == "val":
            r["split"] = "validation"
        r["task"] = _TASK_FIXES.get(r.get("task"), r.get("task"))
        out.append(r)
    return out
```

CLI: `--in`, `--out`; writes `json.dump(..., indent=2)`. Prints a counter of how many records each
transformation touched — a zero count for all three means the file was already normalized and the tool
was unnecessary, which is itself worth logging.

### Step 4 — `tools/cocofv_prep/check_fixations.py` (FR3.4, FR3.5)

The most important file in the feature. Pure stdlib + `os.path`.

```python
def check_fixations(fix_path, image_root, fewshot_subjects, split="test") -> dict:
    records = json.load(open(fix_path))
    if not isinstance(records, list):
        raise CocoFvPreflightError(f"{fix_path}: expected a JSON list, got {type(records)}")

    # FR3.4 — schema
    required = {"name", "subject", "X", "Y", "T", "length", "split", "task"}
    # ... collect records missing keys, raise listing first 5

    rows = [r for r in records if r["split"] == split
                              and r["subject"] in set(fewshot_subjects)]

    # (a) length agreement
    bad_len = [r for r in rows if not (len(r["X"]) == len(r["Y"]) == len(r["T"]) == r["length"])]

    # (b) subject coverage
    present = sorted({r["subject"] for r in rows})
    missing_subjects = sorted(set(fewshot_subjects) - set(present))

    # (c) EQUAL-SUBJECT INVARIANT — the one that silently corrupts results
    counts = Counter((r["task"], r["name"]) for r in rows)
    n = len(fewshot_subjects)
    ragged = {k: v for k, v in counts.items() if v != n}

    # (d) the str.replace('jpg','pth') trap (FR4.4)
    jpg_tasks = sorted({r["task"] for r in rows if "jpg" in r["task"]})

    # (e) coordinate range — counted, never clamped
    oob = sum(1 for r in rows
                for x, y in zip(r["X"], r["Y"])
                if not (0 <= x < 512 and 0 <= y < 320))

    # (f) stimulus presence
    missing_images = [f"{t}/{nm}" for (t, nm) in counts
                      if not os.path.isfile(os.path.join(image_root, t, nm))]

    # FR14.3 — short scanpaths get padded inside the frozen evaluator; count them
    n_short = sum(1 for r in rows if r["length"] < 3)
```

Raise `CocoFvPreflightError` on any of (a), (b), (c), (d), (f) — each message enumerating the first
5–10 offenders and the total count, never just "check failed". (e) and `n_short` are **counted and
returned**, not raised: out-of-range coordinates are a property of the authors' labels, and short
scanpaths are legitimate but change the frozen evaluator's behaviour (TechStack §4).

Return the counter dict; the CLI prints it as JSON so the run script can tee it into the log
(FR14.2's source data).

### Step 5 — `tools/cocofv_prep/check_features.py` (FR4.6)

Imports torch — this one is cluster-only, and it is the sole file in `tools/cocofv_prep/` that does,
mirroring `heatmap_metrics.py`'s role in `tools/eve_bridge/`.

```python
def check_features(fix_path, feat_dir, split="test", expected_shape=(768, 2048)) -> dict:
    pairs = sorted({(r["task"], r["name"])
                    for r in json.load(open(fix_path)) if r["split"] == split})
    missing, bad_shape = [], []
    for task, name in pairs:
        # mirror the loader EXACTLY: it does "{task}/{name}".replace('jpg','pth')
        rel = "{}/{}".format(task, name).replace("jpg", "pth")
        p = os.path.join(feat_dir, rel)
        if not os.path.isfile(p):
            missing.append(rel); continue
        t = torch.load(p, map_location="cpu")
        if tuple(t.shape) != expected_shape:
            bad_shape.append((rel, tuple(t.shape)))
    ...
```

The `rel` construction must be a literal copy of the loader's, unanchored `str.replace` included —
checking a path the loader would not build defeats the check. Raise on any `missing` or `bad_shape`,
listing the first 10.

### Step 6 — Stage the checkpoint and the subject embedding (FR6)

No new files; a documented sequence, and the exact commands go verbatim into `notes.md` so the run is
reproducible (D5). Using the names verified in Step 1(3):

```bash
mkdir -p ISP/COCO_FV/GazeformerISP/src/assets/FV-ex-012/checkpoints
ln -sfn "$PROJECT_DIR/weights/COCO_FV-20260904T121424Z-1-001/COCO_FV/checkpoint_best.pth" \
        ISP/COCO_FV/GazeformerISP/src/assets/FV-ex-012/checkpoints/checkpoint_best.pth

mkdir -p SE-Net/assets/FV-ex-012
ln -sfn "$PROJECT_DIR/weights/COCO_FV-20260904T121424Z-1-001/COCO_FV/fewshot_user_embedding_10.pt" \
        SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt
```

Symlinks, not copies: the weights are git-ignored and large, and a symlink makes the provenance
visible in `ls -la`. If Step 1(5) found a `subject_embed` key in the state dict, resolve it here —
the permitted resolution is to strip that key from the *loaded dict in memory* inside the run script's
own wrapper, or to accept a `strict=False` load documented in `notes.md`. **Not** an edit to
`evaluation.py` or `evaltools/*` (D1); `gazeformer.py` is not on the frozen list but should still be
left alone if a non-invasive option exists.

### Step 7 — `bash/test_cocofv.sh` (FR8)

New file at the repo root under `bash/`. Structure, in order — the cluster idioms are lifted from
`From-Noisy-Eye-Tracking-To-Scanpath/train_ms.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=cocofv_eval
#SBATCH --output=logs/cocofv_out_%j.log
#SBATCH --error=logs/cocofv_err_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonardo.ulloa@rai.usc.gal

set -euo pipefail
echo "Starting COCO-FreeView eval at: $(date)"
echo "Running on node: $SLURM_NODELIST"

# ---- tunables (FR8.4) ----------------------------------------------------
HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
FV_IMAGE_ROOT="${FV_IMAGE_ROOT:-$PROJECT_DIR/data/COCO_FV}"
FV_FIX_JSON="${FV_FIX_JSON:-$PROJECT_DIR/data/coco_fv_fixations_update_duration.json}"
FV_WORK="${FV_WORK:-$PROJECT_DIR/work/cocofv}"      # BeeGFS — features stream from here (FR4.7)
ISP_ENV="${ISP_ENV:-isp}"
FEWSHOT_SUBJECTS="${FEWSHOT_SUBJECTS:-0 1 2}"
SUBJECT_NUM="${SUBJECT_NUM:-3}"
SEED="${SEED:-0}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
BRANCH_DIR="$PROJECT_DIR/ISP/COCO_FV/GazeformerISP"

cd "$HOME_DIR"
echo "Mounting image"
sudo mount_image.py my_env.ext4 --rw

# ---- scratch staging: SMALL inputs only ----------------------------------
# The image_features/ cache is tens of GB (one 6.3 MB .pth per stimulus) and is read
# straight from BeeGFS, NOT copied into the size-limited scratch. The reference
# train_ms.sh records a run that silently failed to rsync a 22 GB cache; do not
# "optimise" this into an rsync.                                    (FR4.7, FR8.7)
mkdir -p "$LOCAL_SCRATCH/cocofv"
cp "$FV_FIX_JSON" "$LOCAL_SCRATCH/cocofv/labels_raw.json"

source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
conda activate "$ISP_ENV"

# ---- FR1.2 / FR1.3 -------------------------------------------------------
python - <<'PY'
import importlib, sys
print("python", sys.version)
for m in ("torch","numpy","scipy","skimage","multimatch_gaze","cv2"):
    try: print(m, importlib.import_module(m).__version__)
    except Exception as e: print(m, "MISSING", e)
PY
python -c "import multimatch_gaze,sys; v=multimatch_gaze.__version__; sys.exit(0 if v=='0.1.3' else \
  print('FATAL: multimatch-gaze', v, '!= 0.1.3 (metric-critical pin)') or 1)"

# ---- FR2.4 / FR3 / FR5 preflight (CPU) -----------------------------------
cd "$PROJECT_DIR"
mkdir -p "$FV_WORK/data/FV"
python tools/cocofv_prep/normalize_fixations.py \
    --in "$LOCAL_SCRATCH/cocofv/labels_raw.json" \
    --out "$FV_WORK/data/FV/fixations.json"
python tools/cocofv_prep/check_fixations.py \
    --fix "$FV_WORK/data/FV/fixations.json" \
    --images "$FV_IMAGE_ROOT" \
    --fewshot-subject $FEWSHOT_SUBJECTS \
    | tee "$FV_WORK/preflight_fixations.json"

# ---- Stage B, guarded (FR8.6) --------------------------------------------
cd "$BRANCH_DIR"
if [ "$FORCE_FEATURES" = "1" ] || \
   ! python "$PROJECT_DIR/tools/cocofv_prep/check_features.py" \
        --fix "$FV_WORK/data/FV/fixations.json" \
        --feat-dir "$FV_WORK/data/FV/image_features" ; then
  echo "Extracting image features into $FV_WORK/data/FV/image_features"
  python src/preprocess/feature_extractor.py \
      --dataset_path "$FV_IMAGE_ROOT" --output_path "$FV_WORK/data/FV" --cuda 0
  python "$PROJECT_DIR/tools/cocofv_prep/check_features.py" \
      --fix "$FV_WORK/data/FV/fixations.json" \
      --feat-dir "$FV_WORK/data/FV/image_features"
fi

# ---- Stage D + E (FR7.1, FR7.2) ------------------------------------------
CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject $FEWSHOT_SUBJECTS \
  --subject_num "$SUBJECT_NUM" \
  --seed "$SEED" \
  --eval_repeat_num 1 \
  --fix_dir  "$FV_WORK/data/FV/fixations.json" \
  --feat_dir "$FV_WORK/data/FV/image_features" \
  --emb_dir  "$FV_WORK/data/FV/embeddings.npy" \
  --img_dir  "$FV_IMAGE_ROOT" \
  --evaluation_dir src/assets/FV-ex-012 \
  --user_emb_path ../../../SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt

# ---- FR15.2: preserve per-seed artefacts before the next seed overwrites --
mkdir -p "result/FV-ex-012/log/seed$SEED"
cp result/FV-ex-012/log/log_test_subject_*.txt \
   result/FV-ex-012/log/prediction.json \
   "result/FV-ex-012/log/seed$SEED/"

echo "Finished at: $(date)"
```

Two subtleties the script must get right:

- `cd "$BRANCH_DIR"` happens **before** both `feature_extractor.py` and `test.py`, because every
  relative default in those files is anchored to `ISP/COCO_FV/GazeformerISP/` (FR7.1). The preflight
  tools are invoked by absolute path so they work from either directory.
- `$FEWSHOT_SUBJECTS` is intentionally **unquoted** in the `--fewshot_subject` position — argparse
  `nargs='+'` needs three separate words. This is the one place `set -u` word-splitting is wanted, and
  it deserves a comment so nobody "fixes" it by adding quotes.

`embeddings.npy` (FR5): if the checked-in `src/embeddings.npy` satisfies FR5.2, the script copies it
to `$FV_WORK/data/FV/embeddings.npy`; otherwise it runs `feature_extractor.text_data()` as part of the
Stage B block. Decide in Step 1 and hardcode the chosen branch, so the run is deterministic.

### Step 8 — Run, then write `notes.md`

Submit three times, once per seed (FR13.3):

```bash
sbatch bash/test_cocofv.sh                      # SEED defaults to 0
SEED=1 sbatch --export=ALL,SEED=1 bash/test_cocofv.sh
SEED=2 sbatch --export=ALL,SEED=2 bash/test_cocofv.sh
```

Then create `spec/2026-09-08-cocofv-baseline-on-cluster/notes.md` with these sections, in order:

1. `## Environment as built` — FR1.4's deviation table.
2. `## Data as staged` — Step 1's observations: label filename, record count, category count, image
   count, the verified on-disk case of `FV-ex-012` and the embedding filename (FR6.5).
3. `## Preflight counters` — `preflight_fixations.json` verbatim: test images, cells, out-of-range
   coordinate count, `n_short` (FR14.3).
4. `## Results` — FR13.1's table across the three seeds, with the paper's COCO-FreeView row and the
   absolute difference. The retrieval column is labelled **`R@2` (logged as `R@3`)** (FR10.4).
5. `## Denominators` — FR14.2's dropped-cell accounting.
6. `## Divergences observed` — FR10's table with the Step 1(6) diff result filled in, plus FR11.3's
   statement that dense index == real subject id **only** because the query set is `0 1 2` ascending,
   plus FR12.2's `prediction.json` schema notes for F6, plus FR7.3's "`--img_dir` is never read".
7. `## Verdict` — pass/fail against FR13.2, and what the run does and does not license.

### Step 9 — Amend the constitution (FR16)

Last, so it records what was built. Edits to `spec/constitution/`:

`Roadmap.md`
- §4: the out-of-scope bullet becomes
  "✗ The **COCO-Search18** branch, and anything task/search-conditioned."
- §1 status board: F1 retitled "Reproduce the **COCO-FreeView** eval baseline on the cluster",
  status ✓ DONE with the date, spec folder linked.
- §3 F1: body replaced by a pointer to this spec, with the OSIE checklist retained under a
  "~~Superseded: the OSIE formulation~~" heading and one line saying the OSIE run was **dropped**,
  not deferred, and why (Step 1's rationale above).
- §2 dependency graph: `F1 OSIE baseline` → `F1 COCO-FV baseline`.
- §6 external dependencies: drop the OSIE stimulus-images row for F1; add COCO-FreeView images and
  labels, marked "already staged on the cluster".
- §3 F6: add that F6 must handle FR12.2's `prediction.json` schema divergences, since its fixture is
  now a COCO_FV artefact rather than an OSIE one.

`TechStack.md`
- §2 repository map: add one sentence stating that F1's baseline branch (`ISP/COCO_FV`) and F5's
  template branch (`ISP/OSIE`) are deliberately different, and why — OSIE carries
  `data_postprocess.py`, a checked-in `fixations.json` and a shipped `prediction.json` that COCO_FV
  lacks (FR16.4).
- New subsection §4.2 "Per-branch metric drift": FR10's divergence table, prefaced by the warning that
  `evaluation.py` is **not** one file replicated across branches, and that all versions are frozen
  under D1 regardless (FR16.5).
- §1 command conventions: add the `sbatch bash/test_cocofv.sh` line and the `tools/cocofv_prep/` CLIs.

---

## Implementation Order

1. **Step 1** — Verify on the cluster: data layout, label schema, weight filenames and case, embedding
   shape, checkpoint keys, `visual_attention_metrics.py` diff. Nothing is written; everything after
   depends on it.
2. **Step 2** — `tools/cocofv_prep/__init__.py` + `CocoFvPreflightError`.
3. **Step 3** — `normalize_fixations.py` (skip if Step 1 showed the labels are already normalized).
4. **Step 4** — `check_fixations.py`, including the equal-subject invariant.
5. **Step 5** — `check_features.py`, mirroring the loader's unanchored `replace('jpg','pth')`.
6. **Step 6** — Symlink the checkpoint and the subject embedding; resolve any `subject_embed` key
   collision found in Step 1(5).
7. **Step 7** — `bash/test_cocofv.sh`.
8. **Step 8** — Three seeded runs; write `notes.md`.
9. **Step 9** — Amend `Roadmap.md` and `TechStack.md`.

Steps 2–5 are runnable and testable on the Windows dev machine (Step 5 needs torch, which is installed
CPU-only there) against a hand-built fixture, so the preflight tooling can be validated before any
cluster allocation is used. Steps 6–8 are cluster-only.
