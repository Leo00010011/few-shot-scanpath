# Mission

> Constitution file 1 of 3. Read together with [TechStack.md](TechStack.md) and [Roadmap.md](Roadmap.md).
> Last updated: 2026-09-07

---

## 1. The core idea

This repository is the official CVPR 2025 release of **ISP-SENet** ("Few-shot Personalized Scanpath
Prediction", Xue et al.). It predicts, for a given stimulus image *and a given individual subject*, the
sequence of fixations (x, y, duration) that subject would produce — and it does so for **unseen** subjects
from only a handful of that subject's example scanpaths, with no test-time fine-tuning.

**We are not the authors of this model.** We are consumers of it. Our purpose in this fork is:

> **Evaluate the released ISP-SENet checkpoints on a new, locally-collected free-viewing eye-tracking
> dataset, and report exactly the scanpath-similarity metrics the original authors report — computed by
> the original authors' own metric code — so that our numbers are directly comparable to the numbers in
> the paper.**

The dataset is ours; the model, the checkpoints, and the metrics are theirs. The scientific value of the
exercise depends entirely on the *metric implementation being unchanged*. A re-implemented ScanMatch or a
differently-parameterised MultiMatch produces numbers that cannot be compared to the published tables,
which would defeat the purpose.

---

## 2. What problem is being solved

Concretely, three problems, in order of difficulty:

**P1 — Data impedance mismatch.**
Our dataset lives in a separate repository behind its own dataloader. This repo consumes a rigid, flat
JSON schema (`fixations.json`) plus precomputed per-image ResNet feature tensors. Nothing bridges the two.
Coordinate space, duration units, subject indexing, and stimulus resolution all differ and all silently
corrupt metrics if mismatched.

**P2 — Subject alignment.**
The model is *personalized*: prediction `i` is meaningful only when scored against ground truth from the
*same* subject `i`. The evaluation code encodes this as a square (subject × subject) score matrix whose
**diagonal** is the personalized score and whose **off-diagonal** feeds a retrieval metric. Any off-by-one
in subject index produces plausible-looking but meaningless numbers.

**P3 — Metric fidelity.**
The metric suite has non-obvious contracts: durations in seconds inside the fixation vectors but
milliseconds inside ScanMatch; a fixed 16×12 spatial binning tied to a specific screen resolution;
short scanpaths padded to length 3 before MultiMatch; NaN elimination that silently changes the
denominator. These must be preserved, not "cleaned up".

---

## 3. Overall pipeline

The end-to-end path for **eval-only inference with released checkpoints** (the mode we are building; see
[Roadmap.md](Roadmap.md) for why training is out of scope):

```
                        ┌──────────────────────────────────────────┐
   OUR DATA             │ external repo: our dataset + its         │
   (foreign)            │ dataloader (raw gaze → fixations)        │
                        └───────────────────┬──────────────────────┘
                                            │  [STAGE A — bridge, TBD]
                                            ▼
                        ┌──────────────────────────────────────────┐
   CANONICAL            │ fixations.json                           │
   FORM                 │ [{name, subject, X, Y, T, length,        │
                        │   split, condition, task}, ...]          │
                        │ + stimulus images (.jpg)                 │
                        └───────────┬──────────────────┬───────────┘
                                    │                  │
                 [STAGE B]          │                  │  [STAGE C]
                 feature_extractor  ▼                  ▼  subject embeddings
                        ┌────────────────────┐  ┌──────────────────────────┐
   PRECOMPUTED          │ image_features/    │  │ SE-Net forward pass over │
   INPUTS               │   <img>.pth        │  │ a support set → user     │
                        │ embeddings.npy     │  │ embedding .pt            │
                        │ (task text emb.)   │  │ (or reuse released .pt)  │
                        └─────────┬──────────┘  └────────────┬─────────────┘
                                  │                          │
                                  └────────────┬─────────────┘
                                               ▼  [STAGE D — inference]
                        ┌──────────────────────────────────────────┐
                        │ ISP / Gazeformer (checkpoint_best.pth)   │
                        │ src/test.py → per-image, per-subject     │
                        │ action-probability maps + log-normal     │
                        │ duration params → Sampling.generate_     │
                        │ scanpath()                               │
                        └───────────────────┬──────────────────────┘
                                            ▼  [STAGE E — evaluation]
                        ┌──────────────────────────────────────────┐
                        │ comprehensive_evaluation_by_subject()    │
                        │  MultiMatch(5) · ScanMatch(w/ + w/o dur) │
                        │  SED · STDE · retrieval (MRR, R@1/3/5)   │
                        └───────────────────┬──────────────────────┘
                                            ▼
                        result/<run>/log/{log_test_*.txt, prediction.json}
```

Stage A is the only stage we author from scratch. Stages B–E already exist and are to be *configured*,
not rewritten. Stage C, in eval-only mode, may reduce to loading a released `*_user_embedding.pt` — but
that embedding was learned for the *original* subjects, not ours, which is the central open scientific
question recorded in [Roadmap.md](Roadmap.md).

---

## 4. Desired behaviours

These are the invariants any spec, plan, or implementation in this repo must satisfy. They are numbered so
specs can cite them (e.g. "satisfies D3").

### D1 — Upstream metric code is read-only
`src/utils/evaluation.py`, `src/utils/evaltools/scanmatch.py`, and
`src/utils/evaltools/visual_attention_metrics.py` are **frozen**. No edits, no "fixes", no refactors, no
vectorisation. If a call site needs different behaviour, change the call site or wrap the function — never
the function. Any deviation must be recorded explicitly in the spec that introduces it and justified
against comparability to the published numbers.

### D2 — One canonical data form
Everything downstream of Stage A sees only `fixations.json` + `image_features/*.pth` + `embeddings.npy`.
Our dataset's native format never leaks past the bridge. This keeps the blast radius of P1 to a single
module and lets us diff our JSON against the shipped OSIE `fixations.json` as a correctness check.

### D3 — Units and coordinate spaces are declared and asserted
- `X`, `Y` in `fixations.json` are in **original stimulus pixel coordinates** (OSIE: 800×600), origin
  top-left, 1-indexed (the loader subtracts 1 before binning).
- `T` in `fixations.json` is in **milliseconds** (integer).
- Fixation vectors handed to the metrics are structured arrays
  `dtype=[('start_x','f8'),('start_y','f8'),('duration','f8')]` with coordinates in the **resized**
  512×384 space and duration in **seconds**; `evaluation.py` multiplies by 1000 itself for ScanMatch.
- Any new dataset branch must pass `origin_size=(H, W)` explicitly. Note that
  `ISP/OSIE/GazeformerISP/src/test.py` does *not* pass it today and relies on the `OSIE_evaluation`
  default `(600, 800)`; `args.origin_width` / `args.origin_height` are parsed but unused on that path.
  A dataset with any other stimulus size that does not fix this will be silently mis-scaled.

### D4 — Subject identity is preserved end-to-end
Row index = predicted subject, column index = ground-truth subject, and the **diagonal is the reported
score**. `select_fewshot_subject()` remaps our original subject ids to a dense `0..N-1` range; the inverse
map must be applied before anything is written to `prediction.json` (see `recover_subject_ids()` /
`get_prediction_list()`). Every run must be able to answer "which of my real subjects is row 3?".

### D5 — Runs are reproducible and self-describing
Seeds are set for numpy, torch, and cudnn (`deterministic=True`, `benchmark=False`). Every run writes a
log containing the full resolved argument namespace, and a `prediction.json` of the generated scanpaths in
the same schema as `fixations.json`, so any reported metric can be recomputed offline from artefacts alone
without a GPU.

### D6 — Report the same headline numbers as the paper
`SM` = harmonic mean of ScanMatch with/without duration; `MM` = arithmetic mean of the five MultiMatch
dimensions (vector, direction, length, position, duration); `SED`; `STDE`; plus the ScanMatch-with-duration
retrieval block (MRR, R@1, R@3, R@5). Means **and** standard deviations, since the evaluator already
returns both.

### D7 — Fail loudly on shape or coverage mismatch
Silent degradation is the enemy: a missing `.pth` feature file, a subject with zero scanpaths on a split,
a scanpath shorter than 3 fixations, an image present in `fixations.json` but absent from the stimuli
directory. These must raise or be counted and reported — never be quietly dropped into the mean. Note the
evaluator's `is_eliminating_nan=True` default already drops NaN MultiMatch rows; the count of dropped rows
must be surfaced.

### D8 — Scope discipline
Eval-only. No training loop changes, no RL/policy-gradient path, no new model architecture, no metric
invention. If a task appears to require training, it stops and gets escalated to a roadmap decision rather
than being absorbed into the current work.

---

## 5. Definition of done for the mission

We are done when a single documented command, run on the GPU cluster, takes our dataset and produces a log
file reporting MultiMatch / ScanMatch / SED / STDE / retrieval for our subjects — computed by unmodified
upstream metric code — alongside a `prediction.json` that can be re-scored offline, and a written statement
of what the numbers do and do not license us to conclude, given that the checkpoint was trained on OSIE
subjects and not ours.
