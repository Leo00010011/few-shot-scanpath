# Roadmap

> Constitution file 3 of 3. Read together with [Mission.md](Mission.md) and [TechStack.md](TechStack.md).
> Last updated: 2026-09-08
>
> **Current phase: F1 — Reproduce the COCO-FreeView baseline.** Nothing else starts until F1 produces
> numbers. The preflight tooling and the run script are built and tested; the cluster run is pending.
>
> **Also live: OPEN-5.** F2's bridge is built and passing its own tests, but running it against
> the real EVE bundle showed the bundle cannot supply the subject/stimulus structure the ISP
> loader requires. F3, F4 and F5 are blocked on that until OPEN-5 is resolved — see §5.

---

## 1. Status board

| ID | Feature | Status | Blocked by |
|---|---|---|---|
| F0 | Constitution + spec workflow | ✓ DONE (2026-09-07) | — |
| F1 | Reproduce the **COCO-FreeView** eval baseline on the cluster | ◐ TOOLING BUILT (2026-09-08), awaiting cluster run | GPU allocation |
| F2 | Dataset bridge: EVE → `fixations.json` + GT heatmaps | ◐ BUILT (2026-09-08), blocked on data | **OPEN-5** |
| F3 | Subject embeddings for our subjects | ⏸ TODO | F2, **OPEN-2**, **OPEN-5** |
| F4 | Feature extraction for our stimuli | ⏸ TODO | F2, **OPEN-5**, **OPEN-6** |
| F5 | Our-dataset eval branch + run script | ⏸ TODO | F2, F3, F4, **OPEN-5** |
| F6 | Offline re-scorer (`prediction.json` → metrics) | ⏸ TODO | F1 |
| F7 | Results write-up + validity statement | ⏸ TODO | F5, F6 |

Legend: ✓ DONE · ◐ CODE COMPLETE but blocked · ▶ IN PROGRESS/NEXT · ⏸ TODO · ✗ DROPPED

---

## 2. Dependency graph

```
        F0 constitution
             │
             ▼
     F1 COCO-FV baseline ──────────────┐
             │                         │
             │  (proves env + metrics) │
             ▼                         ▼
   OPEN-1 ─► F2 data bridge ✓      F6 offline re-scorer
             │                         │
             ▼                         │
        OPEN-5 (blocking) ◄── the bundle cannot feed the loader as-is
             │                         │
      ┌──────┴────────┐                │
      ▼               ▼                │
 F4 features    OPEN-2 ─► F3 subj emb  │
   ▲  │               │                │
OPEN-6└───────┬───────┘                │
              ▼                        │
        F5 eval branch ────────────────┤
              │                        │
              └────────────┬───────────┘
                           ▼
                   F7 write-up
```

---

## 3. Features

### F1 — Reproduce the COCO-FreeView eval baseline on the cluster ◐ TOOLING BUILT

Spec: [`spec/2026-09-08-cocofv-baseline-on-cluster/`](../2026-09-08-cocofv-baseline-on-cluster/)
(requirements · plan · validation). Satisfies D1, D5, D6, D7, D8.

**Why first:** it is the only way to prove the environment, the checkpoints, and the frozen metric code all
work *before* our data is in the picture. Every later failure then has an unambiguous cause.

**Why COCO-FreeView and not OSIE:** its query set is 3 unseen subjects — the same order of magnitude as
the cohort OPEN-5 currently permits on the EVE bundle (2) — so if the retrieval block degenerates at
small `subject_num` we learn it here, on data whose correct answer is published. Its stimuli are
photographic COCO scenes, closer to EVE's. And its `test.py` carries **no** `if i_batch > 100: break`
cap (verified), so the "101 images or the full split?" question that hung over the OSIE checklist does
not arise at all.

Built 2026-09-08 (CPU, Windows dev machine):

- [x] `tools/cocofv_prep/` — `normalize_fixations.py` (FR3.3), `check_fixations.py` (the equal-subject
      invariant, FR3.5c — the check that stops a ragged image silently misaligning every subsequent
      image against the wrong ground truth), `check_features.py` (FR4.6). 31/31 pytest tests pass.
- [x] `bash/test_cocofv.sh` — the single `sbatch` script, all tunables overridable from the environment.
- [x] FR10.7 settled without a GPU: `evaltools/scanmatch.py` **and**
      `evaltools/visual_attention_metrics.py` are both byte-identical between the OSIE and COCO_FV
      branches. Only `evaluation.py` has drifted — see [TechStack.md](TechStack.md) §4.2.
- [x] FR5.4 settled: the checked-in `ISP/COCO_FV/.../src/embeddings.npy` already satisfies FR5.2
      (`{"free-viewing": (768,) float32}`), so the run copies it and needs no hub access from the
      compute node.

Remaining, all cluster-only:

- [ ] Build the `isp` env; record deviations from the [TechStack.md](TechStack.md) §1 pin list, and
      enforce `multimatch-gaze == 0.1.3` (FR1.3).
- [ ] Verify the on-disk case of `FV-ex-012` and `fewshot_user_embedding_10.pt`; symlink the checkpoint
      and the subject embedding (FR6).
- [ ] Confirm the checkpoint's state dict carries no `subject_embed.weight` key (FR6.6).
- [ ] Run Stage B, then the query-set evaluation at `--seed 0 1 2` (FR13.3).
- [ ] Write `notes.md`: environment, staged data, preflight counters, the metric table vs the paper's
      COCO-FreeView row, the denominators, and the verdict.

**Done when:** SM / MM / SED land within the three-seed noise band of the published COCO-FreeView row,
and the exact command + env are recorded in `notes.md`.

<details>
<summary><s>Superseded: the OSIE formulation of F1</s></summary>

The OSIE baseline run is **dropped, not deferred** — the spec above replaces it outright, for the
reasons in "Why COCO-FreeView and not OSIE". OSIE stimulus images are consequently no longer an
external dependency for F1. **`ISP/OSIE/GazeformerISP/` remains F5's template branch** (see
[TechStack.md](TechStack.md) §2): F1's *baseline* branch and F5's *template* branch are deliberately
different things.

- ~~Stage the OSIE stimuli (800×600 `.jpg`) into the path `--img_dir` points at.~~
- ~~Place `weights/OSIE-*/OSIE/checkpoint_best.pth` at `.../OSIE-ex-10to15/checkpoints/`.~~
- ~~`CUDA_VISIBLE_DEVICES=0 python src/test.py --fewshot_subject 10 11 12 13 14`~~
- ~~Decide what to do about the `if i_batch > 100: break` cap in the OSIE `test.py`.~~ Moot: the
  COCO_FV `test.py` has no such cap. The OSIE cap is still documented in TechStack §5 because F5's
  branch mirrors OSIE and must remove or parameterise it.
- ~~Confirm the shipped `result/git-osie-useremb-ex-10to15/log/prediction.json` matches F6's schema.~~
  F6's fixture is now a COCO_FV artefact instead — with a **different** schema (FR12.2).

</details>

---

### F2 — Dataset bridge: EVE → `fixations.json` + GT heatmaps ◐ BUILT, BLOCKED ON DATA

Spec: [`spec/2026-09-08-eve-bridge-and-gt-heatmaps/`](../2026-09-08-eve-bridge-and-gt-heatmaps/)
(requirements · plan · validation · **notes** — the notes file carries the findings below in full).
Implements Stage A; satisfies D2, D3, D4, D7.

- [x] Resolve OPEN-1 → **candidate 1, the converter**. Code lives in a new top-level
      `tools/eve_bridge/` package, deliberately outside the ISP tree so
      `ISP/OSIE/GazeformerISP/`'s import surface stays identical to upstream.
- [x] Converter producing `fixations.json` conforming to [TechStack.md](TechStack.md) §3.1.
- [x] Explicit `subject_id_map.json` (real EVE participant ids ↔ dense `0..N-1`) — D4.
- [x] `origin_size=(1080, 1920)` declared and written to `bridge_report.json` and to the
      heatmap-store attrs, so F5 passes it explicitly instead of inheriting the
      `OSIE_evaluation` default `(600, 800)` (D3).
- [x] Validator (Windows/CPU) enforcing all 11 invariants, plus a `bridge_report.json`
      carrying every drop counter — D7.
- [x] `split` assignment defined: **disjoint by stimulus**, `train` = support pool,
      `test` = query pool. Rationale in the spec (partially answers OPEN-3).
- [x] Ground-truth per-timestep 24×32 heatmaps (`GtHeatmapStore`, `gt_heatmaps.h5`) and a
      `score_step_heatmaps()` NSS/CC/KLD wrapper — the substrate F5's heatmap-metric block needs.
      EVE ships no saliency maps, so the ground truth is derived from the scanpath exactly as
      `OSIE.__getitem__` does, pinned bitwise by a parity test.
- [x] 74/74 pytest tests pass on Windows CPU; frozen files verified untouched (D1).
- [ ] **Blocked: OPEN-5.** The bridge runs, but the bundle cannot supply ≥3 subjects sharing
      ≥21 stimuli, so no scientifically usable configuration exists yet.
- [ ] Re-run the bitwise-parity test under the `isp` env's numpy 1.23.5 before F5 trusts the
      heatmap cache (it was validated on the dev machine's numpy 2.1.2 — TechStack §1).

**Done when:** OPEN-5 is resolved and a real run produces a `fixations.json` +
`gt_heatmaps.h5` for a subject/stimulus configuration F5 can actually score.

---

### F3 — Subject embeddings for our subjects ⏸

Depends on **OPEN-2** and **OPEN-5**. Implements Stage C.

- [ ] Resolve OPEN-5 — until the query cohort is bigger than 2 participants there is nothing worth
      embedding.
- [ ] Resolve OPEN-2 (whose subject embeddings do our subjects get?).
- [ ] Feed SE-Net the **`train`-split** trials only. F2 guarantees that pool is disjoint by stimulus
      from the scored `test` split, which is what makes the few-shot embedding legitimate; reading a
      `test` record into the support set silently invalidates every cell of the score matrix.
- [ ] If generating our own: build the `senet` env including Detectron2 + MSDeformAttn (the highest-risk
      install in the project — [TechStack.md](TechStack.md) §1), add a config under `SE-Net/configs/`, run
      `train.py --eval-only --fewshot_subject ...`.
- [ ] If reusing a released embedding: document precisely which subjects' embeddings are being borrowed
      and what that does to the interpretation of the numbers (feeds F7).
- [ ] Verify the produced tensor's shape matches `args.subject_feature_dim = 384` and its row order matches
      the dense subject indices from F2.

---

### F4 — Feature extraction for our stimuli ⏸

Depends on **OPEN-5** and **OPEN-6**.

- [ ] Resolve OPEN-6 — one `.pth` per `stimulus_name` cannot represent two different renderings of the
      same image, and the bridge reports the conflict rather than choosing for us.
- [ ] Point `feature_extractor.image_data()` at `<out_dir>/stimuli/` (note the hardcoded
      `<dataset_path>/train/` subdirectory) and produce one `.pth` per image.
- [ ] Confirm each tensor is `(768, 2048)`. Our 1920×1080 stimuli are **squashed**, not letterboxed,
      by the 768×1024 resize — 16:9 into 4:3 (OPEN-4). Record it; it feeds F7.
- [ ] Generate `embeddings.npy` containing the `"free-viewing"` key.
- [ ] No further `jpg`-substring check is needed: the bridge already raises on any `stimulus_name`
      containing `jpg` ([TechStack.md](TechStack.md) §3.2), and the validator re-checks it on the
      written artefact.

---

### F5 — EVE eval branch + run script ⏸

Depends on **OPEN-5**. Implements Stage D+E for our data; satisfies D1, D3, D4, D5, D8.

- [ ] Create `ISP/EVE/GazeformerISP/` mirroring the OSIE tree, with `utils/evaluation.py` and
      `utils/evaltools/*` **copied verbatim** (D1). The branch consumes the bridge's *artefacts*, not
      its code — the one exception is `tools/eve_bridge/heatmap_metrics.score_step_heatmaps()`, which
      F5 imports directly.
- [ ] Adapt only: `dataset/dataset.py` (class name, `origin_size`, subject handling) and `src/test.py`
      (pass `origin_size=(1080, 1920)` explicitly — the default `(600, 800)` would silently mis-scale
      every coordinate — set `--subject_num` to our unseen-subject count, remove or parameterise the
      `i_batch > 100` cap).
- [ ] Assert `num_fewshot <= support_pool_size` from `bridge_report.json` (OPEN-3).
- [ ] Add the heatmap-metric block: `GtHeatmapStore.get_batch()` → `score_step_heatmaps()` against the
      model's `all_actions_prob`. Report NSS/CC/KLD **separately** from the scanpath metrics and say so:
      they average over valid *timesteps*, not over (image, subject) cells. Not the same denominator.
- [ ] Add a single documented run script under `bash/` that takes our data end-to-end.
- [ ] Verify `get_prediction_list()` / `recover_subject_ids()` write our *real* EVE participant ids
      into `prediction.json`, using `subject_id_map.json`'s `to_eve` (D4).

---

### F6 — Offline re-scorer ⏸

A small CPU-only tool that reads a `fixations.json` (ground truth) + a `prediction.json` and calls the
frozen `comprehensive_evaluation_by_subject()`. Lets us re-derive every reported number on a laptop from
artefacts alone, and is the natural test harness for F1/F5 (D5).

- [ ] Build the `[image][subject]` nested structured-array inputs, in the resized 512×384 space with
      durations in **seconds** ([TechStack.md](TechStack.md) §4).
- [ ] Reproduce F1's numbers from F1's artefacts as the correctness test.
- [ ] Report the NaN-dropped row count alongside every mean (D7).
- [ ] **Handle the COCO_FV `prediction.json` schema.** F1's fixture is now a COCO_FV artefact, not an
      OSIE one, and its schema differs from [TechStack.md](TechStack.md) §3.5 in four ways (F1 spec
      FR12.2): `name` is the **bare filename** while the loader keys on `"{task}/{name}"`; a `task`
      key is present; a `length` key is present; and `T` is `duration * 1000` as a **float**, not
      `int(round(t * 1000, 3))`, with `X`/`Y` also left as floats. The re-scorer must key on
      `(name, task, subject)`.
- [ ] Use the COCO_FV `evaluation.py`, not the OSIE one, when re-scoring a COCO_FV run — the two have
      drifted (TechStack §4.2) and the published COCO-FreeView numbers came from the COCO_FV file.

---

### F7 — Results write-up + validity statement ⏸

- [ ] Table of MultiMatch (5 dims) / ScanMatch (w/, w/o) / SED / STDE / retrieval, mean ± std, ours vs the
      paper's OSIE row. Plus the NSS/CC/KLD heatmap block, flagged as a per-timestep mean.
- [ ] Explicit statement of what the comparison licenses, given the domain gap between the checkpoint's
      training data and our stimuli/subjects, and given the OPEN-2 decision.
- [ ] State the **squash** (OPEN-4): our stimuli are distorted 16:9 → 4:3 so the metric parameters stay
      bit-identical to the published configuration. Say which of the two was traded for the other.
- [ ] State the **cohort** the OPEN-5 decision produced — how many participants, how many stimuli, and
      whether the retrieval block is reportable at all at that size.
- [ ] Carry `bridge_report.json`'s counters into the write-up: `short_scanpath` (padded inside the
      frozen evaluator, D7), `clamped_coords`, `incomplete_stimulus`, `stimulus_image_conflict`.

---

## 4. Explicitly out of scope

Recorded so they are not silently re-litigated (D8). Reopening any of these is a constitution change.

- ✗ **Training or fine-tuning** ISP or SE-Net on our dataset. Eval-only with released checkpoints.
- ✗ The **RL / policy-gradient** path (`OSIE_rl`, `--start_rl_epoch`).
- ✗ The **COCO-Search18** branch, and anything task/search-conditioned.
  *(Amended 2026-09-08 by [F1's spec](../2026-09-08-cocofv-baseline-on-cluster/requirements.md) FR16.1:
  **COCO-FreeView** was struck from this bullet. It is free-viewing, not search-conditioned, and it is
  now the F1 baseline branch.)*
- ✗ **Modifying, re-implementing, or "improving"** any metric (D1).
- ✗ New metrics not in the paper's suite.
- ✗ Making the repo run on Windows GPUs.

---

## 5. Open decisions

These block specific features. Each must be resolved by the user before its dependent feature starts.

### ~~OPEN-1~~ — How does our dataset get into this repo? ✓ RESOLVED 2026-09-08
**Candidate 1, the converter.** `tools/eve_bridge/` reads the EVE bundle through the installed
`evedataset` wheel and writes `fixations.json` + `stimuli/*.jpg` + `subject_id_map.json` +
`gt_heatmaps.h5` + `bridge_report.json`. Upstream `Dataset` classes are untouched, the output is
diffable against the shipped OSIE JSON (D2), and the whole thing runs on Windows CPU. The
`evedataset` dependency stays outside the ISP tree so `ISP/OSIE/GazeformerISP/`'s import surface
is unchanged.

### OPEN-2 — Whose subject embeddings do our subjects get? *(blocks F3, colours F7)*
The released `fewshot_user_embedding_10.pt` encodes *OSIE subjects 10–14*, not ours. Options:
1. Run SE-Net over a support set of our subjects' real scanpaths to produce genuine embeddings for them
   — scientifically correct, but requires the Detectron2 + MSDeformAttn install.
2. Reuse a released embedding as a stand-in — cheap, but the "personalization" being measured is then
   somebody else's, and F7 must say so plainly.

### OPEN-3 — Support/query split for our subjects ◐ PARTIALLY RESOLVED 2026-09-08 *(still colours F5)*
Answered by F2:
- The split is **by stimulus, not by trial** — `train` = support pool, `test` = query pool, name sets
  disjoint by construction, asserted in memory *and* re-asserted on the written artefact. This is the
  only partition under which "the support scanpaths were never scored" holds for all N × N cells of
  the evaluator's matrix; splitting by trial leaks across the stimulus axis.
- `support_pool_size = 20`, deliberately larger than the paper's `num_fewshot = 10`, so
  `--random_support` repeats draw *varying* support sets that stay inside the pool. With
  `support_pool_size == num_fewshot` every seed draws the same set and averaging measures nothing.
  F5 must assert `num_fewshot <= support_pool_size`; the value is in `bridge_report.json`.

Still open for F5: how many `--random_support` repeats to average over, and — pending OPEN-5 — the
actual `--subject_num`, which the bundle currently caps at 2.

### ~~OPEN-4~~ — Stimulus resolution and resize policy ✓ RESOLVED 2026-09-08: **squash**
EVE is 1920×1080 (16:9), OSIE is 4:3, and the frozen metric configuration is welded to a 512×384
screen with a 16×12 ScanMatch grid (TechStack §4). Letterboxing would keep the grid but hand ~25 % of
the bins to dead bars that still count in ScanMatch's string quantisation; changing `args.width/height`
would change the bin *shape* and forfeit comparability to the published table. So: **squash**.

The bridge rescales nothing — it writes X/Y in native 1920×1080 and declares `origin_size=(1080, 1920)`.
The squash happens inside `OSIE_evaluation`'s `resizescale_x/y` = 3.75 / 2.8125 (non-uniform), which
F5 hits once it passes `origin_size` explicitly. Every metric parameter stays bit-identical to the
published configuration; the stimulus distortion is recorded in `bridge_report.json` and is F7's to
state plainly.

### OPEN-5 — The EVE bundle cannot feed the ISP loader as-is *(blocks F3, F4, F5)* ◀ NEW, BLOCKING
Raised 2026-09-08 by running the built bridge against `eve_shared/EveDataset/bundle`
(3096 samples, 54 participants). Two facts, both properties of the bundle rather than of the code —
the bridge fails loudly and counts the drops exactly as D7 requires:

1. **Every `test*` participant has `valid == False` on every trial** (train 1951/2238 valid,
   val 185/261, **test 0/597**). The natural query cohort is unusable; query subjects must come from
   `train*` / `val*`, or the bundle must be re-exported with usable test-split labels.
2. **Participants see near-disjoint stimulus sets.** The ISP loader requires every image to yield the
   same number of subjects (TechStack §3.1) or `evaluation.py`'s (subject × subject) matrix goes
   ragged. Exhaustive search over the 39 participants with any valid trial:

   | n subjects | max shared stimuli | best group |
   |---|---|---|
   | 2 | **13** | `train02`, `train09` |
   | 3 | 4 | `train06`, `train09`, `train10` |
   | 4 | 2 | `train06`, `train09`, `train10`, `train37` |

   No stimulus is shared by more than 4 participants anywhere in the bundle. `support_pool_size = 20`
   needs ≥ 21 shared stimuli. On the best feasible configuration the equal-subject filter discards
   91 of 104 candidate stimuli.

A 2-subject evaluation makes the retrieval block (MRR, R@1/3/5) degenerate and the personalization
claim untestable, so this must be decided before F3/F4/F5. The options:

1. **Re-export the bundle** so participants share a common image set — the EVE protocol may have a
   shared subset this export did not preserve. *(Preferred if it exists: costs no comparability.)*
2. **Relax the equal-subject invariant**, which means adapting `OSIE_evaluation` and the evaluator's
   matrix construction — i.e. touching frozen code (D1). Not free, and needs its own justification
   against comparability to the published numbers.
3. **Accept 2 subjects**, report only the diagonal metrics and drop the retrieval block. F7 would
   have to state plainly what that does and does not license.

The frontier table above is reproduced by
`py -m pytest tests/eve_bridge -m bundle --bundle-dir <dir> -s -k feasibility`, so it can be
re-derived cheaply after any bundle re-export.

### OPEN-6 — The same `stimulus_name` renders differently per participant *(blocks F4)* ◀ NEW
Raised 2026-09-08 alongside OPEN-5. On the 13 stimuli shared by the best 2-subject group,
`stimulus_image_conflict == 13` — i.e. **all** of them. The images are all 1920×1080 RGB and open
cleanly, but the two participants' renderings differ substantially: mean absolute difference ≈ 9–29
per channel, 24–72 % of pixels differing, whole-image correlation ≈ 0.70 on the case inspected.
Recognisably the same photograph, differently rendered — a brightness/scale/crop difference, not
encoding noise.

The bridge keeps the first exp_key's image in sorted order and counts the collision. But F4 produces
one `.pth` per `stimulus_name`, which cannot represent two renderings — so the participant whose
rendering was discarded gets scored against features of an image they did not see. Decide before F4
whether to pick one rendering, key features per (stimulus, participant), or treat the divergence as a
defect in the bundle export.

---

## 6. External dependencies to obtain

| item | source | needed for |
|---|---|---|
| ISP-SENet checkpoints | already in `weights/` (git-ignored) | F1, F5 |
| COCO-FreeView stimuli (`<category>/*.jpg`) | already staged on the cluster (`$FV_IMAGE_ROOT`) | F1 |
| COCO-FreeView fixation labels | already staged on the cluster (`$FV_FIX_JSON`) | F1 |
| ~~OSIE stimulus images~~ | ~~NUS-VIP `predicting-human-gaze-beyond-pixels` repo~~ | ~~F1~~ — no longer needed; F1 is the COCO-FreeView baseline |
| Detectron2 | source install, per HAT repo | F3 (option 1 only) |
| MSDeformAttn | `SE-Net/src/pixel_decoder/ops/make.sh` | F3 (option 1 only) |
| `stsb-roberta-base-v2` | sentence-transformers hub | F1, F4 |
| Mask R-CNN R50-FPN COCO weights | torchvision download | F1, F4 |
| `evedataset` wheel + `bundle.h5` | `eve_shared/EveDataset/` (installed, git-ignored) | F2 |
| cluster allocation with an NVIDIA GPU | — | F1, F3, F4, F5 |
