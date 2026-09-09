# Roadmap

> Constitution file 3 of 3. Read together with [Mission.md](Mission.md) and [TechStack.md](TechStack.md).
> Last updated: 2026-09-09
>
> **Current phase: F1 — Reproduce the OSIE baseline. FIRST GREEN RUN 2026-09-09.** The pipeline ran
> end to end on the cluster and the metrics came back consistent with the published OSIE row. What is
> left is the seed sweep and the write-up, not the plumbing. **F6 is unblocked.**
>
> **Reverted 2026-09-09.** F1 was briefly re-pointed at COCO-FreeView (commit `ce6b5dd`). That is undone:
> the COCO-FreeView *test* split is a held-out challenge benchmark with no public labels, so the run is
> not reproducible by anyone outside the challenge and cannot serve as the project's environment proof.
> See §4 — COCO-FreeView is back out of scope, permanently and with a reason attached.
>
> **Also live: OPEN-5.** F2's bridge is built and passing its own tests, but running it against
> the real EVE bundle showed the bundle cannot supply the subject/stimulus structure the ISP
> loader requires. F3, F4 and F5 are blocked on that until OPEN-5 is resolved — see §5.

---

## 1. Status board

| ID | Feature | Status | Blocked by |
|---|---|---|---|
| F0 | Constitution + spec workflow | ✓ DONE (2026-09-07) | — |
| F1 | Reproduce the **OSIE** eval baseline on the cluster | ◐ FIRST RUN GREEN (2026-09-09); sweep tooling done, sweep run + `notes.md` outstanding | — |
| F2 | Dataset bridge: EVE → `fixations.json` + GT heatmaps | ◐ BUILT (2026-09-08), blocked on data | **OPEN-5** |
| F3 | Subject embeddings for our subjects | ⏸ TODO | F2, **OPEN-2**, **OPEN-5** |
| F4 | Feature extraction for our stimuli | ⏸ TODO | F2, **OPEN-5**, **OPEN-6** |
| F5 | Our-dataset eval branch + run script | ⏸ TODO | F2, F3, F4, **OPEN-5** |
| F6 | Offline re-scorer (`prediction.json` → metrics) | ▶ **UNBLOCKED** (2026-09-09) — F1 produced its fixture | — |
| F7 | Results write-up + validity statement | ⏸ TODO | F5, F6 |

Legend: ✓ DONE · ◐ CODE COMPLETE but blocked · ▶ IN PROGRESS/NEXT · ⏸ TODO · ✗ DROPPED

---

## 2. Dependency graph

```
        F0 constitution
             │
             ▼
     F1 OSIE baseline ────────────────┐
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

### F1 — Reproduce the OSIE eval baseline on the cluster ◐ FIRST RUN GREEN

Spec: [`spec/2026-09-08-osie-eval-baseline/`](../2026-09-08-osie-eval-baseline/)
(requirements · plan · validation). Satisfies D1, D5, D6, D7, D8.

**Why first:** it is the only way to prove the environment, the checkpoints, and the frozen metric code all
work *before* our data is in the picture. Every later failure then has an unambiguous cause.

**Why OSIE and not COCO-FreeView** *(settled 2026-09-09, after a one-day detour through COCO-FreeView)*:
the COCO-FreeView **test split is a held-out challenge benchmark** — the public label file carries
`train`/`valid` only. A "baseline" computed on it is either not the published number or not reproducible
by anyone outside the challenge, and F1's entire job is to be a reproducible environment proof. OSIE ships
its test labels (`data/osie_fixations_update_duration.json`, 70 images × 15 subjects), ships an example
`prediction.json` to diff against, and is the branch F5 mirrors anyway — so F1 and F5 exercise the same
loader, the same `data_postprocess.py`, and the same `evaluation.py`. The COCO-FreeView detour was not
wasted: its spec is retained as superseded and its FR10 table documents six real divergences in
`ISP/COCO_FV/.../evaluation.py` that F6/F7 still need ([TechStack.md](TechStack.md) §4.2).

Built 2026-09-09 (CPU, Windows dev machine), retargeted from the COCO-FreeView tooling:

- [x] `tools/osie_prep/` — `check_fixations.py` (the equal-subject invariant, FR3.5c — the check that
      stops a ragged image silently misaligning every subsequent image against the wrong ground truth)
      and `check_features.py` (FR4.6). No `normalize_fixations.py`: OSIE's shipped
      `data/osie_fixations_update_duration.json` is already canonical (§3.1 schema, `train`/`validation`/
      `test`, `condition="freeview"`, `task="none"`), so there is nothing to normalise.
- [x] `bash/test_osie.sh` — the single run script (run interactively under `salloc`; `sbatch` still
      works), all tunables overridable from the environment.
- [x] **The `i_batch > 100` cap is inert for OSIE — the roadmap's long-standing "decide and document"
      item is closed by observation, not by a judgement call.** `OSIE_evaluation.__len__` returns the
      number of **images**, not records: the `test` split is 70 images × 15 subjects = 1050 records, so
      the loader yields **70 batches at `--batch 1`** (18 at the default `--batch 4`) and the 101-batch
      cap never fires. The shipped `prediction.json` holds 350 records = 70 images × 5 subjects,
      independently confirming the published number covers the **full** test split. `test.py` is left
      unmodified for F1. The cap becomes live only in F5, where the roadmap already flags it.
- [x] **The `(10, 384)` `fewshot_user_embedding_10.pt` is verified correct for a 5-subject query set.**
      `select_fewshot_subject()` remaps `--fewshot_subject 10 11 12 13 14` to a dense `0..4`, and
      `gazeformer.py` indexes `self.subject_embed[subjects]` with that. Checked on the dev machine: rows
      **5–9 are exactly zero** and rows 0–4 carry the five query subjects. The `10` in the filename is
      `num_fewshot` (10-shot), not a subject count; the tensor is allocated at base-set width with only
      the query rows filled. No cluster check needed.
- [x] `evaltools/scanmatch.py` and `evaltools/visual_attention_metrics.py` are byte-identical between the
      OSIE and COCO_FV branches; only `evaluation.py` has drifted. F1 uses the **OSIE** `evaluation.py`.
- [x] `src/data/embeddings.npy` and `src/data/fixations.json` already ship in the OSIE branch, so Stage B
      reduces to `image_data()`; `text_data()` need not run (and its `__main__` is broken — `args.p` is
      undefined, so `image_data()` is called directly rather than running the module).

Cluster run, 2026-09-09:

- [x] **The `isp` env was never built — the run used a pre-existing env (`scanpath`) far newer than the
      pin list, and the metrics still came back consistent with the published row.** python 3.11.14,
      torch 2.10.0+cu126, torchvision 0.25.0+cu126, numpy **2.1.2**, scipy 1.14.1, plus `scikit-image`,
      `opencv-python` and `multimatch-gaze==0.1.3` installed into it. This is the single largest
      deviation in the project and it is now an *empirical* result rather than a risk — see
      [TechStack.md](TechStack.md) §1.1, which records what was verified and what was not.
- [x] **The pipeline ran end to end and the metrics came back consistent with the published OSIE row.**
      This is F1's whole purpose discharged: the environment, the released checkpoint, the released
      subject embedding, Stage B, the loader, and the frozen metric code are now known-good *as a
      system*, so any discrepancy F5 produces on EVE data is attributable to the data, not the setup.
- [x] Stimuli staged flat at `$PROJECT_DIR/data/stimuli` (700 × 800×600, `1001.jpg`–`1700.jpg`), under
      the git-ignored `data/`. Stage B only — `test.py` never opens an image.
- [x] On-disk case resolved by observation: `weights/OSIE-20260904T121550Z-1-001/OSIE/` carries
      `checkpoint_best.pth` and `fewshot_user_embedding_10.pt` — the spellings `test.py` expects. The
      READMEs' `osie-ex-10to15` / `subject_embedding` variants do not occur on disk.
- [x] FR6.6 confirmed: the checkpoint is `{"model": OrderedDict}` of 273 pure tensors, **no
      `subject_embed.weight` key** — the query subjects' embedding is genuinely the one in play.
- [x] **Sweep tooling complete (2026-09-09), sweep itself not yet submitted.** Three gaps that would
      have made a three-seed sweep unaggregatable are closed, all at the call site — `test.py` remains
      unmodified:
      - `bash/test_osie.sh` now tees `test.py`'s stdout into `log/seed$SEED/stdout.txt`. The headline
        `SM / MM / SED` is a bare `print()` and never reaches `log_test_subject_*.txt`; its only other
        home is `logs/osie_out_<jobid>.log`, named by **job id**, so three seeds produced three logs
        distinguishable only by submission order ([TechStack.md](TechStack.md) §3.5a).
      - the resolved-version dump is teed to `versions.txt` and preserved per seed, so D5's "record
        the stack" holds per run — the point [TechStack.md](TechStack.md) §1.1 insists on.
      - `tools/osie_prep/aggregate_seeds.py` (stdlib only, login-node runnable, 24 tests) pools the
        seed directories into `metrics_sweep.json` + `metrics_sweep.md`. It **raises** if a directory's
        seed disagrees with the seed in its logged arg namespace, or if any argument that changes what
        is measured differs between seeds — pooling non-replicates is the D7 failure mode here.
- [ ] Run the seed sweep at `--seed 0 1 2` (FR13.3) — needs the cluster:
      `bash bash/test_osie.sh`, then `SEED=1 bash ...` and `SEED=2 bash ...` under one `salloc`. Then run the aggregator once
      (command in [TechStack.md](TechStack.md) §1).
- [ ] Write `notes.md`: environment, staged data, preflight counters, **the actual metric table vs the
      paper's OSIE row**, the denominators, and the verdict. The numbers exist only in the run's log
      files so far; they are not yet recorded in the repo. Blocked on the sweep above.

**D6's standard deviations, settled 2026-09-09:** `test.py` stays unmodified, so `cur_metrics_std` is
still discarded. D6's std is supplied by the sweep's **across-seed** spread, and the **per-cell** std
is deferred to F6. These are different quantities and the write-up must not conflate them —
[TechStack.md](TechStack.md) §3.5b states which is which.

**Done when:** SM / MM / SED land within the three-seed noise band of the published OSIE row, and the
exact command + env are recorded in `notes.md`. *(First seed: consistent. Sweep and write-up pending.)*

**Note on `R@5`:** with `--subject_num 5` every rank lies in `{0..4}`, so `p2g`'s `r5` is identically
`100.0`. It is still reported (D6 names R@5) but must be annotated as structurally saturated.

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
- [ ] **Key on `(name, subject)` and use the OSIE `evaluation.py`.** F1's fixture is an OSIE artefact
      again as of 2026-09-09, so [TechStack.md](TechStack.md) §3.5 is the schema and `OSIE_evaluation`
      groups by the bare `name` — no `task` component in the key. The shipped
      `result/git-osie-useremb-ex-10to15/log/prediction.json` is the reference fixture.
- [ ] **Retained for later:** should F6 ever be pointed at a COCO_FV artefact, note that the two
      `evaluation.py` files have drifted (TechStack §4.2) and that the COCO_FV `prediction.json` schema
      differs in four ways — `name` is the bare filename while the loader keys on `"{task}/{name}"`,
      `task` and `length` keys are present, and `T`/`X`/`Y` are left as floats. Documented in the
      superseded [`cocofv-baseline-on-cluster`](../2026-09-08-cocofv-baseline-on-cluster/) spec, FR12.2.
      Not on F6's critical path.

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
- ✗ The **COCO-FreeView** and **COCO-Search18** branches, and anything task/search-conditioned.
  *(Amended 2026-09-08 to strike COCO-FreeView; **re-amended and closed 2026-09-09 — COCO-FreeView is
  back out of scope, permanently.** The reason is stronger than the original one and is recorded here so
  it is not re-proposed in six months: the **COCO-FreeView test split is a held-out challenge benchmark**.
  The public `COCOFreeView_fixations_trainval.json` carries `train`/`valid` only — there are no public
  test labels. Any number we compute on it is therefore either not the published number, or not
  reproducible by anyone outside the challenge. F1's whole purpose is a **reproducible** environment
  proof, so COCO-FreeView cannot serve as the F1 baseline no matter how attractive its 3-subject query
  set or its photographic stimuli are. The superseded spec is retained at
  [`spec/2026-09-08-cocofv-baseline-on-cluster/`](../2026-09-08-cocofv-baseline-on-cluster/) for its
  FR10 divergence table, which F6/F7 still need.)*
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
| ~~COCO-FreeView stimuli / fixation labels~~ | ~~staged on the cluster~~ | ~~F1~~ — dropped 2026-09-09, see §4 |
| ~~OSIE stimulus images (800×600 `.jpg`)~~ | NUS-VIP repo — **obtained 2026-09-09**, staged flat at `$PROJECT_DIR/data/stimuli` (git-ignored) | ~~F1~~ ✓ |
| Detectron2 | source install, per HAT repo | F3 (option 1 only) |
| MSDeformAttn | `SE-Net/src/pixel_decoder/ops/make.sh` | F3 (option 1 only) |
| ~~`stsb-roberta-base-v2`~~ | ~~sentence-transformers hub~~ — **not needed**: every branch ships `embeddings.npy`, `text_data()` is never called, and `bash/test_osie.sh` stubs the import (TechStack §1.1) | ~~F1~~, F4 only if regenerating |
| Mask R-CNN R50-FPN COCO weights | torchvision download — **obtained 2026-09-09** (Stage B ran) | ~~F1~~ ✓, F4 |
| `evedataset` wheel + `bundle.h5` | `eve_shared/EveDataset/` (installed, git-ignored) | F2 |
| cluster allocation with an NVIDIA GPU | — (F1 ran on `hpc-gpu1`, env `scanpath`) | ~~F1~~ ✓, F3, F4, F5 |
