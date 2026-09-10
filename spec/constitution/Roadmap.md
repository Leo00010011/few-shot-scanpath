# Roadmap

> Constitution file 3 of 3. Read together with [Mission.md](Mission.md) and [TechStack.md](TechStack.md).
> Last updated: 2026-09-09
>
> **F1 CLOSED 2026-09-09 — accepted with a flagged gap (OPEN-7).** The pipeline ran end to end, all
> three seeds are computed, and the run record is generated from artefacts: SM **0.3704 ± 0.0041**,
> MM **0.8010 ± 0.0019**, SED **7.3438 ± 0.0381**. Against the paper's n=10 ISP-SENet row
> (0.375 / 0.803 / 7.318) all three land slightly on the **worse** side, SM and MM just outside the
> three-seed band. Close, systematic, unexplained — **recorded as OPEN-7, not chased**: F1's job was to
> prove the environment, and it did.
>
> **F2 CLOSED 2026-09-10 — OPEN-5 resolved, artefacts built for a 38-participant cohort.**
> **38 participants · 354 scored images · 1062 scored cells · `--subject_num 3` · `num_fewshot 10`.**
> All validator invariants pass; every reported number is re-derivable from `bridge_report.json`.
>
> **OPEN-5's premise was wrong, and the error was ours.** The claim "the bundle caps us at 2
> subjects" came from applying the equal-subject rule to the whole run. What `evaluation.py`
> actually requires is a uniform subject **count per scored image** — its loops run over each
> image's *actual* length and its diagonal is **positional**, so subject identities may differ
> from image to image. The `-1`-initialised collectors reduced by a bare `np.mean()` with no
> `!= -1` filter are what force the uniform count; no metric requires a common cohort. And
> `args.subject_num` does **not** size the embedding table (`gazeformer.py` L100 is commented
> out), so a large cohort scores fine at `subject_num = 3`. EVE caps subjects **per image** at
> 2–4, not the cohort. See §5 OPEN-5 and the spec's `notes.md` §6.
>
> **F3, F4 and F5 are unblocked** (F4 still needs OPEN-6, now at 925 conflicts). F6 remains
> startable at any time.

> **Reverted 2026-09-09.** F1 was briefly re-pointed at COCO-FreeView (commit `ce6b5dd`). That is undone:
> the COCO-FreeView *test* split is a held-out challenge benchmark with no public labels, so the run is
> not reproducible by anyone outside the challenge and cannot serve as the project's environment proof.
> See §4 — COCO-FreeView is back out of scope, permanently and with a reason attached.
>
> **~~Also live: OPEN-5.~~** Resolved 2026-09-10 — see §5.

---

## 1. Status board

| ID | Feature | Status | Blocked by |
|---|---|---|---|
| F0 | Constitution + spec workflow | ✓ DONE (2026-09-07) | — |
| F1 | Reproduce the **OSIE** eval baseline on the cluster | ✓ **DONE** (2026-09-09) — accepted, gap flagged as **OPEN-7** | — |
| F2 | Dataset bridge: EVE → `fixations.json` + GT heatmaps | ✓ **DONE** (2026-09-10) — 38 subjects, 1062 scored cells | — |
| F3 | Subject embeddings for our subjects | ▶ **NEXT** | **OPEN-2** |
| F4 | Feature extraction for our stimuli | ⏸ TODO | **OPEN-6** |
| F5 | Our-dataset eval branch + run script | ⏸ TODO | F3, F4 |
| F6 | Offline re-scorer (`prediction.json` → metrics) | ▶ **NEXT** — the only unblocked feature; F1 produced its fixture | — |
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
        OPEN-5 ✓ resolved: 3 subjects/image, 38-participant cohort
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

### F1 — Reproduce the OSIE eval baseline on the cluster ✓ DONE (accepted; gap flagged as OPEN-7)

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
- [x] **Seed sweep run 2026-09-09 (FR13.3): seeds 0, 1, 2 all computed**, interactively under one
      `salloc` (`bash bash/test_osie.sh`, then `SEED=1 bash ...`, `SEED=2 bash ...`). Each seed's five
      artefacts are preserved under `result/OSIE-ex-10to15/log/seed<N>/` — `log_test_subject_10_0.txt`,
      `prediction.json`, `stdout.txt`, `versions.txt`, `preflight_fixations.json`. The whole sweep took
      ~12 minutes wall clock; Stage B was cached after seed 0.
      - Seed 0, verified self-consistent (composites re-derived from the logged means match the printed
        headline to within `.4f`-vs-`round(x,3)` formatting): MultiMatch `.9412 / .6504 / .9222 / .8443
        / .6566`, ScanMatch `.3798` w/o and `.3696` with duration, SED `7.3000`, STDE `.8460`,
        retrieval `pmrr .4661 / pr1 21.1429 / pr3 60.5714 / pr5 100.0000`. Headline **SM 0.375,
        MM 0.803, SED 7.3**. Compared against the paper's OSIE row by hand: close.
      - `--seed` is properly wired — `test.py` seeds numpy, torch and CUDA and pins
        `cudnn.deterministic=True` / `benchmark=False`, so the three runs draw genuinely different
        samples and each is individually reproducible (D5).
- [x] **Run record generated 2026-09-09** — `aggregate_seeds.py` pooled all three seeds with every
      guard passing silently: identical resolved stacks, identical preflight fingerprints, identical
      invariant args, each directory's seed matching its logged `seed`, and each seed's printed
      headline matching the composites re-derived from its own log. The three runs are replicates.

      | headline | mean ± across-seed std | min | max |
      |---|---|---|---|
      | SM | **0.3704 ± 0.0041** | 0.3664 | 0.3746 |
      | MM | **0.8010 ± 0.0019** | 0.7992 | 0.8029 |
      | SED | **7.3438 ± 0.0381** | 7.3000 | 7.3686 |

      Three things the numbers themselves raise, all of which belong in F7:
      - **Seed 0 is the top of the SM range**, not the centre: its `0.3746` is the sweep `max` against
        a mean of `0.3704`. The favourable hand-comparison against the paper was made on the single
        most optimistic seed. Compare the **mean** to the published row, not seed 0.
      - **`pr1` (R@1) is by far the noisiest metric**: `20.4762 ± 1.4094`, ranging 18.86–21.43 — a ~7 %
        relative spread, an order of magnitude worse than any other. Any claim about R@1 needs the band
        attached; a single-seed R@1 means very little at this cohort size.
      - **`n_short = 0` and `oob = 0`** on the scored split. No scanpath was padded to length 3 and no
        coordinate was out of frame, so both of those contamination routes (TechStack §4, D7) are
        confirmed absent for F1 rather than merely assumed.
- [x] **Published row transcribed 2026-09-09** into `paper_reference.json` — the paper's OSIE block
      carries exactly SM / MM / SED, so only those three have a published counterpart; every
      per-dimension MultiMatch value, both ScanMatch variants, STDE and the whole retrieval block are
      left `null` and render as `--` = UNKNOWN. The comparable row is **n = 10, ISP-SENet**, matching
      this run's `--num_fewshot 10` and `fewshot_user_embedding_10.pt`; the n = 1 and n = 5 rows are
      different configurations and must not be compared against.

      | | ours (3-seed mean) | ± seed | paper (n=10) | delta | within seed spread |
      |---|---|---|---|---|---|
      | SM ↑ | 0.3704 | 0.0041 | **0.375** | −0.0046 | **no** |
      | MM ↑ | 0.8010 | 0.0019 | **0.803** | −0.0020 | **no** |
      | SED ↓ | 7.3438 | 0.0381 | **7.318** | +0.0258 | yes |

      **The reproduction is close but not clean, and the shortfall is systematic rather than noise.**
      All three metrics land on the *worse* side of the published value — SM and MM lower, SED higher
      — and for SM and MM the paper's number sits just outside the full three-seed range
      (`0.375 > max 0.3746`; `0.803 > max 0.8029`). Three independent metrics offset the same way is
      the signature of a small systematic difference, not sampling scatter. In relative terms it is
      ≈ 1.2 % on SM, 0.25 % on MM, 0.35 % on SED.

      **Ruled out:** `--eval_repeat_num > 1` cannot be the explanation — it **raises IndexError** on
      this path, because `evaluation.py`'s collectors are shaped `(n_images, subject_num)` while
      `test.py` would hand it `eval_repeat_num * subject_num` rows ([TechStack.md](TechStack.md) §5).
      Do not spend GPU time on it. `--random_support` is likewise inert here: `select_fewshot_subject()`
      returns early when `split != 'train'`, so the query-set score does not depend on it.

      **Outstanding chore (not blocking):** `run_record.md` was generated *before* the reference was
      filled in, so its comparison section still shows `--` throughout. Re-run `aggregate_seeds.py`
      with `--reference` on the cluster to populate it. The numbers above are already correct; only
      the generated file lags.

      **Still open, for F7:** whether the residual comes from the env drift on the float metric paths
      (numpy 2.1.2 vs the pinned 1.24.3 — MultiMatch and STDE are exactly where §1.1 predicts drift
      would show), from the released checkpoint differing from the one behind the table, or from the
      paper's own number being an average over something not reproduced here. **This is a question to
      state honestly in the write-up, not to resolve by tuning.**
- [ ] Transcribe the paper's OSIE row into
      [`spec/2026-09-08-osie-eval-baseline/paper_reference.json`](../2026-09-08-osie-eval-baseline/paper_reference.json)
      — the template ships with every value `null` and its provenance fields blank. This is the
      **only** hand-entered number in the F1 record, because nothing in the repo holds the published
      table (`result-images/main-result.png` is qualitative, the READMEs carry none). Leave anything
      you cannot read off the paper as `null`: a `null` renders as `--` meaning *unknown*, and a
      guessed value would silently turn a real discrepancy into an apparent match.
- [ ] Write F7's **verdict** — what the comparison licenses given the checkpoint was trained on OSIE
      subjects 0–9 and scored on 10–14. This is an argument, not a measurement, so it is prose and
      lives in the spec, not in the generated record.

**There is deliberately no `notes.md`.** *(Decided 2026-09-09.)* A hand-written file of copy-pasted
numbers drifts from the artefacts the moment anything is re-run, which is exactly what D5 exists to
prevent. The run record is **generated** instead: `aggregate_seeds.py --report` reads the stored
outputs — `log_test_subject_*.txt`, `stdout.txt`, `versions.txt`, `preflight_fixations.json` — and
emits environment, denominators, the D6 table, and the paper comparison, every time, from artefacts
alone. The two things it cannot derive (the published row and the verdict) are named as such in the
output rather than quietly omitted. Regenerate it; never hand-edit it.

**D6's standard deviations, settled 2026-09-09:** `test.py` stays unmodified, so `cur_metrics_std` is
still discarded. D6's std is supplied by the sweep's **across-seed** spread, and the **per-cell** std
is deferred to F6. These are different quantities and the write-up must not conflate them —
[TechStack.md](TechStack.md) §3.5b states which is which.

**Done when — met, with a documented deviation.** *(Closed 2026-09-09 by decision: F1's purpose is an
environment proof, and a ≈ 1.2 % shortfall on SM does not undermine it. Chasing further would spend GPU
time on a question F1 was never meant to answer.)*

F1's actual purpose **is** discharged: the environment, the released checkpoint, the released subject
embedding, Stage B, the loader and the frozen metric code are known-good **as a system**, so any
discrepancy F5 produces on EVE data is attributable to the data rather than the setup. The run is
reproducible from one documented command, and every reported number is re-derivable from artefacts by
`aggregate_seeds.py` without a GPU (D5).

The literal criterion — all three inside the three-seed band — is **not** met: SED lands inside, SM and
MM sit just outside on the worse side. That gap is **OPEN-7** (§5). Recorded, not resolved.

**Note on `R@5`:** with `--subject_num 5` every rank lies in `{0..4}`, so `p2g`'s `r5` is identically
`100.0`. Confirmed on all seeds (`pr5 = 100.0000`). It is still reported (D6 names R@5) but must be
annotated as structurally saturated.

**Note on `SED_best` / `STDE_best`:** they are **aliases of `SED` / `STDE`**, not a best-of-N —
`evaluation.py` does `SED_best_metrics = SED_metrics_rlts` with no selection step, so they are
identical in every configuration and at any `--eval_repeat_num` (seed 0: `7.3000 / 7.3000` and
`0.8460 / 0.8460`). Reported side by side they read as two corroborating results; they are one number
printed twice. Frozen under D1 — documented, not fixed. [TechStack.md](TechStack.md) §4.

---

### F2 — Dataset bridge: EVE → `fixations.json` + GT heatmaps ✓ DONE (2026-09-10)

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
- [x] **The scored split needs a uniform subject COUNT, not a common cohort** *(corrected
      2026-09-10, twice)*. First: the equal-subject rule was being applied to the support split,
      which needs nothing of the sort — `test.py` never builds a train-split loader and SE-Net's
      `select_fewshot_subject()` keeps whichever subjects have each drawn image. Second, and the
      bigger error: it does not require a common cohort on the *scored* split either.
      `comprehensive_evaluation_by_subject()` loops over each image's **actual** list length and its
      diagonal is **positional**, so subject identities may differ per image. The real constraint is
      that its `-1`-initialised collectors are reduced by a bare `np.mean()` with **no `!= -1`
      filter** on this branch, so a short image folds `-1` into every metric. And
      `args.subject_num` does **not** size the embedding table — `gazeformer.py` L100
      (`nn.Embedding(subject_num, …)`) is **commented out**.
- [x] **Only the retrieval block needs a shared stimulus.** ScanMatch-with-duration is the sole
      metric computed off-diagonal (unguarded, `evaluation.py` L92) and its off-diagonal cells feed
      `p2g()` alone; its reported value is the diagonal slice. MultiMatch, ScanMatch-w/o-duration,
      SED and STDE are each guarded by `if row_idx == col_idx`.
- [x] **OPEN-5 resolved** (§5): **3 subjects per image**, cohort of **38 participants**, retrieval
      **computed and reported separately** from the paper-comparable block.
- [x] **Real run, 2026-09-10.** `--subjects-per-image 3 --support-pool-size 20 --seed 0` over 38
      ids (`train23` excluded) → **354 scored images · 1062 scored cells · 523 support stimuli ·
      1804 trials** (742 train / 1062 test), `min_support_per_subject = 10` (median 20), **zero
      diversions**. All validator invariants pass including `uniform_subject_count: ok (3 per
      scored image)` and the bundle cross-check; `fixations.json` byte-identical on re-run
      (FR11.4); `fixations_sha256 = 46c6926f6075f4c7…`; 877 stimulus images exported. Artefacts in
      the git-ignored `data/eve_bridge/` (convention 5).
- [x] **`train23` excluded from the cohort, deliberately.** It is the only participant with fewer
      than 10 never-scoreable stimuli (9), which would have capped the run at 9-shot. Dropping it
      costs 6 images / 18 cells (1.7 %) and buys `num_fewshot = 10` — the paper's n = 10 row.
- [x] 78/78 pytest tests pass on Windows CPU; frozen files verified untouched (D1).
- [ ] **Still outstanding — re-run the bitwise-parity test under the `isp` env's numpy 1.23.5**
      before F5 trusts the heatmap cache. Validated here on numpy 2.1.2 (TechStack §1); Python 3.12
      is the only interpreter on the dev machine, so this needs the cluster. **F5's precondition**,
      not F2's.

**Done — 2026-09-10.** A real run produced `fixations.json` + `gt_heatmaps.h5` +
`subject_id_map.json` + 877 stimulus `.jpg`s + `bridge_report.json` for a cohort F5 can score.

**D7 counters that are not zero and must reach F7:** `short_scanpath = 25` (padded to length 3
inside the frozen evaluator), `clamped_coords = 5`, `surplus_trial = 73` (the 4th subject dropped
from images seen by 4 — routing it to support would put one name in both splits),
`stimulus_image_conflict = 925` (OPEN-6).

**Two live traps for F5:**
- **The `i_batch > 100` cap is now live.** 354 scored images at `--batch 1` is 354 batches; the run
  would silently stop at 101 images — 29 % of the test set. Remove or parameterise it.
- **R@3 is structurally saturated at `subject_num = 3`** (every rank lies in `{0,1,2}`). Report
  R@1 and MRR; R@3 = 100 % carries no information.

---

### F3 — Subject embeddings for our subjects ▶ NEXT

Depends on **OPEN-2** only (OPEN-5 resolved 2026-09-10). Implements Stage C. F2 produced a support
pool of **10–20 never-scoreable stimuli per subject** (`min_support_per_subject = 10`, median 20),
disjoint by stimulus from the scored split, so the paper's `num_fewshot = 10` fits exactly. The
embedding tensor needs **38 rows**, in dense-id order per `subject_id_map.json`.

- [ ] Resolve OPEN-2 (whose subject embeddings do our subjects get?).
- [ ] **Run SE-Net once per subject, not once for the cohort** *(new 2026-09-10)*.
      `select_fewshot_subject()` draws `num_fewshot` image names from the **union** over the
      fewshot subjects and keeps whichever subjects have each. Our support pools are
      per-subject **disjoint**, so a single draw of 10 names would hand each subject only ≈ 10/N
      scanpaths, unequally. Invoke it with one `--fewshot_subject` at a time so the draw comes
      from that subject's own pool, then concatenate the rows in dense-id order. Call-site only;
      no frozen code involved. Spec: FR3.4a.
- [ ] Verify the tensor is `(38, 384)` and that row *i* is `subject_id_map.json`'s `to_eve[str(i)]`.
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

Depends on **OPEN-6** only (OPEN-5 resolved 2026-09-10). Scope: **877 images** — 354 scored plus
523 support. OPEN-6 is correspondingly larger: `stimulus_image_conflict = 925`.

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

Depends on F3 and F4 (OPEN-5 resolved 2026-09-10). Implements Stage D+E for our data; satisfies
D1, D3, D4, D5, D8. Cohort: `--subject_num 3`, **354 scored stimuli / 1062 cells**, 38 participants,
`--num_fewshot 10`. Retrieval **computed and reported separately** from the paper-comparable block.

- [ ] Create `ISP/EVE/GazeformerISP/` mirroring the OSIE tree, with `utils/evaluation.py` and
      `utils/evaltools/*` **copied verbatim** (D1). The branch consumes the bridge's *artefacts*, not
      its code — the one exception is `tools/eve_bridge/heatmap_metrics.score_step_heatmaps()`, which
      F5 imports directly.
- [ ] Adapt only: `dataset/dataset.py` (class name, `origin_size`, subject handling) and `src/test.py`
      (pass `origin_size=(1080, 1920)` explicitly — the default `(600, 800)` would silently mis-scale
      every coordinate — set `--subject_num` to our unseen-subject count, remove or parameterise the
      `i_batch > 100` cap).
- [ ] Assert `num_fewshot <= min_support_per_subject` from `bridge_report.json` (OPEN-3, FR3.4).
      The **minimum**, not `support_pool_size`: the support pools are per-subject and share no
      image names, so the thinnest one is the binding constraint.
- [ ] **Report retrieval separately, and annotate R@3.** At `--subject_num 3` every rank lies in
      `{0,1,2}`, so **R@3 is structurally saturated at 100 %** and R@5 doubly so; quote **R@1 and
      MRR**. Keep the block out of the paper-comparable row — the published cohort size differs.
- [ ] **Remove or parameterise the `i_batch > 100` cap — it is now LIVE.** 354 scored images at
      `--batch 1` is 354 batches, so the run would silently stop at 101 images (29 % of the test
      set) and report a plausible, wrong number. This is the single most dangerous item in F5.
- [ ] **Re-run F2's bitwise-parity test under the `isp` env's numpy 1.23.5** before trusting
      `gt_heatmaps.h5`; it was validated under numpy 2.1.2 on the dev machine (TechStack §1).
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
- [ ] State the **cohort** the OPEN-5 decision produced: 38 participants, 354 scored stimuli, 1062
      cells, **3 subjects per scored image with identities varying by image** (the evaluator's
      diagonal is positional, so this is legitimate — say so, since a reader will assume a fixed
      cohort). Note that R@3/R@5 are structurally saturated at K = 3.
- [ ] Carry the **`surplus_trial = 73`** count: images seen by 4 participants contribute only 3, the
      4th trial being dropped to keep the support/query split disjoint by stimulus name.
- [ ] State **OPEN-7**: our own OSIE reproduction lands ~1 % below the published row on SM, with all
      three headline metrics offset the same way. Any comparison of EVE numbers to the paper inherits
      that offset, so it has to be quoted alongside them rather than left in the roadmap.
- [ ] State that `SED_best` / `STDE_best` are **aliases** of `SED` / `STDE`, not a best-of-N
      (`evaluation.py` aliases them outright). Report `SED` and `STDE` once each; listing the `_best`
      pair alongside them reads as corroboration that does not exist. [TechStack.md](TechStack.md) §4.
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
  F5 must assert `num_fewshot <= min_support_per_subject`; both values are in `bridge_report.json`.
- *(2026-09-10)* The pools are **per-subject** and drawn from stimuli seen by fewer than
  `subject_num` participants — i.e. stimuli that could never be scored anyway, so they cost the
  query split nothing. The realised run has `min_support_per_subject = 10`, median 20.

Still open for F5: how many `--random_support` repeats to average over. `--subject_num` is settled
at **3** and `--num_fewshot` at **10** by the OPEN-5 decision.

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

### ~~OPEN-5~~ — "The EVE bundle cannot feed the ISP loader as-is" ✓ RESOLVED 2026-09-10 — **the premise was wrong**

**The blocker was ours, not the bundle's.** OPEN-5 asserted that the equal-subject invariant capped
the run at **2 subjects over 13 stimuli**. That came from requiring one cohort to share every scored
stimulus. The frozen evaluator requires no such thing:

- `comprehensive_evaluation_by_subject()` loops `for row_idx in range(len(predict_fix_vector))` —
  each image's **actual** list length — and its diagonal is **positional**, so position *i* is the
  same participant on both sides whoever that is. **Identities may differ from image to image.**
- The real constraint is arithmetic: the collectors are allocated `(n_images, subject_num, …)`,
  initialised to `-1`, and reduced by a bare `np.mean()` with **no `!= -1` filter** on this branch
  (`evaluation.py` L137-152). An image contributing fewer than `subject_num` records folds `-1`
  sentinels into every metric. The requirement is a uniform **count**, not a common cohort.
- `args.subject_num` does **not** size the model's embedding table: `gazeformer.py` L100
  (`nn.Embedding(subject_num, ...)`) is **commented out** and `self.subject_embed` is whatever
  tensor `--user_emb_path` holds, indexed by the record's dense subject id.

**Only the retrieval block genuinely needs several subjects on one image.** ScanMatch-with-duration
is the only metric computed off-diagonal (L92, unguarded), feeding `p2g()`; MultiMatch,
ScanMatch-w/o-duration, SED and STDE are each guarded by `if row_idx == col_idx` and compare a
subject against itself alone.

What EVE caps is subjects **per image**, not the cohort:

| subjects/image | scored images | scored cells | participants |
|---|---|---|---|
| 2 | 743 | 1486 | 39 |
| **3 - chosen** | **360** | **1080** | **39** |
| 4 | 77 | 308 | 39 |

**Decided:** `subject_num = 3`; cohort of **38** participants (`train23` excluded — its 9
never-scoreable stimuli would have capped the run at 9-shot, and dropping it costs 6 images to buy
the paper's `num_fewshot = 10`); realised as **354 scored images / 1062 cells**. Retrieval is
**computed and reported separately** from the paper-comparable block — no longer degenerate at
K = 3, but **R@3 is structurally saturated** (every rank lies in `{0,1,2}`), so quote R@1 and MRR.

No frozen code was touched (D1 intact) and no bundle re-export was needed.

**What survives from the original finding.** Raised 2026-09-08 against
`eve_shared/EveDataset/bundle` (3096 samples, 54 participants):

1. **Every `test*` participant has `valid == False` on every trial** (train 1951/2238 valid,
   val 185/261, **test 0/597**). **This still stands** — the cohort is drawn from `train*` / `val*`.
2. **Participants see near-disjoint stimulus sets.** The old "max shared stimuli" frontier
   (2 → 13, 3 → 4, 4 → 2) is **superseded**: it answers the wrong question, since no common cohort
   is required. The numbers that matter are that **360 stimuli were seen by ≥ 3 participants** and
   **743 by ≥ 2**. Reopening this for a larger `subject_num` still means re-exporting the bundle —
   at K = 4 only 77 images qualify.

The superseded frontier is still reproducible by
`py -m pytest tests/eve_bridge -m bundle --bundle-dir <dir> -s -k feasibility`.

### OPEN-6 — The same `stimulus_name` renders differently per participant *(blocks F4)*
Raised 2026-09-08 alongside OPEN-5; **rescaled 2026-09-10**. On the realised 38-participant cohort
`stimulus_image_conflict == 925` across 877 exported stimuli — i.e. essentially **all** of them.
(The original figure of 13 was measured on the 2-subject cohort and understated the scale, not the
kind, of the problem.) The images are all 1920×1080 RGB and open
cleanly, but the two participants' renderings differ substantially: mean absolute difference ≈ 9–29
per channel, 24–72 % of pixels differing, whole-image correlation ≈ 0.70 on the case inspected.
Recognisably the same photograph, differently rendered — a brightness/scale/crop difference, not
encoding noise.

The bridge keeps the first exp_key's image in sorted order and counts the collision. But F4 produces
one `.pth` per `stimulus_name`, which cannot represent two renderings — so the participant whose
rendering was discarded gets scored against features of an image they did not see. Decide before F4
whether to pick one rendering, key features per (stimulus, participant), or treat the divergence as a
defect in the bundle export.

### OPEN-7 — F1 lands consistently ~1 % below the published OSIE row *(does not block; colours F7)* ◀ NEW
Raised and **accepted** 2026-09-09 at F1's close. Against the paper's **n = 10, ISP-SENet** row:

| | ours (3-seed mean ± across-seed std) | paper | delta | inside band |
|---|---|---|---|---|
| SM ↑ | 0.3704 ± 0.0041 | 0.375 | −0.0046 | no |
| MM ↑ | 0.8010 ± 0.0019 | 0.803 | −0.0020 | no |
| SED ↓ | 7.3438 ± 0.0381 | 7.318 | +0.0258 | yes |

All three fall on the **worse** side, and for SM/MM the published value sits outside the full
three-seed range. Three independent metrics offset in one direction is a small **systematic**
difference, not scatter.

**Ruled out already — do not re-test:**
- `--eval_repeat_num > 1` **raises IndexError** on this path ([TechStack.md](TechStack.md) §5):
  `evaluation.py`'s collectors are `(n_images, subject_num)` but `test.py` would supply
  `eval_repeat_num * subject_num` rows. R = 1 is structurally forced.
- `--random_support` is inert for a query-set score — `select_fewshot_subject()` returns early when
  `split != 'train'`.
- Padding and coordinate contamination: the sweep's preflight reports `n_short = 0` and `oob = 0`.

**Still candidate:** env drift on the float metric paths (numpy 2.1.2 vs the pinned 1.24.3 — MultiMatch
and STDE are exactly where [TechStack.md](TechStack.md) §1.1 predicts drift would surface); the
released checkpoint differing from the one behind the table; or the published figure averaging over
something not reproduced here.

**Why it does not block.** F1 exists to prove the environment end-to-end, which it did. The residual is
small, bounded, and measured. It is **F7's to state plainly**, not F5's to fix — and it must not be
quietly dropped: a reader comparing our EVE numbers to the paper needs to know our own OSIE
reproduction ran ~1 % low to begin with.

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
