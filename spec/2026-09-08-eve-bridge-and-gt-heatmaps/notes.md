# Closing note — EVE bridge and ground-truth step heatmaps

> Written 2026-09-08 after implementing [plan.md](plan.md) Steps 1–8.
> Companion to [requirements.md](requirements.md) and [validation.md](validation.md).

---

## 1. What was built

| file | role |
|---|---|
| `tools/eve_bridge/__init__.py` | package marker; scope rules |
| `tools/eve_bridge/heatmaps.py` | `build_step_heatmaps`, `to_target_scanpath` (FR7) |
| `tools/eve_bridge/convert.py` | `build_fixations`, `export_stimuli` (FR1–FR6) |
| `tools/eve_bridge/store.py` | `GtHeatmapStore` (FR8) |
| `tools/eve_bridge/validate.py` | `validate`, `BridgeValidationError` (FR9) |
| `tools/eve_bridge/heatmap_metrics.py` | `score_step_heatmaps` (FR10) — the only torch importer |
| `tools/eve_bridge/build.py` | CLI (FR11) |
| `tests/eve_bridge/` | validation.md Groups 1–6 plus the `[bundle]` group |

`.gitignore` gained `*.h5` alongside the existing `*.pt` / `*.pth`.

The frozen files are untouched — `git diff --exit-code` over
`ISP/OSIE/GazeformerISP/src/utils/` and `.../src/models/loss.py` is asserted by a test
(D1, FR10.3).

**Code Correctness: 75 of 75 pytest tests pass** on Windows CPU (Python 3.12; `py -3.8`
is not installed on this machine — see §4).

---

## 2. Open decisions this spec answers

### OPEN-3 — support/query split *(partially answered)*

- Split is **by stimulus**, not by trial: `"split": "train"` is the support pool,
  `"split": "test"` is the query pool, and the two name sets are disjoint by
  construction and asserted before writing (FR3.3) *and* re-asserted on the written
  artefact. This is the only partition under which "the support scanpaths were never
  scored" holds for all N × N cells of the evaluator's matrix — rationale in
  [plan.md](plan.md) §"Why the support/query split is by stimulus".
- `support_pool_size = 20` by default, deliberately larger than the paper's
  `num_fewshot = 10`, so `--random_support` repeats draw varying support sets that stay
  inside the pool. F5 must assert `num_fewshot <= support_pool_size`; the value is
  recorded in `bridge_report.json`.
- Still open for F5: how many `--random_support` repeats to average over.

### OPEN-4 — stimulus resolution and resize policy *(answered: squash)*

The bridge writes X/Y in native 1920×1080 and declares `origin_size = (1080, 1920)`.
The squash to 512×384 happens inside `OSIE_evaluation`'s `resizescale_x/y`
(3.75 / 2.8125, non-uniform). Chosen over letterboxing so that `args.width/height`,
the ScanMatch 16×12 grid and the MultiMatch `screensize` stay bit-identical to the
published configuration. The distortion is F7's to state, not the bridge's to hide.

---

## 3. Findings from the real bundle — **these block the spec's intended configuration**

Run:

```
py -m pytest tests/eve_bridge -m bundle \
    --bundle-dir <dir> --bridge-subjects train02,train09 --bridge-support-pool-size 5 -s
```

against `eve_shared/EveDataset/bundle` (3096 samples, 54 participants).

### F-A — every `test*` participant has `valid == False` on every trial

`samples_df` groups by split as: train 1951/2238 valid, val 185/261 valid,
**test 0/597 valid**. FR1.1's default (`every id starting "test"`) therefore yields zero
candidate trials, and `build_fixations` raises — correctly, per D7. The query subjects
must come from the `train*` / `val*` participants, or the bundle must be re-exported with
usable test-split labels.

### F-B — EVE participants see near-disjoint stimulus sets, so the equal-subject invariant caps N at 2

FR2.3 keeps only stimuli present for **all** selected subjects, because the ISP loader
requires every image to yield the same number of subjects (TechStack §3.1) or
`evaluation.py`'s (subject × subject) matrix goes ragged. Exhaustive search over the 39
participants with any valid trial:

| n subjects | max shared stimuli | best group |
|---|---|---|
| 2 | **13** | `train02`, `train09` |
| 3 | 4 | `train06`, `train09`, `train10` |
| 4 | 2 | `train06`, `train09`, `train10`, `train37` |

No stimulus is shared by more than 4 participants anywhere in the bundle. The spec's
`support_pool_size = 20` needs ≥ 21 shared stimuli; the maximum available is 13, and only
for a **2**-subject group. On the best feasible configuration the equal-subject filter
discards 91 of 104 candidate stimuli.

This is a property of how EVE assigns stimuli, not a bridge defect — the bridge fails
loudly and counts the drops exactly as D7 requires. But it means **F3 and F5 cannot
proceed on this bundle as specced**: a 2-subject evaluation makes the retrieval block
(MRR, R@1/3/5) degenerate and the personalization claim untestable.

Resolving it requires a decision that belongs in the constitution, not here. The options,
for the record:

1. Re-export the bundle so participants share a common image set (the EVE protocol may
   have a shared subset that this export did not preserve).
2. Relax the equal-subject invariant — which means adapting `OSIE_evaluation` and the
   evaluator's matrix construction, i.e. touching frozen code (D1). Not free.
3. Accept a 2-subject evaluation and report only the diagonal metrics, dropping the
   retrieval block. F7 would have to state plainly what that does and does not license.

### F-C — the same `stimulus_name` renders differently per participant

`stimulus_image_conflict == 13` on the 13 shared stimuli — i.e. **all** of them. The
images are all 1920×1080 RGB and open cleanly, but the two participants' renderings of
the same `stimulus_name` differ substantially (mean absolute difference ≈ 9–29 per
channel, 24–72 % of pixels differing, whole-image correlation ≈ 0.70 on the case
inspected). They are recognisably the same photograph, differently rendered — a
brightness/scale/crop difference, not encoding noise.

FR6.1 keeps the first exp_key's image in sorted order and counts the collision. That
makes F4's feature extraction ambiguous: one `.pth` per `stimulus_name` cannot represent
two different renderings, and the fixations of the participant whose rendering was
discarded are then scored against features of an image they did not see. This needs a
decision before F4.

---

## 4. Deviations from the spec, and why

- **`py -3.8` → `py`.** Python 3.8 is not installed on this dev machine; the tests were
  run on 3.12 (numpy 2.1.2, scipy 1.14.1, h5py 3.12.1, torch 2.7.0+cpu). The bridge core
  is version-agnostic, but the numbers that matter — the bitwise-parity test against
  `OSIE.__getitem__` — should be re-run under the `isp` env's numpy 1.23.5 before F5
  relies on the cache, since structured-array and float32 promotion behaviour is exactly
  what TechStack §1 pins numpy for.
- **`counters["duplicate_trial"]`** was added beyond FR2.2/FR2.3. If two candidate rows
  share a `(stimulus_name, subject)` the first is kept and the rest counted, rather than
  silently overwritten — an overwrite would break the exp_key-uniqueness invariant the
  Data Architecture Integrity checks rely on. It is `0` on this bundle.
- **The uniform-prediction NSS floor is not 0.** validation.md Group 5 expects
  "NSS is approximately 0" for a constant prediction. Upstream `NSS` standardises by
  `(x - mean) / (std + 1e-7)`; on a constant float32 map the numerator is pure rounding
  residual (~1e-8) against a bare-epsilon denominator, so the result is O(1) noise whose
  value depends on the constant (−1.79 at 0.37, −0.60 at 1.0, 0.0 at 2.0). FR10.3 forbids
  reimplementing the metric, so the test asserts `CC == 0` exactly and bounds
  `|NSS| < 2`, and a second test establishes the meaningful floor with a *random*
  (non-constant) prediction, where NSS and CC are both ≈ 0 and well conditioned. Worth
  knowing for F5: an NSS near zero on a nearly-flat prediction map is not informative.

---

## 5. Not done

- Plan Step 9 says explicitly **not** to edit the constitution from this spec. The
  Roadmap's F2 row is therefore left as-is; §2 and §3 above are what a later constitution
  update should pick up — in particular, F-A/F-B/F-C are new blockers for F3/F4/F5 that
  belong in Roadmap §5 as an OPEN decision.
- The `[bundle]` Data Validity group cannot be run in the spec's intended configuration
  (see F-A/F-B). It was run on the largest feasible one (2 subjects × 13 stimuli,
  `support_pool_size=5`): 11 checks pass, `test_subject_coverage` xfails with the
  recorded participant count, and `test_stimulus_coverage` / `test_stimulus_files` fail
  with F-B and F-C respectively — which is these checks doing their job, not a code
  defect.

---

## 6. Addendum 2026-09-10 — the cohort was never capped at 2; F2 rebuilt and run

> Two corrections landed in one session, both prompted by the same question:
> *why does the support set need a shared stimulus?* The answer was "it doesn't",
> and pulling that thread showed the scored split didn't need what §3's F-B claimed either.

### 6.1 Correction 1 — the support split never needed the equal-subject rule

§3's F-B applied the equal-subject filter to **both** splits, so a 13-stimulus cohort spent 10 of
them on support and left **3** scored. Two facts, read off the code:

- `ISP/OSIE/GazeformerISP/src/test.py` builds `OSIE_evaluation(..., type="test")` and **never
  constructs a train-split loader**. The embedding arrives precomputed via `--user_emb_path`.
- `SE-Net/common/utils.py::select_fewshot_subject()` (L1118) keeps whichever subjects have each
  drawn image — `if subject in img_name_groups[img_name]` — so it tolerates a subject missing one.

### 6.2 Correction 2 — the scored split needs a uniform *count*, not a common cohort

F-B's real error. `comprehensive_evaluation_by_subject()` loops
`for row_idx in range(len(predict_fix_vector))` — the **actual** per-image length — and its diagonal
is **positional**, so position *i* is the same participant on both sides whoever that is. **Subject
identities may differ from image to image.**

What breaks is the mean, not the matrix: collectors are allocated `(n_images, subject_num, …)`,
initialised to `-1`, and reduced by a bare `np.mean()` with **no `!= -1` filter** on this branch
(L137–152). A short image folds `-1` into every metric.

And `args.subject_num` does **not** size the embedding table — `gazeformer.py` L100
(`nn.Embedding(subject_num, …)`) is **commented out**; `self.subject_embed` is the loaded tensor,
indexed by the record's dense subject id. So a large cohort can be scored at `subject_num = 3`.

**Which metric actually needs a shared stimulus: only the retrieval block.** ScanMatch-with-duration
is the sole metric computed off-diagonal (L92, unguarded), and its off-diagonal cells feed `p2g()`
and nothing else — its *reported* value is the diagonal slice. MultiMatch, ScanMatch-w/o-duration,
SED and STDE are each guarded by `if row_idx == col_idx` and compare a subject only against itself.

The cohort was therefore never capped at 2. What EVE caps is subjects **per image** (2–4):

| subjects/image | scored images | cells | participants |
|---|---|---|---|
| 2 | 743 | 1486 | 39 |
| **3** | **360** | **1080** | **39** |
| 4 | 77 | 308 | 39 |

### 6.3 The decisions taken

- **3 subjects per image** → `--subject_num 3`. Retrieval **computed and reported separately**, not
  folded into the paper-comparable block (it is no longer degenerate at K = 3, but R@3 is
  structurally saturated since every rank lies in `{0,1,2}` — report R@1 and MRR).
- **`train23` excluded.** It is the one participant with fewer than 10 sub-K stimuli (9), which would
  have capped the run at 9-shot. Dropping it costs 6 images / 18 cells (1.7 %) and buys
  `num_fewshot = 10` — the paper's n = 10 row, the entire point of comparability.
- Support pools are **not trimmed to a common size**; only the minimum matters. Trimming to the
  thinnest subject cost ~10 support stimuli per subject and 11 scored images, and bought nothing.

### 6.4 The run

```
py tools/eve_bridge/build.py --bundle-dir <eve_shared>/EveDataset/bundle     --out-dir data/eve_bridge --subjects-per-image 3 --support-pool-size 20 --seed 0     --unseen-subjects train01 train02 train04 ... val05        # 38 ids, train23 excluded
```

**38 participants · 354 scored images · 1062 scored cells · 523 support stimuli · 1804 trials**
(742 train / 1062 test), `min_support_per_subject = 10`, median 20, **zero diversions**. All
validator invariants pass including `uniform_subject_count: ok (3 per scored image)` and the bundle
cross-check; `fixations.json` byte-identical on re-run (FR11.4);
`fixations_sha256 = 46c6926f6075f4c7…`. 877 stimulus images exported. Artefacts in the git-ignored
`data/eve_bridge/`.

D7 counters that are **not** zero and must reach F7: `short_scanpath = 25` (padded to length 3
inside the frozen evaluator — 2.4 % of scored cells at most), `clamped_coords = 5`,
`surplus_trial = 73` (4th subject dropped from images seen by 4), `stimulus_image_conflict = 925`
(OPEN-6, below).

### 6.5 What this does *not* fix

- **OPEN-6 is now much larger.** `stimulus_image_conflict = 925` across 877 stimuli — nearly every
  one. At the old 2-subject scale it was 13. It blocks F4 and is unchanged in kind, only in scale.
- **The `i_batch > 100` cap in `test.py` is now live.** 354 scored images at `--batch 1` is 354
  batches; the run would silently stop at 101 images (29 % of the test set). F5 must remove or
  parameterise it — Roadmap already flags this.
- **F-A stands**: all 15 `test*` participants are `valid == False` throughout, so the cohort is
  `train*` / `val*`.
- **The numpy-1.23.5 parity re-run is still outstanding** (dev machine is Python 3.12 / numpy 2.1.2).
  F5's precondition for trusting `gt_heatmaps.h5`.

### 6.6 For F3 — SE-Net must be invoked once per subject

`select_fewshot_subject()` draws `num_fewshot` names from the **union** across the fewshot subjects.
Our support pools are per-subject and largely disjoint, so one draw of 10 names would hand each
subject a small, unequal share. Run SE-Net with a single `--fewshot_subject` at a time so the draw
comes from that subject's own pool, then concatenate rows in dense-id order. The resulting tensor
needs **38 rows**. Call-site only; no frozen code involved. Recorded as FR3.4a.
