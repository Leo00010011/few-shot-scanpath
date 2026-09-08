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
