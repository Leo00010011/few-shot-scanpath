# Requirements — EVE bridge and ground-truth step heatmaps

> Spec 2026-09-08. Feature: `eve-bridge-and-gt-heatmaps`.
> Implements Roadmap **F2** (Stage A) for the EVE dataset, plus the ground-truth heatmap
> substrate that the later EVE eval branch (F5) needs for its heatmap-metric block.
> Constitution: [Mission](../constitution/Mission.md) · [TechStack](../constitution/TechStack.md) · [Roadmap](../constitution/Roadmap.md)

---

## Goal

Turn the EVE eye-tracking bundle (`bundle.h5`, read through the installed `evedataset`
package) into the canonical inputs ISP-SENet consumes — a `fixations.json` conforming
byte-for-byte to the schema in TechStack §3.1, a flat directory of stimulus `.jpg` files,
and an explicit `subject_id_map.json` — with a support/query partition that is **disjoint
by stimulus**, so the few-shot subject embedding for an EVE participant can be produced by
cross-attention over that participant's *support* scanpaths without ever seeing a scanpath
that is later scored. In addition, because the model's spatial output is a per-timestep
24×32 action-probability map and EVE ships no saliency maps, the feature produces the
ground-truth counterpart of those maps: a per-(trial, timestep) Gaussian-blurred fixation
heatmap built from the ground-truth scanpath coordinates, cached to HDF5 and addressable
offline. Everything here is pure numpy/h5py/PIL/scipy and runs on Windows CPU, so the bridge
can be built and validated on the dev machine before any GPU time is spent.

---

## Scope

**In scope**

- Reading `bundle.samples_df`, `bundle.get_scanpath()`, `bundle.get_stimulus()` from an
  `EveBundle` — no WebGazer, no EyeNet, no `run_key` selection.
- Selection of the unseen (query) EVE participants and their dense `0..N-1` remapping.
- The support/query partition, disjoint **by `stimulus_name`**, emitted as the `split`
  field (`"train"` = support pool, `"test"` = query pool).
- Coordinate/unit conversion into the canonical `fixations.json` contract, with the
  squash policy 1920×1080 → 512×384 declared via `origin_size=(1080, 1920)`.
- Export of one deduplicated `<stimulus_name>.jpg` per stimulus into a flat image dir.
- `build_step_heatmaps()` — the ground-truth per-timestep 24×32 heatmap builder, numerically
  identical to the construction inside `ISP/OSIE/GazeformerISP/src/dataset/dataset.py::OSIE.__getitem__`.
- `GtHeatmapStore` — an HDF5 cache of those heatmaps, keyed to `fixations.json` record order
  and carrying the originating EVE `exp_key` for provenance.
- `score_step_heatmaps()` — a thin CPU wrapper computing NSS / CC / KLD from the model's
  `all_actions_prob` against the cached ground-truth heatmaps, delegating to the existing
  `models/loss.py` implementations.
- A validator that enforces every invariant below and a machine-readable `bridge_report.json`.

**Explicitly out of scope**

- Stage B feature extraction (`*.pth`, `embeddings.npy`) — Roadmap **F4**.
- Stage C SE-Net subject embeddings — Roadmap **F3**. This spec only guarantees the support
  set exists and is disjoint from the query set; it does not run SE-Net.
- Any `ISP/EVE/GazeformerISP/` branch, `test.py`, or run script — Roadmap **F5**.
- Running the scanpath metric suite (MultiMatch / ScanMatch / SED / STDE / retrieval) — F5/F6.
- Any edit to `utils/evaluation.py` or `utils/evaltools/*` (**D1**). This feature does not
  touch them at all.
- Non-image EVE stimuli (video, wikipedia). The bundle carries image stimuli only; a
  non-image row, if ever present, is an error, not a filter.
- Training, fine-tuning, RL (**D8**).

---

## Functional Requirements

### FR1 — Subject selection and dense remapping (D4)

**FR1.1** The builder takes `unseen_subjects: list[str]` — EVE participant ids exactly as
they appear in `samples_df["subject"]` (e.g. `"test01"`). Default when omitted: every
participant whose id starts with `"test"`, sorted ascending.

**FR1.2** Each selected participant is assigned a dense id equal to its index in the
*sorted* `unseen_subjects` list, giving `0..N-1`. Sorting is applied before assignment so
the mapping does not depend on argument order.

**FR1.3** A `subject_id_map.json` is written with both directions and is the sole authority
for "which real EVE participant is row 3?":

```json
{"to_dense": {"test01": 0, "test02": 1}, "to_eve": {"0": "test01", "1": "test02"}}
```

**FR1.4** A participant id absent from `samples_df["subject"]` raises `ValueError` naming
the id and listing the five closest available ids. Never silently skipped.

**FR1.5** `len(unseen_subjects) < subjects_per_image` raises `ValueError` — an image cannot
contribute K records from fewer than K participants.

**FR1.6** *(revised 2026-09-10.)* The default cohort is **every participant with at least one valid
trial**, not every id starting `"test"`. On the EVE bundle all 15 `test*` participants are
`valid == False` throughout (Roadmap F-A), so the old default resolved to an empty cohort.

### FR2 — Trial selection and the equal-subject invariant (D7, TechStack §3.1)

**FR2.1** Candidate rows are `samples_df` rows with `subject in unseen_subjects` **and**
`valid == True`.

**FR2.2** A candidate is dropped, and counted by reason, when any of:
`get_scanpath(exp_key).shape[1] == 0` (`empty_scanpath`); any non-finite value in the
x/y/duration rows (`non_finite`); `stimulus_path` empty (`no_stimulus`).

**FR2.3** *(revised twice on 2026-09-10 — the binding rule is a uniform subject **count** on
the **scored** split, not a common cohort.)*

`comprehensive_evaluation_by_subject()` loops
`for row_idx in range(len(predict_fix_vector))` — each image's *actual* list length — and its
diagonal is **positional**, so position *i* is the same participant in the prediction and in the
ground truth, whoever that participant happens to be. The subject **identities may therefore differ
from image to image.**

What the evaluator cannot tolerate is a varying *count*. Its collectors are allocated
`(n_images, subject_num, …)`, initialised to `-1`, and reduced with a bare `np.mean()` that carries
**no `!= -1` filter on the OSIE branch** (`evaluation.py` L137–152; the COCO_FV branch does filter —
TechStack §4.2 divergence 3). An image contributing fewer than `subject_num` records therefore folds
`-1` sentinels into every metric. That arithmetic, not any metric's definition, is the constraint.

Crucially `args.subject_num` does **not** size the model's subject embedding:
`models/gazeframer.py` L100 (`nn.Embedding(subject_num, …)`) is **commented out**, and
`self.subject_embed` is whatever tensor `--user_emb_path` holds, indexed by the record's dense
subject id. A cohort of many participants can thus be scored at `subject_num = 3`.

So, after drops:

- **Query split** — stimuli seen by **≥ K** subjects (`K = subjects_per_image`), each contributing
  **exactly K** records chosen deterministically. A surplus trial on an image seen by more than K
  subjects is **dropped** and counted as `surplus_trial`; routing it to support would put one
  stimulus *name* in both splits and break FR3.3.
- **Support candidates** — stimuli seen by **< K** subjects. These can never be scored, so using
  them as support costs the query split nothing. Counted as `incomplete_stimulus` (the name is
  retained; it now means "cannot be scored", not "discarded").

Only the **retrieval block** (MRR, R@1/3/5) genuinely requires several subjects on one image: it is
computed by `p2g()` over the ScanMatch-with-duration matrix, the only metric evaluated off-diagonal
(`evaluation.py` L92, unguarded). MultiMatch, ScanMatch-without-duration, SED and STDE are all
guarded by `if row_idx == col_idx` and compare a subject only against **itself**.

**FR2.4** Raise `ValueError` if no stimulus was seen by K subjects (the query split would be empty),
or if the eligible pool cannot both cover the support diversion (FR3.1) and leave at least one
scored stimulus.

**FR2.5** A `stimulus_name` containing the substring `jpg` anywhere raises `ValueError`
(TechStack §3.2: the feature path is built by a literal `str.replace('jpg', 'pth')`).

### FR3 — Support/query partition (OPEN-3, disjointness requirement)

**FR3.1** *(revised 2026-09-10.)* The support pool is filled from stimuli that **can never be
scored** — those seen by fewer than K subjects — and a query-eligible stimulus is diverted into it
**only when some subject would otherwise have no support at all**, since each diversion costs a
scored image for every subject.

1. For each subject, its sub-K candidates are sorted ascending and shuffled with
   `random.Random(f"{seed}-{dense}")`; the subject takes up to `support_pool_size` of them.
2. If the thinnest subject has **zero** such candidates, `support_pool_size` eligible stimuli are
   diverted under `random.Random(seed)`, and each short subject tops up from those it saw.
3. The remaining eligible stimuli form the query pool, each keeping exactly K subjects.

The pools are deliberately **not trimmed to a common size**: each subject's embedding is built from
its own `num_fewshot` scanpaths, so only the **minimum** matters. Levelling every subject down to the
thinnest one discards support the others already have (on EVE: median 20 against a minimum of 9) and
buys nothing.

On a **fully-crossed** dataset no stimulus falls below K, so nobody has a private pool, the deficit
is the whole pool, and this reduces to the original "first `support_pool_size` names" rule exactly —
which is what keeps the fully-crossed fixtures meaningful.

**FR3.2** Support-pool trials get `"split": "train"`; query-pool trials get `"split": "test"`.
No record is emitted with `"split": "validation"`.

**FR3.3** The two pools are disjoint by construction and this is re-asserted before writing:
`set(train_names) & set(test_names) == set()`. A violation raises `AssertionError`.

**FR3.4** `support_pool_size` must be ≥ the `--num_fewshot` the later eval run will use
(paper default 10). The report records `support_pool_size`, `support_per_subject` and
`min_support_per_subject`; F5 asserts `num_fewshot <= min_support_per_subject` — the
**minimum**, because with subject-private support pools the subjects no longer share image
names, so `support_pool_size` alone is not the binding constraint.

**FR3.4a** *(added 2026-09-10, a finding for F3.)* `select_fewshot_subject()` draws
`num_fewshot` image names from the **union** over the fewshot subjects and then keeps
whichever subjects have each. With per-subject-disjoint support pools, a single draw of 10
names therefore gives each subject only ≈ `10 / N` support scanpaths, unequally. To obtain a
genuine n-shot embedding per subject, **F3 must invoke SE-Net once per subject** (a single
`--fewshot_subject`, so the draw comes from that subject's own pool) and concatenate the
resulting rows in dense-id order. This is a call-site decision; no frozen code is involved. Rationale: `select_fewshot_subject()` samples
`num_fewshot` image names *from the `train` split* under `random_support`; keeping the
pool larger than `num_fewshot` lets `--random_support` repeats vary the support set while
FR3.3 keeps every draw disjoint from the scored `test` split.

### FR4 — Coordinate, unit and index conventions (D3)

**FR4.1** `bundle.get_scanpath(exp_key)` returns `(4, F) float32` with rows
`[start_t_sec, duration_ms, x_px, y_px]` in EVE screen space 1920×1080, origin top-left,
0-indexed. The bridge reads rows 2, 3, 1 and ignores row 0.

**FR4.2** `X = float(x_px) + 1.0`, `Y = float(y_px) + 1.0` — converted to the 1-indexed
convention `fixations.json` declares (D3) and that `OSIE.__getitem__` assumes when it
computes `(pos_x - 1) / downscale_x`. The +1 is applied once, in the bridge, and nowhere else.

**FR4.3** After the +1, `X` is clamped to `[1.0, 1920.0]` and `Y` to `[1.0, 1080.0]`. Each
clamped coordinate is counted as `clamped_coords` in the report. Clamping never drops a trial.

**FR4.4** `T = int(round(duration_ms))`, milliseconds, integer, per TechStack §3.1. A
duration rounding to `0` is raised to `1` and counted as `zero_duration`.

**FR4.5** `length == len(X) == len(Y) == len(T)`, no truncation applied by the bridge —
the loader truncates to `args.max_length`. The report counts trials with `length > 16`
(`over_max_length`) and `length < 3` (`short_scanpath`; these get padded with `(1., 1., 1e-3)`
inside the frozen evaluator — D7).

**FR4.6** `"condition": "freeview"` and `"task": "none"` are emitted as constants, matching
the OSIE reference file. The loader hardcodes `task = "free-viewing"` regardless.

**FR4.7** The declared origin size is `origin_size = (1080, 1920)` as `(H, W)`. It is written
to `bridge_report.json` and to the heatmap store attrs so F5 can pass it explicitly rather
than inheriting the `OSIE_evaluation` default `(600, 800)` (D3). The resulting
`resizescale_x = 1920/512 = 3.75` and `resizescale_y = 1080/384 = 2.8125` are non-uniform:
this is the deliberate **squash** policy, chosen so `args.width/height` stay 512/384 and the
ScanMatch 16×12 binning and MultiMatch `screensize` remain bit-identical to the published
configuration.

### FR5 — `fixations.json` output

**FR5.1** A flat JSON list, one object per (stimulus, subject) trial, with exactly the keys
`name, subject, X, Y, T, length, split, condition, task` in that order.

**FR5.2** `name = f"{stimulus_name}.jpg"`.

**FR5.3** Record order is deterministic: sorted by `(name, subject)` ascending. This order
is the index space the heatmap store keys against (FR8.2).

**FR5.4** `X`/`Y` are JSON floats, `T`/`length`/`subject` are JSON ints, written with
`json.dump(..., indent=4)`.

### FR6 — Stimulus export

**FR6.1** One `.jpg` per unique surviving `stimulus_name`, written to `<out_dir>/stimuli/`,
from `bundle.get_stimulus(exp_key)` for the first exp_key of that stimulus in sorted order.

**FR6.2** Images are written at their native 1920×1080; no resize is done here. The
feature extractor (F4) does its own resize to 768×1024.

**FR6.3** Saved with `PIL.Image.fromarray(arr).save(path, quality=95, subsampling=0)`.

**FR6.4** If two exp_keys sharing a `stimulus_name` yield byte-different images, the first
is kept and the collision is counted as `stimulus_image_conflict` in the report. Detected by
comparing the SHA-256 of the raw `(H, W, 3) uint8` array.

**FR6.5** Every `name` in `fixations.json` resolves to a file in `<out_dir>/stimuli/`
(asserted by the validator, FR9.1.8).

### FR7 — Ground-truth step heatmaps

**FR7.1** `build_step_heatmaps(X, Y, length, origin_size=(1080, 1920), action_map=(24, 32),
max_length=16, blur_sigma=1)` returns `(heatmaps, action_mask)`:

- `heatmaps`: `(max_length, 24, 32) float32`
- `action_mask`: `(max_length,) float32`

The body is numerically identical to `OSIE.__getitem__`:
`downscale_x = origin_size[1] / action_map[1]` (= 60.0),
`downscale_y = origin_size[0] / action_map[0]` (= 45.0);
`col = ((X[i] - 1) / downscale_x).astype(np.int32)`, `row = ((Y[i] - 1) / downscale_y).astype(np.int32)`
(truncation, not rounding); a single `1.0` placed at `[row, col]`; then
`scipy.ndimage.gaussian_filter(map, blur_sigma)` followed by division by the map's sum.
`action_mask[i] = 1` for `i < min(length, max_length)`, and if
`action_mask.sum() <= max_length - 1` the next slot is additionally set to 1 (the
termination slot — same rule as upstream).

**FR7.2** Timesteps `i >= min(length, max_length)` leave `heatmaps[i]` all-zero. A valid
step's map sums to `1.0 ± 1e-5`.

**FR7.3** `to_target_scanpath(heatmaps, action_mask) -> (max_length, 24*32 + 1) float32`
reconstructs the upstream `target_scanpath` layout: column 0 is the termination flag
(`1.0` where the step is beyond `length`, else `0.0`), columns `1:` are
`heatmaps[i].reshape(-1)`. Provided so F5 can feed the model's loss/metric code the exact
tensor shape it expects without re-deriving the layout.

**FR7.4** A discretized index outside `[0, 23]` × `[0, 31]` (possible only if FR4.3 clamping
were bypassed) raises `ValueError`; it is never clipped silently.

### FR8 — `GtHeatmapStore` (HDF5 cache)

**FR8.1** Layout of `<out_dir>/gt_heatmaps.h5`:

```
/                       attrs: created_utc (str, ISO-8601 Z), bundle_dir (str),
                               origin_size (2,) int32 = [1080, 1920],
                               action_map  (2,) int32 = [24, 32],
                               max_length  int32 = 16, blur_sigma float32 = 1.0,
                               fixations_sha256 (str), n_trials int32,
                               subject_ids_dense (N_subj,) int32,
                               subject_ids_eve   (N_subj,) vlen utf-8
/trials/trial_key       (N,)             vlen utf-8   "{name}|{subject}"
/trials/exp_key         (N,)             vlen utf-8   originating EVE exp_key (D4 provenance)
/trials/subject         (N,)             int32        dense subject id
/trials/length          (N,)             int32        min(len(X), max_length)
/trials/action_mask     (N, 16)          float32
/trials/heatmaps        (N, 16, 24, 32)  float32, gzip=4, chunks=(1, 16, 24, 32)
```

**FR8.2** Row `i` of every `/trials/*` dataset corresponds to record `i` of `fixations.json`
in the FR5.3 order. `fixations_sha256` is the SHA-256 of the written `fixations.json` bytes;
`GtHeatmapStore.load()` accepts an optional `fixations_path` and raises `ValueError` on
mismatch, so a stale cache can never be silently paired with a regenerated JSON.

**FR8.3** `get(name, subject) -> (16, 24, 32) float32` looks up by `trial_key`; a missing
key raises `KeyError` naming the key. `get_batch(keys) -> (len(keys), 16, 24, 32)` preserves
the requested order.

**FR8.4** `exp_key_of(name, subject) -> str` and `trial_key_of(exp_key) -> str` are inverses
over the stored arrays; both raise `KeyError` on an unknown argument. There is no fallback
that invents a key.

**FR8.5** `GtHeatmapStore.build()` is deterministic: two builds from the same
`fixations.json` produce bit-identical `/trials/heatmaps`.

**FR8.6** The store is written in one pass with the full `(N, 16, 24, 32)` array assembled
in RAM. At the expected scale — 10 subjects × ~40 stimuli = 400 trials — that is
400 × 16 × 768 × 4 B ≈ 20 MB uncompressed. If `n_trials × max_length × 768 × 4 > 2 GB`,
`build()` raises `ValueError` rather than attempting a streaming write; a dataset that
large is out of this feature's scope.

### FR9 — Validator and report

**FR9.1** `validate(out_dir, bundle=None) -> dict` raises `BridgeValidationError` on the
first failed invariant, with a message naming the offending record. Checks:

1. `len(X) == len(Y) == len(T) == length` for every record.
2. `1.0 <= X <= 1920.0`, `1.0 <= Y <= 1080.0`.
3. `T` is `int` and `>= 1`.
4. `subject` ∈ `0..N-1`, and `set(subjects) == set(range(N))`.
5. Every `name` **in the `test` split** groups to exactly N records, one per distinct
   subject (FR2.3). A `train`-split name may carry 1..N records — the support split is
   deliberately ragged — but never a duplicate subject. Checked *after* invariant 6, so a
   split-disjointness violation reports as FR3.3 rather than as a grouping failure.
6. `split` ∈ `{"train", "test"}`; train-names ∩ test-names = ∅ (FR3.3).
7. `condition == "freeview"`, `task == "none"`.
8. Every `name` resolves to a file under `<out_dir>/stimuli/`.
9. No `name` contains `jpg` before its extension.
10. Heatmap store row count, order and `fixations_sha256` match `fixations.json`.
11. For 32 randomly sampled trials (seeded): every valid step's heatmap sums to `1.0 ± 1e-5`,
    steps beyond `length` are exactly zero, and each valid step's argmax bin equals the bin
    recomputed directly from `(X, Y)` by FR7.1's arithmetic.

**FR9.2** `bridge_report.json` records: the resolved arguments; `origin_size`;
`support_pool_size`; `support_per_subject`; `min_support_per_subject`;
`num_trials_train` / `num_trials_test`; `num_subjects`; `num_stimuli_train` / `num_stimuli_test`;
`num_trials`; every drop counter from FR2.2/FR2.3; `clamped_coords`; `zero_duration`;
`over_max_length`; `short_scanpath`; `stimulus_image_conflict`; and `fixations_sha256`.
Counters are always present, `0` when nothing fired (D7 — counts are reported, never
absorbed silently).

**FR9.3** A non-zero `short_scanpath` count is echoed to stderr as a warning naming D7,
because those trials are padded inside the frozen evaluator and change what the mean means.

### FR10 — Heatmap metric wrapper

**FR10.1** `score_step_heatmaps(pred_action_prob, gt_heatmaps, lengths, max_length=16)`
returns `{"NSS": float, "CC": float, "KLD": float}`.

- `pred_action_prob`: `torch.Tensor (B, max_length, 24*32 + 1)` — the model's
  `all_actions_prob` exactly as returned. Column 0 (termination) is dropped, the rest
  reshaped to `(B, max_length, 24, 32)`.
- `gt_heatmaps`: `torch.Tensor (B, max_length, 24, 32)` from `GtHeatmapStore.get_batch()`.
- `lengths`: `(B,) int` — ground-truth scanpath lengths.

**FR10.2** Only `(b, t)` pairs with `t < min(lengths[b], max_length)` are scored. Padding
steps carry an all-zero ground-truth map, on which CC and NSS are undefined; including
them would silently deflate every number.

**FR10.3** The three metrics are computed by importing `NSS`, `CC`, `KLD` from
`ISP/OSIE/GazeformerISP/src/models/loss.py` via `importlib` — not copied, not
reimplemented — and calling them on the flattened `(M, 24, 32)` stacks, where
`M = Σ_b min(lengths[b], max_length)`. Those functions already reduce with `.mean()` over
dim 0, so the returned value is a mean over valid steps, not over trials. The report must
say so; it is not the same denominator as the scanpath metrics.

**FR10.4** `M == 0` raises `ValueError` rather than returning `NaN` or `-1`. (`-1` is the
frozen evaluator's uninitialised-cell sentinel — TechStack §4 — and must not be reused here
to mean anything else.)

**FR10.5** A shape mismatch between `pred_action_prob`, `gt_heatmaps` and `lengths` raises
`ValueError` naming all three shapes.

**FR10.6** This module is the only one in the feature that imports `torch`. The bridge core
(`convert`, `heatmaps`, `store`, `validate`) is numpy/h5py/PIL/scipy only and runs on the
Windows dev machine.

### FR11 — CLI

**FR11.1**

```
py -3.8 tools/eve_bridge/build.py --bundle-dir DIR --out-dir DIR
    [--unseen-subjects test01 test02 ...] [--support-pool-size 20] [--seed 0]
    [--skip-stimuli] [--skip-heatmaps]
```

**FR11.2** The command writes `fixations.json`, `subject_id_map.json`, `stimuli/*.jpg`,
`gt_heatmaps.h5`, `bridge_report.json` into `--out-dir`, then runs `validate()` and exits
non-zero on any failure.

**FR11.3** No absolute path appears in any `.py` file (TechStack §6.4); `--bundle-dir` is
required and has no default.

**FR11.4** Re-running with the same `--seed` and `--support-pool-size` reproduces
`fixations.json` byte-for-byte and `gt_heatmaps.h5` bit-for-bit in `/trials/heatmaps` (D5).

---

## Public API Summary

```python
# tools/eve_bridge/heatmaps.py                       (numpy + scipy only)
def build_step_heatmaps(
    X, Y, length: int,
    origin_size=(1080, 1920),
    action_map=(24, 32),
    max_length: int = 16,
    blur_sigma: float = 1,
): ...                                              # -> (16,24,32) f32, (16,) f32

def to_target_scanpath(heatmaps, action_mask): ...  # -> (16, 769) f32


# tools/eve_bridge/store.py                          (numpy + h5py only)
class GtHeatmapStore:
    @classmethod
    def build(cls, fixations, exp_keys, subject_id_map, attrs, **hm_kwargs): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path, fixations_path=None): ...

    trial_keys: "list[str]"
    def get(self, name, subject): ...            # (16,24,32) f32
    def get_batch(self, keys): ...               # (K,16,24,32) f32
    def length_of(self, name, subject): ...      # int
    def exp_key_of(self, name, subject): ...     # str
    def trial_key_of(self, exp_key): ...         # str


# tools/eve_bridge/convert.py                        (numpy + PIL, imports evedataset)
def build_fixations(bundle, unseen_subjects=None, support_pool_size=20,
                    seed=0, origin_size=(1080, 1920)): ...
    # -> (fixations, exp_keys_aligned, subject_id_map, report_counters)

def export_stimuli(bundle, fixations, exp_keys, out_dir): ...   # -> dict of counters


# tools/eve_bridge/validate.py
class BridgeValidationError(RuntimeError): ...
def validate(out_dir, bundle=None, sample_n=32, seed=0): ...    # -> dict


# tools/eve_bridge/heatmap_metrics.py                (the only torch importer)
def score_step_heatmaps(pred_action_prob,   # (B, 16, 769) torch
                        gt_heatmaps,        # (B, 16, 24, 32) torch
                        lengths,            # (B,) int
                        max_length=16): ... # -> {"NSS":…, "CC":…, "KLD":…}
```

---

## Dependencies

| Direction | Artefact | Detail |
|---|---|---|
| reads | `<bundle_dir>/bundle.h5` | via `evedataset.EveBundle.load()`; `samples_df`, `get_scanpath()`, `get_stimulus()` only |
| reads | `evedataset` package | installed wheel; numpy/pandas/h5py/PIL — no torch, no CUDA |
| reads (import only) | `ISP/OSIE/GazeformerISP/src/models/loss.py` | `NSS`, `CC`, `KLD` loaded via `importlib`; file unmodified |
| reads (reference) | `ISP/OSIE/GazeformerISP/src/dataset/dataset.py` | `OSIE.__getitem__` is the numerical reference for FR7.1; not imported, not modified |
| writes | `<out_dir>/fixations.json` | TechStack §3.1 canonical form; consumed by F4, F5 |
| writes | `<out_dir>/subject_id_map.json` | D4 inverse map; consumed by F5's `prediction.json` writer |
| writes | `<out_dir>/stimuli/*.jpg` | flat dir; consumed by F4 `feature_extractor.image_data()` |
| writes | `<out_dir>/gt_heatmaps.h5` | consumed by F5's heatmap-metric block via `score_step_heatmaps()` |
| writes | `<out_dir>/bridge_report.json` | D7 counters; consumed by F7 write-up |
| untouched | `utils/evaluation.py`, `utils/evaltools/*` | frozen (D1) — not imported, not edited |
