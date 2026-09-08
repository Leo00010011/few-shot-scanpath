# Plan — EVE bridge and ground-truth step heatmaps

> Companion to [requirements.md](requirements.md) and [validation.md](validation.md).

---

## Context and Design Decisions

### Why a converter, and why it lives outside the ISP tree

OPEN-1 listed three integration strategies. This spec takes candidate 1, the converter to
`fixations.json`, for the reasons already recorded: least code, upstream `Dataset` classes
untouched, and the output is directly diffable against the shipped OSIE JSON (D2). The
converter needs the `evedataset` wheel, which pulls pandas/h5py/PIL — fine on the dev
machine and inside `isp`, but it has no business being importable from inside
`ISP/OSIE/GazeformerISP/`, whose import surface must stay identical to upstream. So the
code goes in a new top-level `tools/eve_bridge/` package. When F5 creates
`ISP/EVE/GazeformerISP/`, that branch consumes the bridge's *artefacts*, not its code —
except for `heatmap_metrics.score_step_heatmaps()`, which F5 imports directly.

### Why the support/query split is by stimulus, not by trial

The few-shot subject embedding is produced by cross-attention over some of a participant's
own scanpaths. Any scanpath that feeds that embedding has leaked into the model's
conditioning and can no longer be scored. Splitting *by trial* would leak across the
stimulus axis: subject 0's support trial on image A conditions the embedding used to
predict subject 0's behaviour on image A for a different subject's row of the score matrix.
Splitting *by stimulus* removes the whole image from the query pool for every subject at
once, which is the only partition under which "the support scanpaths were never scored"
is true for all N × N cells of the evaluator's matrix. FR3.3 asserts it rather than
trusting it.

The pool is deliberately larger than `num_fewshot` (20 vs 10) because
`select_fewshot_subject()` re-samples the support images per `--random_support` seed. With
`support_pool_size == num_fewshot` every seed draws the same set and averaging over
`random_support` repeats measures nothing; with a larger pool the draws vary while
remaining inside the pool, hence still disjoint from `test`.

### Why squash rather than letterbox (OPEN-4)

EVE is 16:9, OSIE is 4:3, and the frozen metric configuration is welded to a 512×384
screen with a 16×12 ScanMatch grid (TechStack §4). Changing `args.width/height` would
change the bin *shape* and forfeit comparability to the published table — the entire point
of the exercise per Mission §1. Letterboxing would keep the grid but hand ~25 % of the bins
to dead bars that still count in ScanMatch's string quantisation. Squashing distorts the
stimulus but leaves every metric parameter untouched, and it matches what the sibling EVE
pipeline already does (`configs/data/eve.yaml` resizes 1920×1080 to a square). The
distortion is recorded in `bridge_report.json` and is F7's problem to state, not the
bridge's to hide.

Concretely the bridge does **not** rescale anything: it writes X/Y in original 1920×1080
space (D3) and declares `origin_size=(1080, 1920)`. The squash happens inside
`OSIE_evaluation`'s `resizescale_x/y` division, which F5 will hit once it passes
`origin_size` explicitly instead of inheriting the `(600, 800)` default. This spec's job is
to make that number available and unmissable.

### Why the heatmaps are per-timestep 24×32 and built here

The model's spatial output is `all_actions_prob` of shape `(B, 16, 24*32+1)` — one
probability map per decode step, plus a termination column. The only ground truth with a
matching shape is the per-step blurred fixation map that `OSIE.__getitem__` already builds
for training. EVE ships no saliency maps, so that construction *is* the answer to "where
does the ground-truth heatmap come from" — it is derived from the scanpath coordinates by
one-hot placement plus `gaussian_filter(σ=1)` and sum-normalisation.

Two consequences shape the design:

1. The builder must be numerically identical to upstream, including the truncating
   `astype(np.int32)` discretisation and the odd termination-slot rule in `action_mask`.
   It is re-derived in a standalone function rather than imported, because importing
   `dataset.py` drags in torch, torchvision, skimage and matplotlib — none of which the
   bridge should need. Validation Group 2 pins the two implementations against each other
   numerically so the copy cannot drift silently.
2. The maps are cached rather than recomputed. F5's eval loop is per-image-batch and
   already GPU-bound; recomputing 16 Gaussian filters per trial per epoch-equivalent is
   pure waste, and a cache keyed to `fixations.json` order lets the numbers be re-derived
   offline on a laptop from artefacts alone (D5) exactly as F6 wants for the scanpath
   metrics.

The store's key is `"{name}|{subject}"` with the **dense** subject, because that is the
identity F5 has in hand inside the eval loop. `exp_key` rides along in a parallel dataset
purely so any row can be traced back to the real EVE recording (D4) — that is what makes
"which of my real subjects is row 3?" answerable from the cache alone.

### Why NSS/CC/KLD come from `models/loss.py`

D1 freezes `utils/evaluation.py` and `utils/evaltools/*`. `models/loss.py` is not frozen,
but it already contains exactly these three functions with the epsilon handling and the
normalisation the authors used, and the COCO branches' `test.py` already imports them for
the same purpose. Reimplementing them would produce numbers that are ours, not theirs.
The wrapper therefore loads that module by path via `importlib` and calls into it; it adds
only the masking of padding steps (FR10.2), which is a call-site concern, not a metric
change — the additive-over-invasive rule in TechStack §6.2.

Note the denominator asymmetry this creates and which FR10.3 requires the report to state:
the scanpath metrics average over (image, subject) cells, while NSS/CC/KLD average over
valid *timesteps*. They are not two views of the same mean.

### What this spec deliberately does not do

No feature extraction, no SE-Net, no eval branch, no metric run. Those are F4, F3, F5, F6
and each needs its own spec. F1 (the OSIE baseline) is still the roadmap's next feature and
is not blocked by any of this — the bridge can be built and validated on Windows while the
cluster environment for F1 is still being stood up.

---

## Implementation Steps

### Step 1 — `tools/eve_bridge/__init__.py`

Empty package marker plus the module docstring stating: Stage A only, CPU/Windows-runnable,
never imports from `ISP/*/GazeformerISP/src/utils/`. Add `tools/` to the repo (it does not
exist yet). No new entry in `.gitignore` — these are source files and small JSON artefacts;
the generated `<out_dir>` lives outside the repo or under an ignored path chosen by the
operator, and `*.h5` should be added to `.gitignore` alongside the existing `*.pt`/`*.pth`
rules if it is not covered.

### Step 2 — `tools/eve_bridge/heatmaps.py`

No dependencies on the rest of the feature. Pure numpy + `scipy.ndimage`.

```python
def build_step_heatmaps(X, Y, length, origin_size=(1080, 1920),
                        action_map=(24, 32), max_length=16, blur_sigma=1):
    downscale_x = origin_size[1] / action_map[1]      # 60.0
    downscale_y = origin_size[0] / action_map[0]      # 45.0

    heatmaps    = np.zeros((max_length, *action_map), dtype=np.float32)
    action_mask = np.zeros(max_length, dtype=np.float32)

    pos_x = np.asarray(X, dtype=np.float32)
    pos_y = np.asarray(Y, dtype=np.float32)
    n = min(int(length), max_length)

    for i in range(n):
        col = ((pos_x[i] - 1) / downscale_x).astype(np.int32)   # truncation, as upstream
        row = ((pos_y[i] - 1) / downscale_y).astype(np.int32)
        if not (0 <= row < action_map[0] and 0 <= col < action_map[1]):
            raise ValueError(...)                               # FR7.4
        heatmaps[i, row, col] = 1.0
        if blur_sigma:
            heatmaps[i] = filters.gaussian_filter(heatmaps[i], blur_sigma)
            heatmaps[i] /= heatmaps[i].sum()
        action_mask[i] = 1.0

    if action_mask.sum() <= max_length - 1:                     # upstream termination slot
        action_mask[int(action_mask.sum())] = 1.0

    return heatmaps, action_mask
```

`to_target_scanpath(heatmaps, action_mask)` allocates `(max_length, prod(action_map) + 1)`
float32, sets `col 0 = 1.0` where `heatmaps[i].sum() == 0`, else writes
`heatmaps[i].reshape(-1)` into `cols 1:`. It takes `action_mask` only for the shape/consistency
assertion; the termination flag is derived from the map being empty, matching upstream's
`pos_x_discrete[index] == -1` branch.

Keep the loop scalar and un-vectorised — matching upstream's arithmetic exactly matters more
than speed at 400 trials × 16 steps (D1's spirit, TechStack §6.2).

### Step 3 — `tools/eve_bridge/convert.py`

Depends on Step 2 only through the caller, not directly.

```python
def build_fixations(bundle, unseen_subjects=None, support_pool_size=20,
                    seed=0, origin_size=(1080, 1920)):
```

Order of operations:

1. `df = bundle.samples_df`. Resolve `unseen_subjects` (default: sorted ids starting
   `"test"`). FR1.4 validation via `difflib.get_close_matches` for the error message.
   FR1.5 length check. Build `subject_id_map` from the sorted list.
2. `cand = df[df.subject.isin(subjects) & df.valid]`.
3. For each candidate row, in `samples_df` order: `sp = bundle.get_scanpath(exp_key)`.
   Apply the FR2.2 drop rules, accumulating `counters["empty_scanpath"]` etc. Keep a
   `per_stimulus: dict[str, dict[dense_subject, record_draft]]`.
4. Equal-subject filter (FR2.3): `complete = [s for s, d in per_stimulus.items() if len(d) == N]`;
   `counters["incomplete_stimulus"] = len(per_stimulus) - len(complete)`.
5. FR2.5 `jpg`-substring check over `complete`. FR2.4 pool-size check.
6. Partition: `names = sorted(complete)`; `random.Random(seed).shuffle(names)`;
   `train_names, test_names = names[:support_pool_size], names[support_pool_size:]`;
   assert disjoint (FR3.3).
7. Materialise records. Per trial, convert with a small helper:

```python
def _convert_scanpath(sp, origin_size, counters):
    x = sp[2].astype(np.float64) + 1.0        # FR4.2
    y = sp[3].astype(np.float64) + 1.0
    t = sp[1].astype(np.float64)
    H, W = origin_size
    counters["clamped_coords"] += int((x < 1).sum() + (x > W).sum()
                                      + (y < 1).sum() + (y > H).sum())
    x = np.clip(x, 1.0, float(W)); y = np.clip(y, 1.0, float(H))
    T = [int(round(v)) for v in t]
    n_zero = sum(1 for v in T if v == 0); counters["zero_duration"] += n_zero
    T = [max(v, 1) for v in T]
    return [float(v) for v in x], [float(v) for v in y], T
```

8. Sort records by `(name, subject)` (FR5.3); build `exp_keys_aligned` in the same order.
9. Fill `over_max_length` / `short_scanpath` counters; return
   `(fixations, exp_keys_aligned, subject_id_map, counters)`.

`export_stimuli(bundle, fixations, exp_keys, out_dir)` walks unique `name`s in sorted
order, writes `<out_dir>/stimuli/<name>`, and does the FR6.4 SHA-256 conflict count by
hashing `arr.tobytes()` per exp_key sharing the stimulus.

### Step 4 — `tools/eve_bridge/store.py`

Depends on Step 2 (`build_step_heatmaps`) and Step 3's output shape.

```python
class GtHeatmapStore:
    @classmethod
    def build(cls, fixations, exp_keys, subject_id_map, attrs, **hm_kwargs):
        # FR8.6 size guard first
        # for each record i: heatmaps[i], action_mask[i] = build_step_heatmaps(
        #     rec["X"], rec["Y"], rec["length"], **hm_kwargs)
        # trial_key[i] = f'{rec["name"]}|{rec["subject"]}'
        # length[i] = min(rec["length"], max_length)
```

`save()` writes the FR8.1 layout with `h5py.special_dtype(vlen=str)` for the two string
datasets and `compression="gzip", compression_opts=4, chunks=(1, max_length, *action_map)`
on `/trials/heatmaps`. `load()` reads everything eagerly (20 MB) into numpy and builds two
dicts — `trial_key -> row` and `exp_key -> trial_key` — so `get`, `exp_key_of` and
`trial_key_of` are O(1) and never re-open the file. When `fixations_path` is given, hash it
and compare against the stored `fixations_sha256` (FR8.2), raising `ValueError` with both
hashes on mismatch.

`get_batch(keys)` builds the row index list first and raises `KeyError` on the *first*
missing key before touching the array, so a partial result is never returned.

### Step 5 — `tools/eve_bridge/validate.py`

Depends on Steps 2–4.

```python
class BridgeValidationError(RuntimeError): ...

def validate(out_dir, bundle=None, sample_n=32, seed=0) -> dict:
```

Runs FR9.1 checks 1–11 in order, raising `BridgeValidationError(f"...record {i} ({name}, subject {s}): ...")`
on the first failure. Check 11 re-derives the expected bin with the same arithmetic as
Step 2 and compares against `np.unravel_index(hm[t].argmax(), hm[t].shape)`. Returns a dict
of `{check_name: "ok"}` plus the sampled trial keys, which the CLI merges into
`bridge_report.json` under `"validation"`.

`bundle` is optional: when supplied, check 11 additionally re-reads `get_scanpath()` for the
sampled trials and confirms `X - 1 == sp[2]` after clamping, closing the loop back to the
source of truth rather than only checking internal consistency.

### Step 6 — `tools/eve_bridge/heatmap_metrics.py`

Depends on Step 4 for its input shape. The only torch importer.

```python
def _loss_module():
    """Load ISP/OSIE/GazeformerISP/src/models/loss.py by path, without importing the branch."""
    root = Path(__file__).resolve().parents[2]
    path = root / "ISP" / "OSIE" / "GazeformerISP" / "src" / "models" / "loss.py"
    spec = importlib.util.spec_from_file_location("isp_osie_loss", path)
    ...  # cached in a module-level global

def score_step_heatmaps(pred_action_prob, gt_heatmaps, lengths, max_length=16):
    # shape checks (FR10.5)
    # valid = [(b, t) for b in range(B) for t in range(min(lengths[b], max_length))]
    # pred = pred_action_prob[:, :, 1:].reshape(B, max_length, 24, 32)
    # P = torch.stack([pred[b, t] for b, t in valid])        # (M, 24, 32)
    # G = torch.stack([gt_heatmaps[b, t] for b, t in valid])
    # if M == 0: raise ValueError (FR10.4)
    # return {"NSS": float(L.NSS(P, G)), "CC": float(L.CC(P, G)), "KLD": float(L.KLD(P, G))}
```

Note the argument order the upstream functions expect: `NSS(input, fixation)`,
`CC(input, salmap)`, `KLD(input, salmap)` — prediction first, ground truth second in all
three. Getting this backwards silently produces a plausible but wrong KLD (it is not
symmetric), so Group 5 pins it with an asymmetry test.

### Step 7 — `tools/eve_bridge/build.py` (CLI)

Depends on Steps 3–5. Argparse per FR11.1, then:

```
bundle = EveBundle.load(args.bundle_dir)
fixations, exp_keys, subject_map, counters = build_fixations(...)
write fixations.json (indent=4)  ->  sha256
write subject_id_map.json
if not skip_stimuli:  counters |= export_stimuli(...)
if not skip_heatmaps: GtHeatmapStore.build(...).save(out/gt_heatmaps.h5)
write bridge_report.json (args + origin_size + counters + sha256)
report["validation"] = validate(out_dir, bundle=bundle)
warn on short_scanpath > 0 (FR9.3)
```

Exit 1 on `BridgeValidationError`, printing the message. Print a one-screen summary:
subjects, stimuli train/test, trials, and every non-zero counter.

### Step 8 — Tests, `tests/eve_bridge/`

pytest, CPU, no bundle required except where marked. Fixtures build a tiny synthetic
`fixations`-shaped list and a fake bundle object exposing `samples_df`, `get_scanpath`,
`get_stimulus` — the three methods FR-Dependencies names — so the whole bridge is testable
without the 100 GB dataset. One `@pytest.mark.bundle` group takes `--bundle-dir` and runs
the Data Validity checks from [validation.md](validation.md) against real data.

Group 2's parity test imports the upstream `OSIE.__getitem__` arithmetic by transcribing it
inline in the test (not by importing `dataset.py`) and asserts bitwise-equal heatmaps.

### Step 9 — Documentation

Append to the spec folder only. Do **not** edit the constitution here; instead the spec's
closing note records that OPEN-3 is now partially answered (`support_pool_size=20`,
`num_fewshot=10`, split by stimulus) and OPEN-4 is answered (squash), so a later
constitution update can pick them up with the rationale already written down in
[requirements.md](requirements.md) FR3/FR4.

---

## Implementation Order

1. `tools/eve_bridge/__init__.py` — package skeleton.
2. `tools/eve_bridge/heatmaps.py` — `build_step_heatmaps`, `to_target_scanpath`.
3. `tools/eve_bridge/convert.py` — `build_fixations`, `export_stimuli`.
4. `tools/eve_bridge/store.py` — `GtHeatmapStore`.
5. `tools/eve_bridge/validate.py` — `validate`, `BridgeValidationError`.
6. `tools/eve_bridge/heatmap_metrics.py` — `score_step_heatmaps`.
7. `tools/eve_bridge/build.py` — CLI wiring all of the above.
8. `tests/eve_bridge/` — Groups 1–5 from validation.md.
9. Spec closing note on OPEN-3 / OPEN-4.
