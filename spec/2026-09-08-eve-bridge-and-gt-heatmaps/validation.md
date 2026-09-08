# Validation — EVE bridge and ground-truth step heatmaps

> Companion to [requirements.md](requirements.md) and [plan.md](plan.md).
> Everything here runs on Windows CPU (`py -3.8 -m pytest tests/eve_bridge`). Checks marked
> **[bundle]** need `--bundle-dir` pointing at a real EVE bundle; the rest run against the
> synthetic fixtures from plan.md Step 8.

---

## Code Correctness

### Group 1 — Heatmap construction (`heatmaps.py`)

- [ ] `build_step_heatmaps([61.0], [46.0], 1)` returns `heatmaps.shape == (16, 24, 32)`,
      `heatmaps.dtype == np.float32`, `action_mask.shape == (16,)`,
      `action_mask.dtype == np.float32`.
- [ ] With `X=[61.0], Y=[46.0]` (→ `col = int(60/60) = 1`, `row = int(45/45) = 1`) the
      argmax of `heatmaps[0]` is `(1, 1)`. Off-by-one in the `-1` of FR4.2/FR7.1 moves it
      to `(0, 0)` or `(1, 0)`.
- [ ] `heatmaps[0].sum() == pytest.approx(1.0, abs=1e-5)` for every fixation of a
      length-5 scanpath; `heatmaps[5:].sum() == 0.0` exactly.
- [ ] `action_mask` for `length=5` is `[1]*5 + [1] + [0]*10` — five fixation slots plus the
      one termination slot. For `length=16` it is `[1]*16` with **no** extra slot (the
      `sum() <= max_length - 1` guard does not fire). Both cases asserted separately.
- [ ] `blur_sigma=0` yields a one-hot map: exactly one cell equal to `1.0`, all others `0.0`.
- [ ] Boundary coordinates `X=1.0, Y=1.0` map to bin `(0, 0)`; `X=1920.0, Y=1080.0` map to
      bin `(23, 31)`. `X=1921.0` raises `ValueError` (FR7.4), not a silent clip.
- [ ] `length` greater than `max_length` is honoured as `min(length, max_length)`: a
      length-20 scanpath produces 16 non-zero step maps and no `IndexError`.
- [ ] `to_target_scanpath` returns `(16, 769) float32`; for `length=5`,
      `out[:5, 0] == 0.0`, `out[5:, 0] == 1.0`, and
      `np.allclose(out[2, 1:], heatmaps[2].reshape(-1))`.

### Group 2 — Parity with upstream `OSIE.__getitem__` (D1 spirit)

- [ ] For 200 randomly generated scanpaths (seeded; lengths 1–20, coordinates uniform in
      `[1, 1920] × [1, 1080]`), `build_step_heatmaps(...)` output is **bitwise equal**
      (`np.array_equal`) to the heatmap produced by the arithmetic transcribed verbatim
      from `ISP/OSIE/GazeformerISP/src/dataset/dataset.py::OSIE.__getitem__` lines 120–147
      with `origin_size=(1080, 1920)`, `action_map=(24, 32)`, `blur_sigma=1`. Any drift —
      rounding instead of truncation, normalising before blurring, a different filter mode —
      fails here.
- [ ] The same 200 cases agree on `action_mask` bitwise, including the termination slot.
- [ ] `git diff --exit-code ISP/OSIE/GazeformerISP/src/utils/ ISP/OSIE/GazeformerISP/src/models/loss.py`
      is clean after the feature is implemented — the frozen files and the borrowed loss
      module are untouched (D1, FR10.3).

### Group 3 — Conversion and split (`convert.py`)

- [ ] Fake bundle with 3 subjects × 5 stimuli, `support_pool_size=2`: `build_fixations`
      returns 15 records; 6 with `split == "train"` (2 stimuli × 3 subjects) and 9 with
      `split == "test"`.
- [ ] `set(train_names) & set(test_names) == set()` (FR3.3). Also asserted at the *record*
      level: no `(name, subject)` pair appears in both splits.
- [ ] Every record's key order is exactly
      `["name","subject","X","Y","T","length","split","condition","task"]`, and
      `condition == "freeview"`, `task == "none"`.
- [ ] `len(X) == len(Y) == len(T) == length` for all records; `X`/`Y` are `float`,
      `T`/`length`/`subject` are `int` (assert `type(...) is int`, not just `isinstance`
      — numpy scalars must not leak into the JSON).
- [ ] A fake scanpath with `x_px = -5.0` produces `X == 1.0` and increments
      `counters["clamped_coords"]` by 1; the trial is still emitted (FR4.3).
- [ ] A fake scanpath with `duration_ms = 0.4` produces `T == 1` and increments
      `counters["zero_duration"]`.
- [ ] `X[i] == pytest.approx(sp[2][i] + 1.0)` for an unclamped fixation (FR4.2). A test
      that would pass without the `+1` is not acceptable — use a coordinate whose bin
      changes under the shift.
- [ ] A stimulus missing for one subject is dropped for *all* subjects and
      `counters["incomplete_stimulus"] == 1` (FR2.3).
- [ ] Empty scanpath, non-finite coordinate, and empty `stimulus_path` each increment their
      own counter and drop only that trial (FR2.2).
- [ ] `unseen_subjects=["test01", "nope"]` raises `ValueError` whose message contains
      `"nope"` and at least one close match (FR1.4).
- [ ] `unseen_subjects=["test01"]` raises `ValueError` (FR1.5).
- [ ] A stimulus named `"a-jpg-thing"` raises `ValueError` (FR2.5).
- [ ] Fewer than `support_pool_size + 1` surviving stimuli raises `ValueError` naming both
      counts (FR2.4).
- [ ] Two runs with `seed=0` produce identical `fixations` lists; `seed=1` produces a
      different train/test partition (with ≥5 stimuli the probability of collision is
      small enough to assert directly).
- [ ] Subject dense ids: `unseen_subjects=["test03","test01","test02"]` yields
      `to_dense == {"test01":0, "test02":1, "test03":2}` — argument order is irrelevant
      (FR1.2).

### Group 4 — HDF5 store roundtrip (`store.py`)

- [ ] `GtHeatmapStore.build(...).save(p)` then `load(p)`: `len(trial_keys) == len(fixations)`
      and `trial_keys[i] == f'{fixations[i]["name"]}|{fixations[i]["subject"]}'` for all `i`
      (FR8.2).
- [ ] `/trials/heatmaps` has `shape == (N, 16, 24, 32)`, `dtype == float32`,
      `chunks == (1, 16, 24, 32)`, and gzip compression is present in
      `dset.compression`.
- [ ] Root attrs contain `origin_size == [1080, 1920]`, `action_map == [24, 32]`,
      `max_length == 16`, `blur_sigma == 1.0`, a parseable ISO-8601 `created_utc`, and a
      64-hex-char `fixations_sha256`.
- [ ] `store.get(name, subject)` equals the in-memory `build_step_heatmaps` output for that
      record (`np.array_equal`), i.e. compression and HDF5 round-trip are lossless.
- [ ] `store.get("missing.jpg", 0)` raises `KeyError` containing the trial key.
      `store.get_batch([good, missing])` raises `KeyError` and returns nothing partial.
- [ ] `store.get_batch([k2, k0, k1])` returns rows in the requested order, not stored order.
- [ ] `load(p, fixations_path=<regenerated different json>)` raises `ValueError` mentioning
      both hashes (FR8.2). `load(p, fixations_path=<the original>)` succeeds.
- [ ] Two `build()` calls on the same input produce byte-identical `/trials/heatmaps`
      arrays (FR8.5).
- [ ] `n_trials` large enough to exceed the 2 GB guard raises `ValueError` (FR8.6) —
      tested by monkeypatching the threshold, not by allocating 2 GB.

### Group 5 — Heatmap metrics (`heatmap_metrics.py`)

- [ ] `score_step_heatmaps` returns a dict with exactly keys `{"NSS","CC","KLD"}`, all
      Python `float` (not tensors).
- [ ] **Perfect-prediction case**: feed `pred_action_prob` whose `[:, :, 1:]` is the
      ground-truth heatmap itself (renormalised over 769 columns is not required — the
      metrics normalise internally). Then `CC == pytest.approx(1.0, abs=1e-4)` and
      `KLD == pytest.approx(0.0, abs=1e-4)`. NSS is positive and well above the
      uniform-prediction value.
- [ ] **Uniform-prediction case**: `pred[:, :, 1:]` constant. `CC == pytest.approx(0.0, abs=1e-4)`
      and NSS is approximately 0. This is the floor the real numbers must beat for the
      metric block to mean anything.
- [ ] **Argument-order pin**: `KLD(pred, gt) != KLD(gt, pred)` on an asymmetric pair, and
      the wrapper's value equals the former. Guards against the silent
      prediction/ground-truth swap called out in plan.md Step 6.
- [ ] **Masking**: a batch where `lengths = [2, 16]` scores `M == 18` steps. Changing the
      *padding* steps of trial 0 (indices 2–15) does not change any returned number
      (FR10.2). Without the mask, CC would move.
- [ ] `lengths = [0, 0]` raises `ValueError` (FR10.4), not `NaN`, and the message does not
      use `-1`.
- [ ] Mismatched shapes — `pred (4,16,769)` with `gt (3,16,24,32)` — raise `ValueError`
      naming all three shapes (FR10.5).
- [ ] `pred_action_prob` of width `768` (termination column already stripped by a caller)
      raises `ValueError` rather than silently misaligning every bin.
- [ ] The module imports `NSS`/`CC`/`KLD` from the on-disk `models/loss.py`: assert
      `_loss_module().__file__` ends with
      `ISP/OSIE/GazeformerISP/src/models/loss.py` (FR10.3).

### Group 6 — CLI and validator (`build.py`, `validate.py`)

- [ ] Running `build.py` against the fake bundle writes exactly
      `fixations.json`, `subject_id_map.json`, `bridge_report.json`, `gt_heatmaps.h5`,
      `stimuli/` and nothing else into `--out-dir`.
- [ ] `--skip-stimuli` omits `stimuli/`; the run then fails validation check 8 with a
      `BridgeValidationError` naming the first missing file, and exits non-zero. (Skipping
      is for iterating on the JSON, and must not silently pass.)
- [ ] `--skip-heatmaps` omits `gt_heatmaps.h5` and skips checks 10–11 without failing.
- [ ] Each of FR9.1's 11 checks has a negative test: corrupt one field in a written
      `fixations.json`, re-run `validate`, assert `BridgeValidationError` whose message
      names the record index and the specific invariant.
- [ ] `bridge_report.json` contains every counter named in FR9.2, all present with value
      `0` on a clean fixture run (no counter is conditionally omitted).
- [ ] `short_scanpath > 0` emits a stderr warning containing `"D7"`.
- [ ] Two identical invocations produce byte-identical `fixations.json` (FR11.4).
- [ ] `grep -rn "C:\\\\\|/mnt/\|/home/" tools/eve_bridge/*.py` returns nothing (FR11.3).

---

## Data Validity

Run against the real bundle **[bundle]**; each is a notebook cell or a
`@pytest.mark.bundle` test. Each states the expected outcome — a deviation is a finding to
record in `bridge_report.json` and carry into F7, not something to silence.

- [ ] **Subject coverage.** With the default `test*` selection, `num_subjects == 10` and
      `set(to_dense.values()) == set(range(10))`. If EVE's test partition on this bundle is
      not 10 participants, record the actual number — it becomes `--subject_num` for F5.
- [ ] **Stimulus coverage.** `num_stimuli_train + num_stimuli_test` should be close to the
      ~60 image stimuli EVE presents per participant. A number far below 60 means the
      equal-subject filter (FR2.3) is discarding heavily; report
      `incomplete_stimulus` alongside it. Expected: `incomplete_stimulus` small relative to
      the total; if it exceeds ~25 % of candidate stimuli, stop and reconsider the subject
      selection rather than proceeding to F4.
- [ ] **Trial count identity.** `num_trials == num_subjects × (num_stimuli_train + num_stimuli_test)`
      exactly. Any other value means the equal-subject invariant leaked.
- [ ] **Scanpath length distribution.** Report min / median / max `length`. Expected median
      in the 5–12 range for a 3 s free-viewing exposure. `short_scanpath` (length < 3) is
      reported explicitly — these are padded with `(1., 1., 1e-3)` inside the frozen
      evaluator (D7) and inflate nothing visibly.
- [ ] **Duration sanity.** All `T` in `[100, 1200]` ms — the EVE fixation pipeline already
      filters to ≥100 ms and ≤1200 ms, so anything outside means the wrong array row was
      read (FR4.1 reads row 1, not row 0).
- [ ] **Coordinate distribution.** Histogram `X` and `Y`. Expect a central bias typical of
      free viewing and no mass piled on the clamp boundaries. `clamped_coords` should be a
      small fraction of total fixations; a large value means off-screen gaze was projected
      into the bundle and needs a decision, not a clip.
- [ ] **Cross-check against the source.** For 20 random trials, re-read
      `bundle.get_scanpath(exp_key)` and assert `X == sp[2] + 1` and `Y == sp[3] + 1`
      (modulo clamping) and `T == round(sp[1])`. This is the only check that closes the
      loop back to EVE rather than testing the bridge against itself.
- [ ] **Stimulus files.** Every exported `.jpg` opens, is `1920 × 1080`, RGB.
      `stimulus_image_conflict == 0` expected; a non-zero value means two participants saw
      different renderings of the same `stimulus_name` and F4's feature extraction would be
      ambiguous.
- [ ] **Heatmap mass.** Over the whole store, `heatmaps.sum(axis=(2,3))` is `1.0 ± 1e-5` at
      every step below `length` and exactly `0.0` at or above it. Report the count of
      violations — expected `0`.
- [ ] **Heatmap spatial coverage.** Aggregate `heatmaps.sum(axis=(0,1))` into a single
      24×32 map and plot it. Expect a centre-biased blob, not a uniform sheet and not mass
      concentrated in one corner (a corner peak indicates the row/col swap or the missing
      `-1`).
- [ ] **Diffability against OSIE.** Load the shipped
      `ISP/OSIE/GazeformerISP/src/data/fixations.json` and assert our records have the
      identical key set and per-key Python types (D2). Value ranges differ by design
      (800×600 vs 1920×1080); the schema must not.
- [ ] **Loader smoke test.** Instantiate `OSIE_evaluation` against the produced
      `fixations.json` with `origin_size=(1080, 1920)`, `resize=(384, 512)`,
      `type="test"`, `fewshot_subject=list(range(num_subjects))`, and a stub feature dir.
      Assert `len(dataset) == num_stimuli_test` and that one `__getitem__` yields
      `len(fix_vectors) == num_subjects` with dtype
      `[('start_x','f8'),('start_y','f8'),('duration','f8')]`, coordinates inside
      `[0, 512] × [0, 384]`, and durations in seconds (< 2.0). This is the contract F5
      depends on and the cheapest place to catch a units error.

---

## Data Architecture Integrity

Keying invariants. These are the checks that make "which of my real subjects is row 3?"
answerable (D4) and prevent a stale cache from being scored as if it were fresh.

- [ ] **exp_key roundtrip.** For every row `i` in the store:
      `trial_key_of(exp_key[i]) == trial_key[i]` and
      `exp_key_of(*parse(trial_key[i])) == exp_key[i]`. Both directions, all rows, no
      sampling.
- [ ] **exp_key uniqueness.** `len(set(exp_key)) == len(exp_key)` — one store row per EVE
      recording. A duplicate means two `fixations.json` records were built from the same
      trial.
- [ ] **No phantom keys.** `set(store.trial_keys)` equals
      `{f'{r["name"]}|{r["subject"]}' for r in fixations}` — equality in both directions,
      so the store neither invents a key nor drops one.
- [ ] **Dense-id roundtrip.** For every store row, `to_eve[str(subject[i])]` is the EVE
      participant prefix of `exp_key[i]` (EVE exp_keys are `f"{subject}_{step}"`). This is
      the check that would have caught an off-by-one in the dense remapping — P2 in
      Mission §2.
- [ ] **Order verification is not bypassable.** `load()` with a `fixations_path` whose
      records are *reordered but otherwise identical* must fail: the SHA-256 changes even
      though the record set does not. Assert this explicitly — an order-insensitive check
      (e.g. hashing a sorted set of keys) would let a mis-indexed cache through, since the
      store addresses rows positionally.
- [ ] **Split labels do not leak into keys.** `trial_key` contains no split information, so
      the same key resolves identically whether the trial is support or query. Assert that
      `store.get(name, subject)` works for both a train-split and a test-split record.
- [ ] **Support/query disjointness survives the roundtrip.** Re-read the written
      `fixations.json`, group names by split, and assert the intersection is empty (FR3.3
      re-checked on the artefact, not only in memory). This is the invariant that makes the
      few-shot embedding legitimate; it is cheap and must be checked at the artefact level
      because F3 and F5 read the artefact, not the builder.
- [ ] **`subject_id_map.json` is a bijection.** `to_dense` and `to_eve` invert each other,
      `set(to_dense.values()) == set(range(N))`, and every subject appearing in
      `fixations.json` is a key of `to_eve`.
