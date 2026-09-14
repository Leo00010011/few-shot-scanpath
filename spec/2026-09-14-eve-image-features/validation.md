# F4 — Validation

> Spec folder 3 of 3: [requirements.md](requirements.md) · [plan.md](plan.md) · **validation**

Groups 1–5 are pytest, CPU, Windows-runnable unless marked. Group 6 needs the real bundle. Group 7
needs the cluster run. Data Validity and Data Architecture Integrity are run against the artefacts the
real run produces.

---

## Code Correctness

### Group 1 — Preprocessing parity (the load-bearing group)

- [ ] **FR4.4 — bit-identity with upstream.** With a synthetic 1080×1920×3 JPEG, `extract_one()` on
      the array the JPEG decodes to and upstream `image_data()` on a directory containing that file
      produce tensors satisfying `torch.equal`. Not `allclose` — **exact equality**. Compare against
      the *decoded* array, never the pre-encode one (JPEG is lossy; that failure would look like a
      transform defect and is not one). **Unmarked, runs everywhere (FR14.3).** If this fails, the
      FR4.3 transcription is wrong and nothing else in F4 is trustworthy.
- [ ] `preprocess()` returns `(1, 3, 768, 1024)` float32 for a `(1080, 1920, 3)` uint8 input.
- [ ] `preprocess()` raises `EvePrepError` naming the shape for `(1080, 1920)`, `(1080, 1920, 4)`,
      `(600, 800, 3)`, and for a `float32` input — four separate cases (FR3.2).
- [ ] `extract_one()` returns `(768, 2048)` float32 **on CPU** (`t.device.type == "cpu"`), and the
      tensor is not a view into CUDA memory (`t.is_cuda is False`).
- [ ] `extract_one()` raises `EvePrepError` when the backbone returns a wrong-shaped tensor
      (monkeypatched backbone returning `(700, 2048)`); the message contains `(768, 2048)`.
- [ ] `preprocess()` output is unchanged when the same array is passed twice (no in-place mutation of
      the caller's array — assert the input array is bytewise unchanged after the call).

### Group 2 — Trial → exp_key mapping

- [ ] `load_trial_exp_keys()` on the synthetic store returns a dict of the expected length with
      `(name, subject)` tuple keys, `subject` an `int`, and exp_key `str` — assert types, not just
      values.
- [ ] `parse_trial_key("a|b|c.jpg|7")` returns `("a|b|c.jpg", 7)` — splits on the **last** separator
      (FR2.2). `parse_trial_key("noseparator")` and `parse_trial_key("x|notanint")` each raise.
- [ ] A store with a repeated exp_key raises `EvePrepError` containing `"duplicate exp_key"` and the
      offending key (FR2.3).
- [ ] `load_trial_exp_keys(store, fixations)` raises when the store's `fixations_sha256` attr does
      not match the file, and the message shows **both** hashes (FR2.5).
- [ ] `load_trial_exp_keys()` never reads `trials/heatmaps`: monkeypatch `h5py.Dataset.__getitem__`
      to raise if called on that dataset, or assert the peak resident growth is under ~5 MB. The
      cheap version — assert `"heatmaps"` appears nowhere in the module source — is acceptable and
      is what the F5 reuse actually depends on.
- [ ] `derive_trial_exp_keys()` on the synthetic `samples_df` reproduces the synthetic store's
      mapping exactly.
- [ ] `derive_trial_exp_keys()` raises when a `(stimulus_name, eve_subject)` pair matches **zero**
      rows, and separately when it matches **two** — the message states the count (FR2.4).
- [ ] `crosscheck_exp_keys()` returns `{"agree": True, "n": N}` on agreement.
- [ ] `crosscheck_exp_keys()` raises **three distinguishable** messages: a trial present only in the
      store, a trial present only in the derivation, and a trial where the two disagree on the value.
      A single generic "mismatch" is a failure of this test — the three mean different upstream
      problems.
- [ ] `split_of()` returns every `(name, subject)` with its split and raises on a duplicate trial key
      in `fixations.json`.

### Group 3 — Filename and keying rules

- [ ] `exp_key_filename("train24_step059")` == `"train24_step059.pth"`.
- [ ] `exp_key_filename("has_jpg_inside")` raises, and the message cites the unanchored
      `str.replace('jpg','pth')` (FR5.2).
- [ ] `exp_key_filename("bad-key!")`, `exp_key_filename("with space")`, `exp_key_filename("")` each
      raise on the charset rule.
- [ ] Round-trip: for every key in the real mapping, `exp_key_filename(k)[:-4] == k`.

### Group 4 — Preflight (`check_features.py`)

- [ ] A complete synthetic cache returns exit `0` and a summary whose `n_checked` equals the number
      of trials in the selected split, with `n_test + n_train == n_checked` when `--split both`.
- [ ] One deleted `.pth` → exit `1`, `EvePrepError` message contains the missing relative path and
      the total count (FR7.2).
- [ ] A tensor saved as `(700, 2048)` → exit `1` with a **bad-shape** message, distinct from the
      missing-file message.
- [ ] A tensor saved as float64 of the right shape → bad-shape branch (dtype is checked, FR7.2).
- [ ] `--split test` passes while train-split features are entirely absent, and vice versa.
- [ ] A trial present in `fixations.json` but absent from the store raises, naming the trial
      (FR2.1) — this is the "phantom trial" guard.
- [ ] `check_features.py` imports with **no bundle present and `evedataset` uninstalled** — the guard
      must run before the bundle is staged. Assert by monkeypatching `evedataset` out of
      `sys.modules` and blocking its import.
- [ ] `main()` returns `0`/`1` and never raises out of `main()`; stdout is parseable JSON on success.

### Group 5 — Task embeddings and the report

- [ ] `copy_task_embeddings()` on a synthetic `{"free-viewing": np.zeros(768, np.float32)}` produces a
      destination that loads to an equal dict and `sha256(src) == sha256(dst)` (FR6.3).
- [ ] Raises when the source dict lacks `"free-viewing"`; when the value is `(384,)`; when the value
      is float64 — three separate cases (FR6.2).
- [ ] Verification reads the **destination**, not the source: a test that truncates the destination
      after copy and before verification must fail.
- [ ] `feature_report.json` is written when `extract_all` performs **zero** work because the cache is
      complete (FR8.3), and its `n_extracted == 0`, `n_skipped_existing == n_trials`.
- [ ] Every FR8 key is present in the report, and every FR8.1 counter is present with value `0` on a
      clean run — assert the key set explicitly, so a counter cannot be dropped silently.
- [ ] `feature_sha256` has one entry per written tensor and each value is 64 hex chars.
- [ ] An interrupted write leaves no loadable partial file: kill `extract_all` between `torch.save`
      to `.tmp` and `os.replace`, then assert `check_features` reports the trial as **missing**, not
      as bad-shape.

### Group 6 — Bundle integration (`bundle`-marked; needs `--bundle-dir`)

- [ ] `EveBundle.load(bundle_dir).samples_df` has the six expected columns and ≥ 3096 rows.
- [ ] The authoritative mapping has **1804 entries, 1804 distinct exp_keys**, split
      **1062 test / 742 train** — cross-checked against `bridge_report.json`'s `num_trials`,
      `num_trials_test`, `num_trials_train`.
- [ ] `crosscheck_exp_keys()` agrees on **all 1804** trials (FR2.4). This is the D4 gate; a single
      disagreement stops F4.
- [ ] `get_stimulus()` on 20 sampled exp_keys returns `(1080, 1920, 3)` uint8 (FR3.2).
- [ ] Every exp_key in the mapping satisfies `exp_key_filename()` (FR7.4) — all 1804, not a sample.
- [ ] `bridge_report.json`'s `origin_size == [1080, 1920]` agrees with the observed stimulus shape.

### Group 7 — Cluster run (`extract_eve_features.sh`)

- [ ] The version probe reports `cuda_available: true` and a non-empty version for each of `torch`,
      `torchvision`, `numpy`, `PIL`, `h5py`, `pandas`, `evedataset`; `versions.txt` exists in
      `$OUT_DIR` (FR1.5, D5).
- [ ] The probe imports `evedataset` **whole**, not `evedataset.bundle` — confirmed by the probe
      failing on an env with a torchvision lacking `transforms.v2` (FR1.3). If no such env is
      available, assert the probe's source imports the package, and record that the negative case was
      not exercised.
- [ ] The preconditions fail fast and legibly on: a missing `bundle.h5`, a missing `stimuli/`, a
      missing `gt_heatmaps.h5`, and insufficient free space — four separate deliberate runs, each
      exiting non-zero before the env is activated where possible (FR9.4).
- [ ] `set +u` / `set -u` around `conda activate` is present and `-e`/`-o pipefail` are never lifted
      (FR9.3); grep the script.
- [ ] A second invocation with the cache complete skips extraction via the exit-code guard, still
      rewrites `feature_report.json`, and completes in well under a minute (FR9.6, FR8.3).
- [ ] `FORCE_FEATURES=1` re-extracts and the resulting `feature_sha256` map is **identical** to the
      first run's — the extractor is deterministic under `no_grad` on a fixed device.
- [ ] The post-extraction `check_features.py` exits `0`, and a deliberately deleted `.pth` makes the
      job fail rather than pass silently.
- [ ] `stdout.txt` exists and contains the headline counts (FR9.7).

---

## Data Validity

Run against `data/eve_features/` after the real run. Each check states its expected outcome; several
are notebook cells rather than pytest.

- [ ] **Coverage is exact, not approximate.** `ls image_features/*.pth | wc -l` equals **1804**
      (`--split both`) or **1062** (`--split test`). Not "about" — any other number means trials were
      skipped and D7 was violated.
- [ ] **Every file is non-trivial.** Min file size ≥ 6.2 MB (a `(768, 2048)` float32 tensor is
      6,291,456 bytes plus pickle overhead). A file materially smaller is a truncated write.
- [ ] **No dead tensors.** For 30 sampled tensors: `t.abs().sum() > 0`, `t.std() > 0`, and
      `torch.isfinite(t).all()`. A zero or non-finite tensor means the backbone ran on a blank or
      corrupt image.
- [ ] **Post-ReLU sanity.** ResNet `layer4` output passes through ReLU, so **`t.min() >= 0`** on every
      sampled tensor. A negative value means the wrong module was tapped.
- [ ] **Sparsity is in the expected band.** The fraction of exact zeros per tensor sits in roughly
      `0.3-0.8` (post-ReLU ResNet features are sparse but not empty). Record the actual
      min/median/max; a median above 0.95 or below 0.05 is worth investigating before F5 consumes the
      cache.
- [ ] **The per-trial keying actually did something — the falsifiable prediction.** Pick 10
      `stimulus_name`s viewed by ≥ 2 participants. For each, the two participants' tensors must be
      **different** (`torch.equal` false) and yet **strongly related**
      (flattened cosine similarity ≥ 0.7, reflecting the same photograph at a different scale).
      *If they were bit-identical, per-trial keying silently collapsed back to per-name and OPEN-6 is
      not resolved.* Record the cosine distribution — it is the quantitative statement F7 makes about
      the display-scale augmentation, and the number OPEN-6 always lacked.
- [ ] **Cross-stimulus contrast.** The same cosine measured between tensors of *different*
      `stimulus_name`s is materially lower than the within-name figure above. If it is not, the
      features are not discriminating images and the comparison above means nothing.
- [ ] **The squash is recorded and correct.** `feature_report.json`'s `squash` reads
      `x = 0.5333…`, `y = 0.7111…`, `uniform: false`, and `resize_input == [768, 1024]` (FR4.6).
- [ ] **`embeddings.npy` is byte-identical to the OSIE source**, and its sha256 matches the value
      `bash/test_osie.sh` scored F1 with. This is what licenses comparing F5's run to F1's baseline on
      this input (FR6.3).
- [ ] **Cross-check against `bridge_report.json`**: `n_trials`, `n_trials_test`, `n_trials_train` in
      `feature_report.json` equal `num_trials`, `num_trials_test`, `num_trials_train` — 1804 / 1062 /
      742. Derived independently on both sides; agreement is evidence, not tautology.
- [ ] **Spot-check the image is the right one.** For 5 random trials, load
      `bundle.get_stimulus(exp_key)`, render it, and confirm the participant's `fixations.json`
      coordinates fall inside the visible stimulus panel — the property verified on three trials
      during specification (4/4, 5/5, 6/6 fixations inside). A fixation on cream background means the
      trial is mapped to the wrong capture.

---

## Data Architecture Integrity

These are the keying invariants. They are checked separately because each one, if violated, produces
numbers that look entirely reasonable.

- [ ] **exp_key round-trip.** For every trial in `fixations.json`:
      `store.trial_key_of(exp_of[(name, subject)])` returns `"{name}|{subject}"`. Uses the store's
      *reverse* index, so a mapping built by the forward path cannot validate itself.
- [ ] **The cross-check is real, not vacuous.** Deliberately corrupt one `samples_df` row's
      `stimulus_name` in a copy and confirm `crosscheck_exp_keys()` **raises** and names that trial.
      A cross-check that cannot fail is not a check.
- [ ] **No phantom keys.** `set(exp_of.values())` equals the set of `.pth` stems in
      `image_features/` (for the extracted split). Neither direction may have extras: a stem with no
      trial is a stale file from an earlier run, a trial with no stem is a coverage hole. Both are
      failures.
- [ ] **Subject identity survives.** For 10 random trials, `subject_id_map.json`'s
      `to_eve[str(subject)]` equals the `subject` field of the `samples_df` row for that trial's
      exp_key. This is D4 checked end to end: dense id → EVE participant → the capture they saw.
- [ ] **Dense ids are exactly `0..37`**, each appearing in at least one trial, matching
      `bridge_report.json`'s `num_subjects: 38`.
- [ ] **The bridge's per-name stimuli are not consumed.** `grep -rn "eve_bridge/stimuli"
      tools/eve_prep/ bash/extract_eve_features.sh` is empty (FR11.4), and
      `feature_report.json`'s `keying == "exp_key"`. This is what prevents a later refactor from
      quietly reintroducing OPEN-6.
- [ ] **`str.replace('jpg','pth')` appears nowhere on the EVE path.**
      `grep -rn "replace('jpg'" tools/eve_prep/` is empty. F5's loader must build paths by
      concatenation (FR5.2, FR10.1).
- [ ] **F2 and F3 artefacts are untouched.** `fixations.json` still hashes to
      `46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b`; `gt_heatmaps.h5`,
      `subject_id_map.json` and `bridge_report.json` have unchanged mtimes and sizes; F3's
      `eve_fewshot_user_embedding_10_seed0.pt` still hashes to `e904a165c985b33f…`. F4 is read-only
      with respect to every earlier stage.
- [ ] **Upstream is unmodified.** `git status --porcelain ISP SE-Net` is empty (D1, convention 2,
      FR14.4).
