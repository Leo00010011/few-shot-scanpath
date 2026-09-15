# F5 — Validation

> Spec folder 3 of 3: [requirements.md](requirements.md) · [plan.md](plan.md) · **validation**

Groups 1–7 are `pytest`, CPU, runnable on the Windows checkout. Group 8 needs the cluster.
The Data Validity and Data Architecture Integrity blocks run **after** the real run, against real
artefacts.

---

## Code Correctness

### Group 1 — Branch provenance and D1

- [ ] `sha256(ISP/EVE/.../utils/evaluation.py) == sha256(ISP/OSIE/.../utils/evaluation.py)`. Any
      difference fails with both hashes — this is the check TechStack §4.2's COCO_FV drift went
      without (FR2.2).
- [ ] Same for `utils/evaltools/scanmatch.py` and `utils/evaltools/visual_attention_metrics.py`.
- [ ] Walking both `src/` trees, the set of relative paths whose bytes differ is **exactly**
      `{"dataset/dataset.py", "test.py"}`. Any extra path fails naming it (FR2.3). `__pycache__` is
      excluded from the walk.
- [ ] The set of relative paths present in one tree and absent from the other is **empty**.
- [ ] `ISP/EVE/.../src/models/gazeformer.py`, `models/models.py`, `models/sampling.py`,
      `models/loss.py`, `utils/logger.py`, `utils/data_postprocess.py` and `preprocess/*` are each
      byte-identical to their OSIE counterparts.
- [ ] `import test` (the EVE one) binds `EVE_evaluation` and **not** `OSIE_evaluation`; the module's
      AST contains no `from dataset.dataset import` name other than `EVE_evaluation` (FR2.4).
- [ ] The AST of `EVE_evaluation.__getitem__` contains **zero** `Call(func=Attribute(attr="replace"))`
      nodes. Fails naming the line number (FR2.5).
- [ ] `EVE_evaluation.__getitem__`'s AST contains no `Image.open` / `io.imread` call, and a run over
      the synthetic fixture with `stimuli_dir` pointed at a **non-existent** directory succeeds
      (FR4.2 — the bridge's `stimuli/` is unused, OPEN-6).

### Group 2 — `EVE_evaluation` construction

Fixture: 6 images × 3 subjects = 18 records, subjects drawn from a 5-subject pool so trios differ
per image; a matching `gt_heatmaps.h5`; 18 `(768, 2048)` float32 tensors, each filled with a distinct
constant so identity is checkable.

- [ ] `len(dataset) == 6` — the **image** count, not 18 (FR4.5).
- [ ] `dataset.resizescale_x == 3.75` and `dataset.resizescale_y == 2.8125`, compared with `==`, not
      `pytest.approx` (FR4.1, FR6.2). Constructing with the OSIE default `(600, 800)` instead gives
      `1.5625 / 1.5625` — asserted explicitly, so the test documents what the mistake looks like.
- [ ] `dataset.downscale_x == 1920/32 == 60.0` and `downscale_y == 1080/24 == 45.0`.
- [ ] `__getitem__(0)["image"].shape == (3, 768, 2048)` and `.dtype == torch.float32`.
- [ ] `__getitem__(0)["trial_key"]` is a `list[str]` of length 3, each `"{name}|{dense subject}"`,
      in the same order as `["subject"]` (FR4.6).
- [ ] `__getitem__(0)["length"]` is `(3,) int32` and each entry equals
      `min(record["length"], 16)`.
- [ ] `fix_vectors[0].dtype.names == ("start_x", "start_y", "duration")` with all three formats
      `f8`; `start_x == X/3.75`, `start_y == Y/2.8125`, `duration == T/1000.0`, each to within
      `1e-12` (FR4.7).
- [ ] A record whose `X` reaches 1920 maps to `start_x == 512.0` exactly — the top of the metric
      screen, not past it.
- [ ] `collate_func` on a 2-image batch returns `images (2, 3, 768, 2048)`,
      `trial_keys` a `list` of 2 `list`s of 3 `str`, `lengths` a `(2, 3)` int32 tensor, and
      `subjects` a `(2, 3)` tensor whose flattening matches the per-image `subject` lists in order.
- [ ] `images.view(-1, *images.shape[2:]).shape == (6, 768, 2048)` and row `k` equals the tensor for
      `flat_trial_keys[k]` — the flattening order the heatmap block and the prediction writer both
      depend on (FR5.4, FR8.2).

### Group 3 — Per-trial feature loading (OPEN-6 at Stage D)

- [ ] Two subjects on the same image receive **different** tensors:
      `not torch.equal(item["image"][0], item["image"][1])`. Under upstream's per-image load they
      would be identical — assert that too, against a deliberately per-image-keyed fixture, so the
      test states the difference rather than only the fix (FR5.1).
- [ ] The path is built by concatenation: monkeypatch `torch.load` to capture its argument and assert
      it equals `join(feat_dir, exp_key + ".pth")` for every one of the 18 trials.
- [ ] An `exp_key` containing `"jpg"` raises `EvePrepError` from `exp_key_filename()` before any load
      (FR5.1, TechStack §3.2).
- [ ] A missing `(name, subject)` in `exp_key_map` raises `KeyError` naming **both** the name and the
      subject — never a fallback, never a sibling subject's tensor (FR5.3).
- [ ] A tensor of shape `(768, 1024)` raises `ValueError` naming the `exp_key`, `(768, 2048)` and
      `(768, 1024)` (FR5.2).
- [ ] A `torch.float64` tensor of the right shape raises (FR5.2).

### Group 4 — D4: the cohort, the remap, and the subject ids

- [ ] `--fewshot_subject` **descending** (`[4,3,2,1,0]` on the fixture) raises in
      `EVE_evaluation.__init__` with a message containing "ascending" and naming the first record
      whose subject changed (FR4.4). **This is the trap's test**: without the assertion the run
      completes and every metric looks plausible.
- [ ] A **rotated** list (`[1,2,3,4,0]`) raises for the same reason — not only reversal.
- [ ] Ascending `[0,1,2,3,4]` leaves every record's `subject` byte-equal to the file's, and the
      constructor does not raise.
- [ ] A list shorter than the cohort (`[0,1,2]`) drops records and therefore makes images ragged;
      the preflight catches it first (Group 6), and the constructor's assertion still passes — the
      test asserts *which* layer fires, so the two are not confused.
- [ ] Simulated prediction writing over the fixture: for every `(name, slot)`, the written `subject`
      equals `fixations`'s dense subject at that positional slot, and `subject_eve` equals
      `to_eve[str(subject)]` (FR7.2).
- [ ] The **upstream** formula `args.fewshot_subject[subject_idx]` is computed alongside and asserted
      **wrong** on the fixture — it yields `0,1,2` on every image while the truth varies. The test
      documents the defect FR7.1 exists for.
- [ ] A dense id absent from `to_eve` raises `KeyError` naming the id (FR7.5).
- [ ] `recover_subject_ids()` is never called on the EVE path (AST check on `test.py`) (FR7.5).

### Group 5 — Heatmap block wiring

- [ ] `score_step_heatmaps` is imported from `tools/eve_bridge/heatmap_metrics.py` and neither NSS,
      CC nor KLD is redefined anywhere under `ISP/EVE/` or `tools/eve_eval/` (grep + AST) (FR8.3).
- [ ] `store.get_batch(flat_keys)` returns `(N, 16, 24, 32)` float32 with `N == 3 * batch_size`, and
      row `k` is the store row for `flat_keys[k]` — checked by writing a distinct constant into each
      fixture row (FR8.2).
- [ ] Passing `flat_keys` in a **permuted** order changes the returned NSS/CC/KLD; the test asserts
      the two differ, so a silent misalignment cannot pass unnoticed.
- [ ] `N != len(flat_keys)` raises `ValueError` from the call site naming both counts (FR8.2).
- [ ] Argument order: `score_step_heatmaps(pred, gt, …)` and `score_step_heatmaps(gt, pred, …)`
      produce **different** KLD on a fixture where the two differ — the asymmetry check that pins
      prediction-first (FR8.3).
- [ ] The weighted accumulation over two batches with unequal `M` equals the single-batch result over
      their concatenation, to `1e-6` (FR8.4). A plain mean-of-batch-means differs and is asserted to
      differ.
- [ ] `--heatmap_dir ""` produces `metrics["heatmap"] is None` with a non-empty `reason` string
      (FR8.7).
- [ ] `GtHeatmapStore.load` with a `fixations.json` whose bytes were changed raises before any
      heatmap is read (FR8.1).

### Group 6 — Preflight (`tools/eve_eval/check_eval.py`)

Each case builds a deliberately broken artefact set and asserts the specific failure.

- [ ] A clean fixture returns `ok: true`, exits `0`, and writes `preflight.json` (FR11.2, FR11.3).
- [ ] Any missing artefact from FR3.1 exits `1`, names the path, and the message mentions that
      `data/` is git-ignored.
- [ ] `fixations.json` byte-changed → all three sha handshakes fail, and **all three** are printed,
      not just the first (FR3.2, FR11.2).
- [ ] `fixations.json` **reordered** but otherwise identical → still fails. An order-insensitive check
      would let a mis-indexed positional cache through (working convention 10).
- [ ] The embedding tensor byte-changed → FR3.3 fails naming both hashes.
- [ ] `senet_report["seed"] = 1` with a `_seed0.pt` filename → fails (FR3.3).
- [ ] `min_support_per_subject = 9` with `num_fewshot = 10` → fails, and the message says
      **minimum**, not `support_pool_size` (FR3.5).
- [ ] `bridge_report["origin_size"] = [600, 800]` → fails (FR3.6). Likewise an h5 `action_map` of
      `[30, 40]` or `max_length` of 20.
- [ ] `embeddings.npy` missing the `"free-viewing"` key → fails; a byte-different copy of the OSIE
      file → fails (FR3.7).
- [ ] One image given 2 subjects → fails naming that image and `2` (FR3.8). This is the invariant
      that stops `-1` folding into every metric.
- [ ] One image given 4 subjects → fails the same way.
- [ ] Subject ids `{0,1,2,4}` (a gap) → fails on "dense 0..N-1" (FR3.8).
- [ ] Every `T` replaced by a decile bin `0..9` → fails on `max(T) <= 20` with the TechStack §3.7
      message (FR3.8). Every other invariant passes on that fixture — asserted, because that is
      exactly why the check exists.
- [ ] `condition = "search"` on one record → fails naming it.
- [ ] One `.pth` deleted → fails listing it and the total; at 15 missing, lists 10 and says
      `(+5 more)` (FR3.9).
- [ ] One `.pth` byte-changed → FR3.4 fails naming the `exp_key`.
- [ ] `--fast` skips only FR3.4; the missing-file case still fails, and `preflight.json` carries
      `"feature_sha256_checked": false` (FR11.4).
- [ ] A failing run still writes `preflight.json`, with `"ok": false` and the full failure list
      (Step 3).
- [ ] `check_eval` imports no `torch` and no `evedataset` (module-import assertion) (FR1.3, FR11.1).

### Group 7 — `test.py` arguments and the batch cap

- [ ] `--eval_repeat_num 2` is rejected at parse time with a message naming `IndexError` and
      `subject_num` (FR6.5).
- [ ] Defaults: `origin_width 1920`, `origin_height 1080`, `im_h 24`, `im_w 32`, `subject_num 3`,
      `num_fewshot 10`, `batch 1`, `max_batches -1`, `width 512`, `height 384`, `max_length 16`
      (FR6.1, FR6.3).
- [ ] With `--max_batches -1` over a 6-image fixture, all 6 batches run and
      `metrics["counts"]["n_batches_run"] == 6`.
- [ ] With `--max_batches 2`, exactly 2 run, and the FR6.4 truncation assertion does **not** fire
      (it is gated on `max_batches <= 0`).
- [ ] Simulating a loader that yields fewer batches than `len(test_loader)` with `--max_batches -1`
      raises `RuntimeError` naming both counts (FR6.4).
- [ ] `metrics.json` is written even when `predict_results` is empty (FR9.1).
- [ ] `metrics.json` carries `per_cell_std: null` **and** a non-empty `reason` mentioning F6 — the
      field is present, never omitted (FR9.2).
- [ ] `metrics["metrics"]["VAME"]` carries `SED`, `STDE`, `SED_best`, `STDE_best`, and
      `metrics["notes"]` contains a string stating the `_best` pair are **aliases** (FR9.3).
- [ ] `metrics["notes"]` contains the FR13.1–FR13.6 strings: R@3 saturation at K=3, the per-image
      varying trio, the three squashes, the bridge counters, the OSIE-trained checkpoint + OPEN-7,
      and the MultiMatch NaN-drop count.
- [ ] `metrics["headline"]["SM"]` equals `scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))`
      to `1e-12`, and `MM` the mean of the five MultiMatch dims — the same numbers the bare `print()`
      emits (FR6.7, FR9.2).
- [ ] `prediction.json` with one record removed → the FR7.4 key-set check raises listing it.
- [ ] `prediction.json` with a duplicated `(name, subject)` → raises.

### Group 8 — Run script (cluster, `bash -n` locally)

- [ ] `bash -n bash/test_eve.sh` parses on the dev machine.
- [ ] `shellcheck` reports no error-level finding.
- [ ] Sourcing the tunables block with every variable unset yields defaults under
      `/mnt/beegfs/home/`; no `/mnt/imagenes` and no bare `/tmp` write path appears (TechStack §1.3,
      convention 4).
- [ ] `grep -n 'conda info --base' bash/test_eve.sh` is empty; `conda.sh` is sourced by explicit path
      (FR12.3).
- [ ] `set +u` appears immediately before `source conda.sh` and `set -u` immediately after
      `conda activate`; `set -e` and `set -o pipefail` are never lifted (FR12.3).
- [ ] The `FEWSHOT_SUBJECTS` assertion rejects a hand-set descending `N_SUBJECTS` list (run the
      embedded python block directly with a permuted list) (FR12.4).
- [ ] A missing `data/eve_bridge/fixations.json` makes the script exit non-zero **before** any GPU
      work, printing the git-ignore note (FR12.5).
- [ ] The preflight exiting `1` aborts the script (FR12.5).
- [ ] Running seed 0 twice leaves `seed0/` with six files and no partial; running seed 1 does not
      overwrite `seed0/` (FR10.2).
- [ ] Copying `metrics.json` from seed 0 into `seed1/` makes the FR10.3 assertion fail with a message
      naming both seeds.

---

## Data Validity

Run after the real seed-0 run, against real artefacts. Each check states its expected outcome; a
notebook cell is acceptable where pytest is not.

- [ ] **Coverage is exact.** `prediction.json` holds **1062** records over **354** names and **38**
      subjects, and its `(name, subject)` key set equals the `test` split's exactly. Expected: zero
      difference in both directions.
- [ ] **Every image contributed 3 cells.** `Counter(name).values()` is `{3: 354}`. Any other value
      means the run truncated or the bridge changed.
- [ ] **No truncation.** `metrics["counts"]["n_batches_run"] == 354` and
      `metrics["args"]["max_batches"] == -1`. The `i_batch > 100` cap is the single most dangerous
      item in F5; this is the number that proves it did not fire.
- [ ] **Subject identities really do vary per image.** The number of distinct subject trios across
      the 354 images is **> 1** — expected to be well above 100 given a 38-participant cohort at 3
      per image. If it were 1, OPEN-5's premise would have been right after all and the cohort would
      need re-deriving.
- [ ] **The predicted coordinates live in the metric screen.** `X ∈ [0, 512]`, `Y ∈ [0, 384]` over all
      1062 records. Values beyond either bound mean `origin_size` did not take effect.
- [ ] **Ground-truth coordinates land where the squash says.** Over the scored split,
      `max(X)/3.75 <= 512` and `max(Y)/2.8125 <= 384`, and the realised maxima are close to those
      bounds (EVE is full-screen, so expect near-saturation on both axes).
- [ ] **Durations are seconds inside the vectors and milliseconds in the artefacts.** A sampled
      `fix_vector`'s `duration` is in roughly `[0.02, 2.0]`; `prediction.json`'s `T` is integer and in
      roughly `[20, 2000]`. A `T` in `[0, 9]` means the decile-bin file reached the run despite
      FR3.8.
- [ ] **Scanpath lengths are plausible.** Predicted lengths in `[1, 16]`, mean expected around the
      ground-truth mean over the scored split (report both; a predicted mean at the 16 ceiling for
      most cells indicates the termination action is never being sampled and is worth investigating,
      not a pass/fail).
- [ ] **The headline numbers exist and are finite.** `SM`, `MM`, `SED`, `STDE` all finite; no metric
      is `-1`. An all-`-1` block means the loop never ran (TechStack §4). Compare against F1's OSIE
      row (SM 0.3704 ± 0.0041, MM 0.8010 ± 0.0019, SED 7.3438 ± 0.0381) — **as context, not as a
      target**: a different dataset, cohort and stimulus aspect ratio. A large gap is a finding for
      F7, not a failure here.
- [ ] **The MultiMatch NaN-drop count is reported and small.** Derived from `score_details`. Expected
      well under 5 % of 1062. A large count changes the denominator silently and must reach F7 (D7,
      FR13.6).
- [ ] **Retrieval is saturated exactly where predicted.** `R@3 == 100.0` and `R@5 == 100.0` at
      `subject_num = 3` — structural, not a result. `R@1` and `MRR` are the informative pair;
      `R@1` at chance would be ≈ 33.3 %, so report the realised value against that.
- [ ] **The heatmap block's denominator is the timestep count.** `metrics["heatmap"]["M"]` equals
      `Σ min(length, 16)` over all 1062 scored trials, recomputed independently from
      `fixations.json`. Expected well above 1062 — they are different denominators, which is the whole
      point of reporting them separately.
- [ ] **NSS is interpreted with its caveat.** Record the action map's std alongside NSS. If the map
      is near-flat, NSS is O(1) noise and must not be read as chance level (FR8.6). CC near 0 is the
      honest floor.
- [ ] **Three seeds agree to within a band.** Across seeds 0/1/2, report mean, sample std (ddof=1)
      **and range** for `SM`, `MM`, `SED`, `STDE`, `R@1`, `MRR`. At n = 3 the std is a noisy estimate;
      the range rides alongside it. Expected: a band comparable in width to F1's
      (`SM ± 0.004`). A seed-to-seed spread an order of magnitude larger indicates something other
      than sampling noise.
- [ ] **The three `versions.txt` agree.** Pooling seeds run on different stacks mixes sampling noise
      with a version change (TechStack §3.5b). Expected: byte-identical across the three seeds.
- [ ] **The three `preflight.json` fingerprints agree** on split, image/subject/cell counts and all
      three sha256 handshakes.
- [ ] **`multimatch_gaze` is 0.1.3** in every seed's `versions.txt`. The one pin that was held in F1.

---

## Data Architecture Integrity

The keying invariants. These are the checks that catch a plausible, wrong number.

- [ ] **`exp_key` roundtrip.** For all 1062 scored trials,
      `store.trial_key_of(store.exp_key_of(name, subject)) == f"{name}|{subject}"`. Neither direction
      has a fallback that invents a key; both must be total over the cohort.
- [ ] **No phantom keys.** `set(exp_key_map.values())` over the scored split equals the set of
      `.pth` stems under `image_features/` for that split, exactly — no extra file, no missing one.
      Expected 1062 on both sides.
- [ ] **No shared tensor between trials.** All 1062 `feature_sha256` values from
      `feature_report.json` are **distinct**. A repeat means two participants were handed one screen
      capture — OPEN-6 reappearing (F4 measured no bit-identical pair; assert it still holds).
- [ ] **The per-trial keying is real at Stage D, not just at Stage B.** Sample 20 multi-viewer
      stimuli from the scored split, load the actually-loaded tensors, and compute within-name cosine.
      Expected median ≈ **0.775** against a cross-stimulus median ≈ **0.450** (F4's measured values),
      with no pair bit-identical. If within-name similarity is 1.0, the per-image load survived
      somewhere and every subject is seeing the first participant's screen.
- [ ] **The order verification is not bypassable.** Corrupt one byte of `fixations.json` and re-run:
      `GtHeatmapStore.load`, `load_trial_exp_keys` and all three preflight handshakes must each fail
      independently. Then **reorder** the records without changing any value and confirm every one
      still fails — an order-insensitive check would let a mis-indexed positional cache through
      (working convention 10).
- [ ] **The heatmap store is positional and stays aligned.** For 20 random scored trials, the store
      row fetched by `trial_key` has `length` equal to `min(fixations[i]["length"], 16)` for the
      record `i` at that trial's position in `fixations.json`. A positional drift of one would still
      produce finite NSS/CC/KLD.
- [ ] **`prediction.json` answers D4's question.** For a randomly chosen record, `subject_eve` equals
      `subject_id_map["to_eve"][str(subject)]`, and that participant's real EVE id appears in
      `bridge_report["args"]["unseen_subjects"]`. "Which of my real subjects is row 3?" must have one
      answer, derivable from the artefact alone.
- [ ] **The identity remap held.** `metrics["args"]["fewshot_subject"] == list(range(38))`, strictly
      ascending. Asserted from the written artefact, not from the script — the script generated it,
      the log is the evidence it arrived intact.
- [ ] **The embedding table is the one F3 built.** `metrics["sha256"]["user_embedding"]` starts
      `e904a165` and equals `senet_report["embedding_sha256"]`; the loaded tensor is `(38, 384)`
      float32 with no all-zero row and no duplicated row.
- [ ] **The frozen files are still frozen at run time.** `metrics.json` records the sha256 of the
      EVE branch's `evaluation.py` and both `evaltools/*`, and they match the OSIE originals. The
      Group 1 test proves it in the repo; this proves it on the machine that produced the numbers
      (D1, D5).
