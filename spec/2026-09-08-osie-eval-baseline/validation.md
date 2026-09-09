# F1 — Validation

> Companion to [requirements.md](requirements.md) and [plan.md](plan.md).
> Constitution: [Mission](../constitution/Mission.md) · [TechStack](../constitution/TechStack.md) ·
> [Roadmap](../constitution/Roadmap.md).

F1 ships no library code, so there is no pytest suite to build. The checks below are **gates on a run**:
each is an assertion executed against the environment or the written artefacts. Groups 1–3 must pass
before GPU time is spent; Groups 4–6 gate the reported numbers; Group 7 is the honesty check on what F1
was allowed to touch.

Where a check can run on the Windows dev machine it is marked **[CPU]** and uses `py`. Everything else
is **[GPU]** and runs on the cluster from `ISP/OSIE/GazeformerISP/`.

---

## Code Correctness

### Group 1 — Environment and pins *(gates everything; FR1)*

- [ ] **[GPU]** `conda activate isp` succeeds and `python --version` reports **3.8.x**. A different minor
      version means `scikit-learn==0.22.2` resolved differently and the env is not the one the pins
      describe.
- [ ] **[GPU]** `importlib.metadata.version("multimatch-gaze") == "0.1.3"` **exactly**. Any other value is
      a **hard stop**, not a recorded deviation: MultiMatch would be a different metric and the MM column
      would not be comparable to the published table (D1's whole purpose). Expected on failure:
      `AssertionError: HARD STOP: multimatch-gaze pinned 0.1.3, got <x>`.
- [ ] **[GPU]** `numpy.__version__ == "1.23.5"` and `scipy.__version__ == "1.10.0"`. A deviation is
      permitted only with a written note in `environment.md` — structured-array dtype behaviour feeds
      every metric input and `scipy.stats.hmean` produces the `SM` headline.
- [ ] **[GPU]** `torch.cuda.is_available() is True` and `torch.cuda.get_device_name(0)` is recorded. A
      `False` here means `test.py`'s `torch.device('cuda')` will raise at model construction.
- [ ] **[GPU]** Every package in the TechStack §1 pin table appears in `environment.md` marked either
      `match` or `deviation: pinned X, got Y`. No package in the table may be absent from the record.

### Group 2 — Asset resolution *(FR2, FR3)*

- [ ] **[GPU]** `ls weights/OSIE-*/OSIE/` output is transcribed verbatim into `environment.md`, including
      the exact case of the directory and the exact spelling of `user` vs `subject` in the embedding
      filename. Expected: `checkpoint_best.pth`, `ckp_11999.pt`, `fewshot_user_embedding_10.pt`,
      `train_user_embedding.pt` — but the **observed** listing, not this expectation, is what gets
      recorded (TechStack §3.4 documents the docs disagreeing).
- [ ] **[GPU]** `os.path.exists("src/assets/OSIE-ex-10to15/checkpoints/checkpoint_best.pth")` is `True`
      **from the `ISP/OSIE/GazeformerISP/` CWD**. Checking from any other directory proves nothing —
      `test.py` joins `args.evaluation_dir` with `"checkpoints"` relative to the CWD.
- [ ] **[GPU]** `os.path.exists("../../../SE-Net/assets/OSIE-ex-10to15/fewshot_user_embedding_10.pt")` is
      `True` from that same CWD. On failure `test.py` raises `FileNotFoundError` inside `torch.load`
      *after* the dataset has been built.
- [ ] **[GPU]** The loaded user embedding has `shape[-1] == 384` (`args.subject_feature_dim`) and
      `shape[0] >= 5` (`args.subject_num`). Record the full shape and dtype. A trailing dim other than
      384 would broadcast silently inside `gazeformer` rather than raise. Expected on failure:
      `AssertionError: subject_feature_dim mismatch: got (N, D)`.
- [ ] **[GPU]** If the loaded object is a `dict` rather than a tensor, its key structure is recorded in
      `environment.md` — F3 needs the container format to decide OPEN-2.

### Group 3 — Stage B features *(gates the first run; FR4, FR5, FR6, FR7)*

- [ ] **[GPU]** `<STAGE_B_ROOT>/train/` exists and contains a `.jpg` for **every** distinct `name` in
      `fixations.json` (all splits). Expected on failure: `AssertionError` naming up to 5 missing
      stimuli. The `train/` subdirectory name is hardcoded in `image_data()` and must not be changed.
- [ ] **[GPU]** `src/data/image_features/` exists **before** `image_data()` is called. Without it every
      `torch.save` raises `FileNotFoundError` and the function reports no error of its own.
- [ ] **[GPU]** `image_data()` was invoked with `overwrite=True`. With `overwrite=False` the function
      skips existing files, so a partial earlier run completes without error and leaves stale or missing
      tensors behind — a silent-degradation failure of exactly the kind D7 forbids.
- [ ] **[GPU]** `python preprocess/feature_extractor.py` was **not** used. Its `__main__` calls
      `text_data(dataset_path=args.p, ...)` and `args.p` is undefined, raising `AttributeError`
      (Finding D); it would also regenerate the shipped `embeddings.npy`.
- [ ] **[GPU/CPU]** For every distinct `name`: `src/data/image_features/<name.replace('jpg','pth')>`
      exists, loads, has shape exactly `(768, 2048)`, dtype `torch.float32`, and
      `torch.isfinite(t).all()`. `768 == 24*32` and `2048` is the ResNet-50 channel count; any other
      first dimension means the `(384*2, 512*2)` resize or the backbone differs. Expected on failure: an
      `AssertionError` listing the offending filenames and what was wrong with each.
- [ ] **[CPU]** `src/data/embeddings.npy` is byte-identical to the shipped file (unchanged mtime/size, or
      a hash comparison against git). `text_data()` must not have run (FR6).

### Group 4 — The `test.py` edit is print-only *(FR8, FR9)*

- [ ] **[CPU]** `git diff ISP/OSIE/GazeformerISP/src/test.py` touches **only** the metric-logging region
      after `comprehensive_evaluation_by_subject()` returns. No hunk may touch `parser.add_argument`, the
      dataset construction, the model, the sampling loop, the evaluator call, or the `prediction.json`
      write.
- [ ] **[CPU]** The diff contains no change to the `SM`, `MM`, or `SED` right-hand-side expressions.
      `SM = scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))` and
      `MM = np.mean(list(cur_metrics["MultiMatch"].values()))` must be character-identical to upstream, so
      an edited and an unedited run produce the same headline values.
- [ ] **[CPU]** Std lookup uses `cur_metrics_std.get(group, {}).get(name)`, never `cur_metrics_std[group]`.
      `retrieval scanmatch w/ duration` exists in `cur_metrics` only (Finding G); a hard index raises
      `KeyError` at the very last line, after the full evaluation has already run.
- [ ] **[CPU]** The `if i_batch > 100: break` line is **unchanged** (FR9). `git diff` must show no hunk
      containing it.
- [ ] **[GPU]** The produced log contains, for every `MultiMatch`, `ScanMatch` and `VAME` entry, a
      `<mean> +/- <std>` pair; and for the four `retrieval scanmatch w/ duration` entries, a mean with no
      std. Total: 5 + 2 + 4 = 11 mean±std lines and 5 retrieval lines.
- [ ] **[GPU]** The log contains exactly one `HEADLINE seed=<n> SM=... MM=... SED=... STDE=...` line, and
      `<n>` matches the `seed` value in the argument-namespace dump at the top of the same file (D5).

### Group 5 — The run and its artefacts *(FR10, FR11, FR12, FR13, FR14)*

- [ ] **[GPU]** `bash/test.sh` runs to completion from `ISP/OSIE/GazeformerISP/` and from nowhere else.
      Running it from the repo root must fail on path resolution rather than silently pick up a wrong
      file.
- [ ] **[GPU]** The inference progress bar totals **70** steps (`len(test_loader) * repeat_num`, 70 test
      images × `repeat_num=1`) and the evaluator's bar totals **70**. A total of 101 or 1050 means the
      batch size or the split filter is wrong.
- [ ] **[GPU]** `bash/test.sh` contains no absolute path (Working convention 4) and no `--img_dir`
      (Finding B: `OSIE_evaluation` never opens a stimulus).
- [ ] **[GPU]** Five logs exist — `log_test_subject_10_{0,1,2,3,4}.txt` — none overwritten. The filename
      carries `--random_support`, which the script varies with the seed precisely because Finding E
      established it is inert on the test path.
- [ ] **[GPU]** Five prediction files exist as `prediction_seed{0,1,2,3,4}.json`. The unsuffixed
      `prediction.json` is overwritten by each run and is **not** the artefact of record.
- [ ] **[CPU]** For the seed-0 prediction: key set is exactly `{"name","subject","X","Y","T"}`;
      `len(records) == 350`; `len({(name, subject)}) == 350`; `len({name}) == 70`;
      per record `len(X) == len(Y) == len(T)` and `1 <= len(X) <= 16`; every `X`/`Y`/`T` value is `int`.
      Expected on failure: `AssertionError` naming the violated clause.
- [ ] **[CPU]** `{name}` in the prediction equals the set of `name`s on the `test` split of
      `fixations.json`, and equals `{name}` in the shipped
      `result/git-osie-useremb-ex-10to15/log/prediction.json`.
- [ ] **[CPU]** A `len(records)` that is a multiple of 350 greater than 350 means `--eval_repeat_num > 1`
      leaked in: `predict_results` appends inside the repeat loop, so duplicates per `(name, subject)`
      would break F6's ability to index by trial (FR11).

---

## Data Validity

These are sanity checks on the numbers themselves. They are judgement calls recorded in `results.md`, not
pass/fail assertions — except where stated.

### Group 6 — Are the numbers real?

- [ ] **[GPU]** **No diagonal mean equals `-1.0000`.** *(Hard fail, FR16.)* `-1` is the evaluator's
      uninitialised sentinel, not a bad score (TechStack §4). An all-`-1` block means the double loop
      never executed and the run must be discarded, not reported.
- [ ] **[CPU]** **NaN-dropped MultiMatch rows are counted and reported.** *(Hard requirement, D7/FR15.)*
      The denominator is `70 × 5 = 350` diagonal rows; `is_eliminating_nan=True` removes NaN rows before
      the mean. Re-derive `dropped = 350 - kept` from the returned `score_details` (MultiMatch slots are
      `NaN` where the metric failed, `-1` where the cell never ran). Report it beside every MultiMatch
      mean. Expected: a small number, plausibly 0. A large drop (>5 %) makes the MultiMatch mean a
      different statistic from the paper's and must be flagged, not averaged over.
- [ ] **[CPU]** **Short-scanpath padding is counted.** `fixations.json` holds 11 records with
      `length < 3` across all splits. Count how many land in the test split's query cohort (subjects
      10–14). Each is padded to length 3 with `(1., 1., 1e-3)` inside the frozen evaluator, and the padded
      array then replaces the original for **all** subsequent metrics in that cell (TechStack §4).
      Expected: 0–2. Record the exact number; it is the OSIE counterpart of `bridge_report.json`'s
      `short_scanpath` that F7 will report for our data.
- [ ] **[CPU]** **Value ranges are plausible.** ScanMatch scores lie in `[0, 1]`; MultiMatch dimensions
      lie in `[0, 1]`; STDE lies in `[0, 1]`; SED is a non-negative edit distance. A ScanMatch mean of
      exactly `1.0` would indicate the prediction and ground truth quantised to identical sequences — the
      `score != score` NaN-coercion branch firing across the board — and is a bug signal, not a
      triumph.
- [ ] **[CPU]** **`pr5 == 100.0` exactly.** *(Expected, not a failure.)* `p2g` scores `r5` as the fraction
      of ranks `< 5`, and with `subject_num=5` every rank is in `{0..4}`. The value is structurally
      saturated and carries no information (Finding F); it must be annotated as such wherever it appears,
      including in F7. If `pr5 != 100.0`, something is wrong with the matrix, not with the model.
- [ ] **[CPU]** **`pr1 > 20.0`.** Chance-level R@1 with 5 subjects is 20 %. A value at or below chance
      means the personalization signal is absent — either the user embedding is not reaching the model or
      the subject axis is misaligned. This is the cheapest end-to-end check that D4 held.
- [ ] **[CPU]** **`rsum == pr1 + pr3 + pr5`.** Internal consistency of the retrieval block as the
      evaluator constructs it.
- [ ] **[GPU]** **Seed spread is reported.** Mean and min–max across seeds 0–4 for `SM`, `MM`, `SED`,
      `STDE`, `pmrr`, collected from the `HEADLINE` lines. This spread is the *definition* of "within
      sampling noise" (FR12) — the comparison to the paper may not be called a match or a mismatch
      without it.
- [ ] **[GPU]** **Comparison to the paper's OSIE row is judged against that spread.** If the paper's
      values fall inside it, F1 is done. If not, F1 is **not** done: write the discrepancy up and escalate
      as a new roadmap open decision (FR18). No adjustment of metric parameters, resize, or evaluator is
      permitted to close the gap (D8) — tuning it into agreement destroys the diagnostic value F1 exists
      to provide.
- [ ] **[CPU]** **Cross-check against the shipped prediction.** Score our seed-0 `prediction.json` and the
      shipped `git-osie-useremb-ex-10to15/prediction.json` through the same evaluator. The two are
      independent stochastic draws from the same model, so the numbers will differ — but they should
      differ by no more than the seed spread. A large gap points at the environment or the checkpoint,
      not at sampling. *(This doubles as the seed test for F6.)*

---

## Data Architecture Integrity

### Group 7 — Keying invariants and what F1 was allowed to touch

- [ ] **[CPU]** **Subject-id roundtrip (D4).** `sorted({r["subject"] for r in prediction})` is
      `[10, 11, 12, 13, 14]` — the **original** OSIE ids. `select_fewshot_subject()` remaps 10–14 to a
      dense `0..4` for the model; `get_prediction_list()` must invert it via
      `args.fewshot_subject[subject_idx]` before writing. A dense `0..4` in the output means the inverse
      map was skipped and every diagonal cell scored the wrong person against the wrong person, while
      still producing plausible-looking numbers. *(Hard fail.)*
- [ ] **[CPU]** **The mapping is order-derived, not value-derived.** `get_prediction_list()` indexes
      `args.fewshot_subject` **positionally** by `subject_idx`, and `select_fewshot_subject()` builds its
      dense map by `enumerate(fewshot_subjects)` — the argument order, not sorted order. Passing
      `--fewshot_subject 14 13 12 11 10` would therefore relabel every row consistently but differently.
      Confirm the run used ascending `10 11 12 13 14` and that `bash/test.sh` pins that order literally.
- [ ] **[CPU]** **No phantom subjects.** Every `(name, subject)` in the prediction has a matching record on
      the `test` split of `fixations.json`. A prediction for a `(name, subject)` pair with no ground truth
      means the loader's grouping and the writer's indexing disagree.
- [ ] **[CPU]** **Equal-subject invariant holds and is not bypassable (FR15).** On the test split every
      image carries exactly 15 subjects before filtering and exactly 5 after restricting to 10–14. This
      is what keeps `evaluation.py`'s `(subject × subject)` matrix square; a ragged input does **not**
      raise, it silently produces a matrix whose off-diagonal feeds the retrieval block with garbage. The
      check must be run against the data, not inferred from the fact that the run completed.
- [ ] **[CPU]** **Row/column semantics are as documented.** In the evaluator, row = predicted subject,
      column = ground-truth subject, diagonal = the reported score (D4, `evaluation.py` lines 14–15).
      Confirm `args.subject_num == 5` in the logged namespace, since it sizes both axes; a value of 15
      would allocate a 15×15 matrix over 5 subjects of data and fill 200 of 225 cells with the `-1`
      sentinel, which then enters the SED/STDE means (those do **not** get NaN elimination).
- [ ] **[CPU]** **Frozen files are untouched (D1).** `git diff --stat` shows **no** change to
      `ISP/OSIE/GazeformerISP/src/utils/evaluation.py`,
      `ISP/OSIE/GazeformerISP/src/utils/evaltools/scanmatch.py`, or
      `ISP/OSIE/GazeformerISP/src/utils/evaltools/visual_attention_metrics.py`. *(Hard fail.)*
- [ ] **[CPU]** **No unplanned files changed.** The full `git status` for F1 is: modified
      `ISP/OSIE/GazeformerISP/src/test.py` (print-only), new
      `ISP/OSIE/GazeformerISP/bash/test.sh`, and new files under
      `spec/2026-09-08-osie-eval-baseline/`. Anything else — in particular `dataset/dataset.py`,
      `models/*`, or `preprocess/feature_extractor.py` — is out of scope and must be reverted.
- [ ] **[CPU]** **No weights, features, stimuli, or gaze data are staged for commit** (Working convention
      5). `.gitignore` covers `*.pt`, `*.pth`, `*.h5`; confirm no `.jpg` stimulus or
      `result/**/prediction*.json` is added. The prediction files are run artefacts, not repo content —
      their *numbers* go into `results.md`.
- [ ] **[CPU]** **No new `.pyc` files were left in upstream directories** (Working convention 6). The 89
      tracked `__pycache__` directories stay as they are; any *new* `.pyc` dropped by F1's runs is
      deleted before commit.
- [ ] **[GPU]** **Determinism controls are on.** The logged namespace shows the run's `seed`, and
      `test.py`'s module-level block sets `np.random.seed`, `torch.manual_seed`,
      `torch.cuda.manual_seed_all`, `cudnn.benchmark = False`, `cudnn.deterministic = True` (D5).
      Confirm these five lines are present and unmodified in the diff.
