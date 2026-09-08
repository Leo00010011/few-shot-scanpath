# Validation — COCO-FreeView baseline on the cluster

> Companion to [requirements.md](requirements.md) and [plan.md](plan.md).
> Groups 1–3 run on the Windows dev machine (`py -m pytest tests/cocofv_prep -q`) against fixtures.
> Groups 4–8 run on the cluster and are recorded in `notes.md`.
> The feature is **not** done until every box is ticked or explicitly struck with a written reason.

---

## Code Correctness

### Group 1 — `normalize_fixations` (FR3.3)

- [ ] `normalize([{"split": "val", "task": "bottle", ...}])` returns `split == "validation"`; the
      input dict is **not** mutated (assert the original still has `"val"`).
- [ ] `task == "potted plant"` → `"potted_plant"` and `task == "stop sign"` → `"stop_sign"`; a task
      already underscored (`"potted_plant"`) passes through unchanged.
- [ ] A task not in the fix table (`"bottle"`) is returned verbatim — no `KeyError`, no `None`.
      Regression guard: `_TASK_FIXES.get(r["task"], r["task"])`, not `_TASK_FIXES[r["task"]]`.
- [ ] `split == "train"` and `split == "test"` pass through untouched. Only `"val"` is rewritten.
- [ ] Every key present on the input record is present on the output record with the same value,
      for a record carrying an extra unknown key (`"extra": 1`) — FR3.4 permits extras.
- [ ] Running `normalize` twice is idempotent: `normalize(normalize(x)) == normalize(x)`.
- [ ] The CLI prints all-zero transformation counters on an already-normalized file, and exits 0.

### Group 2 — `check_fixations` hard invariants (FR3.5)

Each case builds a minimal in-memory fixture, writes it to a tmp path, and asserts
`CocoFvPreflightError` is raised **and** that the message names the offending item.

- [ ] **(a) length disagreement.** A record with `len(X) == 4`, `len(Y) == 4`, `len(T) == 3`,
      `length == 4` raises; the message contains that record's `(task, name, subject)`.
- [ ] **(a) `length` field lying.** `len(X) == len(Y) == len(T) == 4` but `length == 5` raises.
      This is the case a naive `len(X) == len(Y) == len(T)` check would miss, and
      `COCOSearch_evaluation.__getitem__` iterates `range(fixation["length"])`, so it would
      `IndexError` at load time — catching it here is the point.
- [ ] **(b) missing query subject.** Fixture containing only subjects 0 and 1 with
      `fewshot_subjects=[0,1,2]` raises, naming `2`.
- [ ] **(c) ragged image — too few.** One `(task, name)` with 2 subjects and the rest with 3 raises;
      the message names that pair and prints `2 != 3`.
- [ ] **(c) ragged image — too many.** One `(task, name)` with 4 records (a duplicate
      `(name, subject)`) raises. Duplicates are as fatal as omissions: they shift the slice.
- [ ] **(c) passes on a well-formed fixture.** 3 images × 3 subjects raises nothing and returns
      `n_images == 3`, `n_cells == 9`.
- [ ] **(d) `jpg` in a task name.** A fixture with `task == "jpgholder"` raises, quoting FR4.4's
      `str.replace` trap. Verify by asserting `"jpgholder/x.jpg".replace("jpg", "pth")` ==
      `"pthholder/x.pth"` — a path that cannot exist.
- [ ] **(f) missing stimulus file.** A `(task, name)` with no file under `image_root` raises,
      listing the relative path `"<task>/<name>"`.
- [ ] **Split filtering.** A fixture whose `train` split is deliberately ragged but whose `test`
      split is clean passes with `split="test"` and raises with `split="train"`. Confirms the
      checker filters before checking.
- [ ] **Subject filtering.** A fixture containing subjects 0–9 where only 0, 1, 2 are well-formed
      passes with `fewshot_subjects=[0,1,2]`. Confirms the checker filters to the query set the way
      `select_fewshot_subject()` does, rather than checking the whole dataset.

### Group 3 — `check_fixations` soft counters and `check_features` (FR3.5e, FR14.3, FR4.6)

- [ ] **(e) out-of-range coordinates are counted, not raised.** A fixture with `X == 512.0` and one
      with `Y == -1.0` returns `oob == 2` and does **not** raise. Never clamp.
- [ ] `x == 511.9` and `y == 319.9` are in range; `x == 512.0` is not. Bound is half-open `[0, 512)`.
- [ ] **`n_short` counts scanpaths with `length < 3`.** A fixture with lengths `[1, 2, 3, 4]` returns
      `n_short == 2`. `length == 3` is not short.
- [ ] The returned dict serialises with `json.dumps` (all values are `int`/`str`/`list`, no numpy
      scalars) — the run script tees it into `preflight_fixations.json`.
- [ ] **`check_features` builds the loader's exact path.** For `task="potted_plant"`,
      `name="000000012345.jpg"`, the checked path ends
      `potted_plant/000000012345.pth`. Assert it equals
      `os.path.join(feat_dir, "{}/{}".format(task, name).replace("jpg", "pth"))` character for
      character — computed the same way, not re-derived.
- [ ] `check_features` raises on a missing `.pth`, listing at most 10 paths plus a total count.
- [ ] `check_features` raises on a `.pth` of shape `(768, 1024)` or `(2048, 768)`, reporting the
      actual shape alongside the expected `(768, 2048)`.
- [ ] `check_features` returns `{"n_checked": N, "missing": [], "bad_shape": []}` and exits 0 on a
      correct fixture — the run script's `if ! check_features` guard (FR8.6) depends on the exit code,
      so assert the CLI exits **0** on success and **non-zero** on failure.
- [ ] `check_features` deduplicates: a fixture with 3 subjects × 1 image checks **1** feature file,
      not 3.
- [ ] Neither `check_fixations` nor `__init__` imports `torch` — assert by parsing the module source
      for `import torch`. They must run on the login node (plan Step 2).
- [ ] Nothing in `tools/cocofv_prep/` imports from `ISP/*/GazeformerISP/src/utils/` (TechStack §6.9's
      rule, applied to this package).

### Group 4 — Environment and staging (FR1, FR6)

- [ ] `python -c "import multimatch_gaze; print(multimatch_gaze.__version__)"` prints exactly `0.1.3`
      inside the `isp` env. Any other value: the run script exits non-zero and this box stays open.
- [ ] The FR1.2 version dump appears in `logs/cocofv_out_%j.log` above the preflight output.
- [ ] `ls -la ISP/COCO_FV/GazeformerISP/src/assets/FV-ex-012/checkpoints/checkpoint_best.pth` resolves
      (symlink target readable), and the on-disk case of `FV-ex-012` is recorded in `notes.md`.
- [ ] `ls -la SE-Net/assets/FV-ex-012/fewshot_user_embedding_10.pt` resolves, with its verified
      filename recorded — `fewshot_user_embedding_10.pt` vs the `SE-Net/README.md` spelling
      `fewshot_subject_embedding.pt` (FR6.5).
- [ ] `torch.load(user_emb_path)` returns a **tensor** (not a dict, not an `nn.Embedding`) of shape
      `(S, 384)`, `S >= 3`, dtype `float32` (FR6.4). Record `S`.
- [ ] `model.load_state_dict` completes without a missing/unexpected-key error (FR6.6). If it does
      not, the failure and its non-invasive resolution are in `notes.md`, and
      `git status --porcelain ISP/COCO_FV/GazeformerISP/src/utils/` is still empty.
- [ ] `np.load(emb_path, allow_pickle=True).item()` is a dict containing key `"free-viewing"` whose
      value has shape `(768,)` and dtype `float32` (FR5.2). Record whether it came from the
      checked-in `src/embeddings.npy` or from `text_data()` (FR5.4).

### Group 5 — D1 frozen-code integrity

- [ ] `git status --porcelain ISP/COCO_FV/GazeformerISP/src/utils/` is **empty** after the whole
      feature. Non-empty is a D1 violation and blocks reporting (FR10.8).
- [ ] `git status --porcelain ISP/OSIE/ ISP/COCO_Search18/` is empty — this feature touches neither.
- [ ] `diff ISP/OSIE/.../evaltools/scanmatch.py ISP/COCO_FV/.../evaltools/scanmatch.py` is empty
      (already verified; re-assert after the run).
- [ ] `diff` of `visual_attention_metrics.py` between the two branches is run and its result — empty
      or not — is recorded in `notes.md` §Divergences (FR10.7).
- [ ] No new `.pyc` files are left in upstream directories: `git status --porcelain` shows no
      untracked `*.pyc` outside the 89 already-tracked `__pycache__` dirs (FR15.4, TechStack §6.6).
- [ ] `git status --porcelain` shows no `*.pth`, `*.pt`, `*.h5`, `*.npy` staged for commit (FR15.3).

### Group 6 — Run-script behaviour (FR8)

- [ ] `bash -n bash/test_cocofv.sh` parses clean.
- [ ] With `FV_IMAGE_ROOT` pointed at a nonexistent path, the script exits **non-zero before**
      `conda activate` completes any GPU work, and the error names the path (FR2.4).
- [ ] With a deliberately ragged `fixations.json`, the script exits non-zero at the
      `check_fixations` step and **never reaches** `src/test.py`. Confirm no
      `result/FV-ex-012/log/` output was produced by that submission.
- [ ] `FORCE_FEATURES=0` on a second submission with a complete feature cache **skips** extraction:
      the log shows the `check_features` success and no "Extracting image features" line (FR8.6).
- [ ] `FORCE_FEATURES=1` re-extracts.
- [ ] The script contains no absolute path outside the tunables block (FR2.3, TechStack §6.4) —
      `grep -n '/mnt/beegfs' bash/test_cocofv.sh` matches only inside that block.
- [ ] `image_features` never appears in an `rsync`/`cp` to `$LOCAL_SCRATCH`, and the exclusion carries
      its explanatory comment (FR4.7, FR8.7).
- [ ] `$FEWSHOT_SUBJECTS` is unquoted at the `--fewshot_subject` position and the resolved namespace
      in the log shows `fewshot_subject: [0, 1, 2]` — a **3-element list**, not `['0 1 2']`. This is
      the check that catches an over-eager quoting "fix".
- [ ] Each seed's artefacts land in `result/FV-ex-012/log/seed{0,1,2}/` before the next run overwrites
      the shared filename (FR15.2).

### Group 7 — `prediction.json` schema (FR12)

- [ ] `json.load` succeeds — no `TypeError: Object of type float32 is not JSON serializable`
      (FR12.3). If it failed, the workaround is in the run script and recorded, and
      `utils/evaluation.py` is still untouched.
- [ ] Every record has exactly the keys `{"name", "task", "subject", "X", "Y", "T", "length"}`
      (FR12.1) — assert the key set, so a schema drift is caught rather than assumed.
- [ ] `len(records) == n_test_images * subject_num * eval_repeat_num` (FR12.4). Compute
      `n_test_images` from `preflight_fixations.json`, not from the prediction file.
- [ ] `(name, task, subject)` is unique across records when `eval_repeat_num == 1`.
- [ ] `set(subject values) == {0, 1, 2}` (FR11.2 — dense indices, which equal the real ids only for
      this query set).
- [ ] `len(X) == len(Y) == len(T) == length` for every record, and `length <= 20`
      (`args.max_length`, FR9.4).
- [ ] `T` values are floats in milliseconds — spot-check that the median is in a plausible fixation
      range (roughly 100–800 ms), not 0.1–0.8. Catches a missed `* 1000` (FR12.2).
- [ ] `X` values lie in `[0, 512)` and `Y` in `[0, 320)` — the `Sampling` output space, matching
      `args.width`/`args.height`.
- [ ] `name` is a bare filename with no `/`, while the ground-truth loader keys on `"{task}/{name}"`.
      Recorded explicitly for F6 (FR12.2).

---

## Data Validity

Sanity checks on the numbers actually produced. Each is a line in `notes.md`, not a pytest.

### Group 8 — The reported metrics (FR13)

- [ ] `SM`, `MM`, `SED` printed by `test.py` match the values recomputed by hand from the log's
      `ScanMatch` / `MultiMatch` / `VAME` lines: `SM == scipy.stats.hmean([wo_dur, w_dur])`,
      `MM == mean(5 MultiMatch dims)`. A mismatch means the log was misread.
- [ ] All five MultiMatch dimensions lie in `[0, 1]`. A dimension at exactly `0` or `1` for every
      image is a wiring failure, not a result.
- [ ] Both ScanMatch values lie in `[0, 1]` and with-duration < without-duration (the temporal
      constraint can only reduce the match). If not, investigate before reporting.
- [ ] `SED > 0` and `STDE ∈ [0, 1]`.
- [ ] Retrieval: `R@1 <= R@2 <= R@5` and `MRR ∈ [0, 1]`. With `subject_num == 3`, `R@5` is
      necessarily `100.0` and `R@2` — logged as `R@3` — covers 2 of 3 ranks, so chance is ≈ 66.7 %.
      **Only `R@1` and `MRR` are informative at this cohort size.** State that; it is the same
      degeneracy Roadmap OPEN-5 predicts for a 2-subject EVE cohort, observed here on data whose
      answer is published.
- [ ] The three seeds' `SM`, `MM`, `SED` are tabulated with their spread. The spread is the noise
      band FR13.2's success criterion uses.
- [ ] Each of `SM`, `MM`, `SED` lands within that band of the paper's COCO-FreeView row. If any does
      not, the feature is **not** done (FR13.4) and the discrepancy is investigated and written up
      before F6 or F7 proceed.
- [ ] The retrieval column in `notes.md` is labelled **`R@2` (logged as `R@3`)** with a pointer to
      FR10.4. If it is compared to the paper's `R@3` column, that comparison carries an explicit
      caveat that the published number was produced by the same `rank < 2` code.

### Group 9 — Coverage and denominators (FR14.2, FR14.3)

- [ ] `len(test_loader)` in the log equals the number of distinct `(task, name)` pairs in the test
      split from `preflight_fixations.json`. Confirms FR9.6: **no `i_batch > 100` cap**, the full
      split was scored. A value of exactly 101 would mean a cap exists after all — re-read `test.py`.
- [ ] Number of `(image, subject)` cells `== n_images * 3`, recorded.
- [ ] Cells dropped by FR10.3's `-1` filtering, recorded. A large drop means many cells were never
      scored and the mean is over a different population than assumed.
- [ ] MultiMatch rows dropped by `is_eliminating_nan=True`, recorded (D7). SED/STDE do **not** get
      that treatment — note it, since the two means then have different denominators.
- [ ] Retrieval rows skipped by FR10.5's all-`-1` guard, recorded — the retrieval denominator is the
      number of scored rows, not `n_images * subject_num`.
- [ ] `n_short` (ground-truth scanpaths with fewer than 3 fixations) recorded, with the note that
      each was padded with `(1., 1., 1e-3)` inside the frozen evaluator and the padded array then
      replaced the original for **all** subsequent metrics in that cell (TechStack §4).
- [ ] Out-of-range coordinate count from FR3.5(e) recorded. Non-zero means the authors' labels exceed
      the 512×320 frame; state it, do not clamp.
- [ ] Ground-truth scanpath lengths are **not** truncated to 20 while predictions are (FR9.4). Record
      the ground-truth length distribution (min / median / max) so the asymmetry is visible and is not
      later mistaken for a bug.

---

## Data Architecture Integrity

### Group 10 — Keying invariants (D4)

- [ ] **`select_fewshot_subject` remap verified, not assumed.** With `--fewshot_subject 0 1 2` the
      remap is the identity. Assert it by checking that a prediction for dense subject `k` is scored
      against the ground truth of real subject `k`: pick one `(task, name)`, pull its three ground
      truth records from `fixations.json` and its three prediction records, and confirm the subject
      values line up (FR11.1).
- [ ] **The remap is order-dependent.** Verify on a fixture that
      `select_fewshot_subject(..., fewshot_subjects=[2, 0, 1], ...)` maps `2 → 0`, `0 → 1`, `1 → 2` —
      i.e. **argument order**, not sorted order. Record in `notes.md` that the run's identity mapping
      is a property of passing `0 1 2` ascending and nothing more (FR11.3). This is the trap F5 must
      avoid for EVE, where `subject_id_map.json` exists precisely to close it.
- [ ] **No phantom subjects.** The set of `subject` values in `prediction.json` equals the set in the
      filtered ground truth. No id appears in one and not the other.
- [ ] **Per-image subject order is preserved through the slice.** `test.py` builds
      `image_prediction_dict[idx]` from `[idx*subject_num : (idx+1)*subject_num]`. Confirm on one
      image that the k-th prediction corresponds to the k-th entry of
      `COCOSearch_evaluation.imgid_to_sub[img_name]`. FR3.5(c) is what makes the slice valid — the
      equal-subject check is **not** bypassable and must not be downgraded to a warning.
- [ ] **The score matrix diagonal is the reported number.** Confirm from `evaluation.py` (read-only)
      that row index = predicted subject, column = ground-truth subject, and that the reported
      diagonal metrics are taken from `collect_*_diag_rlts`. Record the confirmation; do not edit.
- [ ] **`imgid` keying is `"{task}/{name}"`, not `name`.** Confirm that two identical basenames under
      different categories (if any exist in the test split) are treated as two distinct stimuli. If
      none exist, record that fact — it means the collision case is untested here and F6 must not
      assume `name` alone is a key (FR12.2).
- [ ] **Feature path derivation is single-sourced.** `check_features.py` and
      `COCOSearch_evaluation.__getitem__` must build the same string from the same inputs. Verified in
      Group 3; re-assert here that no third derivation exists anywhere in `tools/cocofv_prep/`.

### Group 11 — Constitution consistency (FR16)

- [ ] `Roadmap.md` §4 no longer lists COCO-FreeView as out of scope, and **still** lists COCO-Search18
      and anything task/search-conditioned.
- [ ] `Roadmap.md` §1, §2, §3 and §6 all describe F1 as the COCO-FreeView baseline — no residual
      "OSIE baseline" text anywhere except the clearly-marked superseded note.
- [ ] `Roadmap.md` F6 records that its fixture is now a COCO_FV artefact with the FR12.2 schema
      divergences.
- [ ] `TechStack.md` states that F1's baseline branch and F5's template branch are deliberately
      different, with the reason (FR16.4). Grep confirms `ISP/OSIE/GazeformerISP/` is still annotated
      as the branch F5 mirrors.
- [ ] `TechStack.md` §4.2 carries FR10's divergence table with the warning that `evaluation.py` is not
      one file replicated across branches (FR16.5).
- [ ] `TechStack.md` §1 command conventions include `sbatch bash/test_cocofv.sh` and the
      `tools/cocofv_prep/` CLIs.
- [ ] No claim anywhere in the constitution survives that this run licenses conclusions about *our*
      subjects. It proves the environment and the metric code, nothing more — the domain-gap statement
      remains F7's job.
