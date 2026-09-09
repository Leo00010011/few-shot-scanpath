# F1 — Reproduce the OSIE eval baseline on the cluster

> Read against [Mission.md](../constitution/Mission.md), [TechStack.md](../constitution/TechStack.md),
> [Roadmap.md](../constitution/Roadmap.md).
> Roadmap feature: **F1**. Satisfies **D1, D5, D6, D7, D8**.
>
> **Reinstated 2026-09-09.** This spec was withdrawn on 2026-09-08 in favour of
> [`../2026-09-08-cocofv-baseline-on-cluster/`](../2026-09-08-cocofv-baseline-on-cluster/) and is now
> back in force; that one is marked superseded and retained for its FR10 divergence table only.
> COCO-FreeView was withdrawn because its **test split is a held-out challenge benchmark** - no public
> test labels, so no reproducible baseline (Roadmap section 4, now a permanent out-of-scope entry).
> **See the Amendments section at the end of this file** for what changed on reinstatement: the run
> script moved to the repo root, preflight tooling was added, and Finding H (the duration-bin label
> file) was discovered.

---

## Goal

Stand up the `isp` conda environment on the GPU cluster, wire the released OSIE checkpoint and few-shot
subject embedding into the paths `src/test.py` expects, produce the Stage-B image features, and run the
query-set (unseen-subject) evaluation end to end so that the resulting MultiMatch / ScanMatch / SED / STDE
/ retrieval block can be placed next to the paper's OSIE row. The purpose is *not* the numbers themselves
— it is to prove that the environment, the checkpoint, and the frozen metric code all work **before our
own data enters the picture**, so that every later discrepancy in F5 has an unambiguous cause. F1 is the
only point in the project where a mismatch can be attributed to the setup rather than to our dataset.

---

## Scope

### In scope
- Building the `isp` conda environment from `ISP/environment.yml` on the cluster and recording every
  deviation from the [TechStack.md](../constitution/TechStack.md) §1 pin list.
- Staging the OSIE 800×600 stimuli and running Stage B (`preprocess/feature_extractor.py::image_data`)
  to produce `src/data/image_features/*.pth`.
- Verifying the on-disk case and filenames of the checkpoint and the user-embedding tensors, and placing
  them at the paths `test.py` resolves.
- One **additive, print-only** edit to `ISP/OSIE/GazeformerISP/src/test.py` that logs the full D6 metric
  block (means **and** standard deviations, plus the retrieval block) instead of the means-only subset it
  logs today.
- A documented, path-parameterised run script `ISP/OSIE/GazeformerISP/bash/test.sh`.
- The **query-set** run: `--fewshot_subject 10 11 12 13 14`, `--eval_repeat_num 1`, repeated across a
  seed sweep so "within sampling noise" has a measured spread behind it.
- Verifying the produced `prediction.json` against the shipped
  `result/git-osie-useremb-ex-10to15/log/prediction.json` schema, which F6 will assume.
- Recording, inside this spec folder, the resolved environment and the baseline number table.

### Explicitly out of scope
- **The base-set run** (`--ex_subject 10 11 12 13 14` with `train_user_embedding.pt`). It exercises
  `recover_subject_ids()`, but it is not the path F5 mirrors for EVE and it is not what the shipped
  `prediction.json` corresponds to. Deferred; may be picked up by F7 if the write-up needs it.
- Any edit to `utils/evaluation.py` or `utils/evaltools/*` — frozen (D1).
- Any edit to `dataset/dataset.py`, `models/*`, or `preprocess/feature_extractor.py`. Where upstream is
  broken (see Finding D) F1 works around it at the call site, never in the file.
- Training, fine-tuning, the RL path, and the COCO branches (D8, Roadmap §4).
- Building the `senet` environment, Detectron2, or MSDeformAttn. F1 reuses the released
  `fewshot_user_embedding_10.pt`; no subject embedding is generated (that is F3 / OPEN-2).
- The offline re-scorer itself — F6. F1 only guarantees the artefacts it will consume.
- Anything touching the EVE bridge or OPEN-5.

---

## Findings that fix the requirements below

Established by reading the code on 2026-09-08. They *close* three open F1 checklist items and remove work
the roadmap anticipated, so they are stated before the requirements that depend on them.

**A. The `i_batch > 100` cap is inert for OSIE.** (Closes the roadmap's "decide and document" item and
TechStack §5 for this feature.) The `test` split of the shipped `fixations.json` contains exactly
**70 images × 15 subjects = 1050 records**; `OSIE_evaluation.__len__` returns the number of *images*, so
at `--batch 1` the loader yields 70 batches — below the 101-batch cap, which therefore never fires. The
shipped `prediction.json` holds **350 records = 70 images × 5 subjects**, independently confirming the
published number was computed over the **full test split**, not a truncated 101 images. The cap is left
in place unmodified for F1; it becomes live only in F5, where the roadmap already flags it.

**B. `test.py` does not read the stimulus images.** `OSIE_evaluation.__getitem__` loads only
`join(self.feature_dir, name.replace('jpg','pth'))`; the `Image.open` / `transform` lines are commented
out and `stimuli_dir` is stored but never used. `--img_dir` is therefore irrelevant to inference. The raw
stimuli are needed by Stage B **only**.

**C. `embeddings.npy` and `fixations.json` already ship** in `ISP/OSIE/GazeformerISP/src/data/`
(3,459 B and 6,058,435 B). Stage B for F1 reduces to `image_data()`; `text_data()` need not run.

**D. `feature_extractor.py`'s `__main__` is broken.** Its last line is
`text_data(dataset_path=args.p, ...)` and `args.p` is never defined — running the module as a script
raises `AttributeError`. Combined with Finding C, F1 invokes `image_data()` directly rather than running
the module, which sidesteps the bug without editing a file (Working convention 2, additive over
invasive).

**E. `--num_fewshot` and `--random_support` are inert on the test path.**
`select_fewshot_subject()` returns `filtered_scanpath` immediately when `split != 'train'`, before any
sampling. On `type="test"` they change nothing but the log filename
(`log_test_subject_{num_fewshot}_{random_support}.txt`). They must not be described as controlling the
query set. (`args.log_root` is likewise never touched on this path — `OSIE_evaluation` passes `None` for
`log_dir` — so `test.py`'s lack of a `--log_root` argument is not a latent crash.)

**F. `pr5` is degenerate at `subject_num=5`.** `p2g` computes `r5` as the fraction of ranks `< 5`, and
with 5 subjects every rank lies in `{0..4}`. `pr5` is therefore identically **100.0** and carries no
information. It is still reported (D6 names R@5) but must be annotated as structurally saturated.

**G. `cur_metrics_std` has no retrieval key.** The evaluator populates `cur_metrics_std` for
`MultiMatch`, `ScanMatch` and `VAME` only; `retrieval scanmatch w/ duration` exists in `cur_metrics`
alone. Any std-logging loop must tolerate the asymmetry rather than assume parallel structure.

**H. There are two OSIE label files and only one of them is a `fixations.json`.** *(New 2026-09-09.
This is the most dangerous thing in F1 and it was not known when this spec was first written.)*
`data/osie_fixations_update_duration.json` and `ISP/OSIE/GazeformerISP/src/data/fixations.json` carry
the **same 10,500 records, the same `(name, subject, split)` key set, the same nine keys, and
byte-identical `X`/`Y`**. Only `T` differs: the branch file holds durations in **milliseconds**
(20-1975; 20-1033 on the test split), while `data/osie_fixations_update_duration.json` holds the
**decile bin index, 0-9** - verified by joining on `(name, subject)`: ten equal-count buckets,
bin 0 = 20-96 ms ... bin 9 = 358-1975 ms.

Feeding the binned file to `test.py` **raises nothing**. Every TechStack 3.1 invariant holds.
`evaluation.py` then multiplies a bin index by 1000 and hands it to ScanMatch's `TempBin=50`, and
MultiMatch compares bins along its duration dimension - so both headline `SM` and `MM` come out
plausible, wrong, and uncomparable to the published table. Exactly the silent degradation D7 exists
to stop.

Consequences, all binding: `--fix_dir` is the **branch's** `src/data/fixations.json` and never the
`data/` file; FR3.5 gains invariant **(g)** below; and TechStack gains section 3.7 recording the pair.

---

## Functional Requirements

### Environment and assets

**FR1 — The `isp` environment is built and its deviations recorded.**
Create the env from `ISP/environment.yml` on the cluster. Capture `conda list` and `pip freeze`, and
record in `environment.md` inside this spec folder the resolved version of every package in the
[TechStack.md](../constitution/TechStack.md) §1 pin table, marking each `match` or
`deviation: pinned X, got Y`. **`multimatch-gaze` must resolve to exactly `0.1.3`**; any other version is
a hard stop, not a deviation to note, because the MultiMatch numbers would no longer be comparable to the
published table. `numpy` must resolve to `1.23.5` and `scipy` to `1.10.0`; a deviation in either is
permitted only with an explicit written note, since structured-array dtype behaviour and
`scipy.stats.hmean` are load-bearing for the metric contract.

**FR2 — Checkpoint and embedding paths are resolved by observation, not assumption.**
Before wiring any path, enumerate the actual on-disk names under `weights/OSIE-*/OSIE/` on the cluster and
record them verbatim. The known-good source layout on the dev machine is:

```
weights/OSIE-20260904T121550Z-1-001/OSIE/checkpoint_best.pth
weights/OSIE-20260904T121550Z-1-001/OSIE/ckp_11999.pt
weights/OSIE-20260904T121550Z-1-001/OSIE/fewshot_user_embedding_10.pt
weights/OSIE-20260904T121550Z-1-001/OSIE/train_user_embedding.pt
```

The two consumer paths must end up as:

| consumer | resolved path (relative to `ISP/OSIE/GazeformerISP/`) | source |
|---|---|---|
| `--evaluation_dir` + `/checkpoints/checkpoint_best.pth` | `src/assets/OSIE-ex-10to15/checkpoints/checkpoint_best.pth` | `weights/OSIE-*/OSIE/checkpoint_best.pth` |
| `--user_emb_path` | `../../../SE-Net/assets/OSIE-ex-10to15/fewshot_user_embedding_10.pt` | `weights/OSIE-*/OSIE/fewshot_user_embedding_10.pt` |

The checkpoint target directory currently contains only a placeholder `README.md`, and `SE-Net/assets/`
does not exist at all on the dev checkout. On the case-sensitive cluster filesystem the
`OSIE-ex-10to15` vs `osie-ex-10to15` and `user` vs `subject` ambiguities recorded in TechStack §3.4 must
be resolved by `ls` and the resolved spelling written into `bash/test.sh`. Staging is by **symlink or
copy**; `weights/` itself is git-ignored and must not be moved.

**FR3 — The user embedding has the expected shape.**
Before the first run, load `fewshot_user_embedding_10.pt` and assert its trailing dimension equals
`args.subject_feature_dim == 384` and that it provides at least `args.subject_num == 5` rows. A mismatch
raises immediately rather than surfacing as a silent broadcast inside `gazeformer`. The observed shape and
dtype are recorded in `environment.md`.

### Stage B — image features

**FR4 — OSIE stimuli are staged into the hardcoded `train/` subdirectory.**
`feature_extractor.image_data()` reads `join(dataset_path, 'train/')` — a hardcoded subdirectory
(TechStack §3.2). The OSIE `.jpg` stimuli must be placed at `<stage_b_root>/train/*.jpg` by symlink or
copy. Editing the hardcoded path is out of scope; the filesystem is adapted to the code.

**FR5 — The output directory must pre-exist.**
`image_data()` calls `torch.save(..., join(output_path, 'image_features/', ...))` and never creates the
directory. `src/data/image_features/` must be created before invocation or every save raises
`FileNotFoundError`.

**FR6 — Stage B is invoked function-first, not module-first.**
Because of Finding D, `python preprocess/feature_extractor.py` must **not** be used. Stage B is invoked by
importing `image_data` and calling it directly (see the Public API Summary). `text_data()` is **not**
called: `src/data/embeddings.npy` already ships (Finding C) and regenerating it would make the run depend
on a `stsb-roberta-base-v2` download whose output is not guaranteed bit-identical to the shipped file.

**FR7 — Feature tensors are verified for count, shape and dtype.**
After Stage B, assert:
- one `.pth` exists for **every distinct `name`** in `fixations.json` across all splits, with `.jpg`
  mapped by the literal `str.replace('jpg','pth')` the loader uses;
- each loaded tensor has shape exactly `(768, 2048)` = `(24*32, 2048)` and dtype `torch.float32`;
- no tensor contains `NaN` or `Inf`.

A missing or misshapen file **raises and names the file** (D7). It must never be discovered as a
`torch.load` failure mid-run and never be silently skipped — note that `image_data()`'s
`overwrite=False` branch skips existing files, so a partial earlier run can masquerade as a complete one.

### The additive `test.py` edit

**FR8 — `test.py` logs the full D6 block.**
`ISP/OSIE/GazeformerISP/src/test.py` is **not** on the D1 frozen list. Exactly one region changes: the
metric-logging block after `comprehensive_evaluation_by_subject()` returns. The edit is **print-only** —
it must not touch `args`, the dataset, the model, the sampling, the evaluator call, or the
`prediction.json` write, and therefore cannot alter any computed number. After the edit the log contains:

1. every `cur_metrics[group][name]`, as today;
2. the matching `cur_metrics_std[group][name]` where the group exists in `cur_metrics_std`, tolerating
   the missing `retrieval scanmatch w/ duration` group (Finding G);
3. the headline `SM` / `MM` / `SED` line already printed, plus `STDE`, emitted through `logger` rather
   than bare `print` so it lands in the log file (D5).

`SM` remains `scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))` and `MM` remains
`np.mean(list(cur_metrics["MultiMatch"].values()))` — unchanged expressions, so the headline numbers are
bit-identical to an unedited run.

**FR9 — The `i_batch > 100` cap is left untouched.**
Per Finding A it is inert for OSIE, and removing it would make F1's run differ from the one that produced
the shipped `prediction.json`. It is documented in `results.md`, not modified.

### The run

**FR10 — A single documented run script exists.**
`ISP/OSIE/GazeformerISP/bash/test.sh` mirrors the style of the existing `bash/train.sh` and takes the
query-set run end to end. It must be invoked **from the `ISP/OSIE/GazeformerISP/` directory** — every
default path in `test.py` is relative to it (TechStack §1). It parameterises `SEED` and passes every path
explicitly rather than relying on an argparse default, so the resolved command is legible from the script
alone. It must not hardcode an absolute path (Working convention 4).

**FR11 — The query-set run configuration is fixed.**

```
--fewshot_subject 10 11 12 13 14
--subject_num 5
--eval_repeat_num 1
--batch 1
--seed <swept>
--user_emb_path   <FR2 resolved fewshot embedding>
--evaluation_dir  src/assets/OSIE-ex-10to15
```

`--eval_repeat_num` stays at 1: it is what the shipped `prediction.json` used (its 350 records contain no
duplicate `(name, subject)` pair, and `predict_results` accumulates across repeats), and raising it would
produce a `prediction.json` with duplicate keys that F6's re-scorer cannot index.
`--num_fewshot` and `--random_support` are left at their defaults and documented as affecting **only the
log filename** (Finding E).

**FR12 — A seed sweep quantifies sampling noise.**
The run is repeated at **five seeds — `0, 1, 2, 3, 4`** — with everything else identical. Seed 0 is the
reference run. For each of `SM`, `MM`, `SED`, `STDE` and `pmrr`, report the mean and the min–max range
across the five seeds. That spread is the operational definition of "within sampling noise" in the
done-condition; no comparison to the paper may be called a match or a mismatch without it.

**FR13 — Every run is self-describing.**
Each run writes `result/OSIE-ex-10to15/log/log_test_subject_<num_fewshot>_<random_support>.txt`
containing the full resolved argument namespace (already emitted by the existing loop over `vars(args)`)
followed by the FR8 metric block (D5). Because the log filename does **not** include the seed, the sweep
must rename or relocate each per-seed log so run *k* does not overwrite run *k−1*; the script is
responsible for this and the naming convention is recorded in `results.md`. The same applies to
`prediction.json`, which is written to a fixed path within that folder.

### Artefact verification

**FR14 — `prediction.json` conforms to the F6 contract.**
The produced `result/OSIE-ex-10to15/log/prediction.json`, checked against the shipped
`result/git-osie-useremb-ex-10to15/log/prediction.json`, must satisfy:

- exactly the five keys `{"name", "subject", "X", "Y", "T"}`, no more (TechStack §3.5);
- `len(records) == 70 * 5 == 350`;
- `X` and `Y` are `int`, `T` is `int` milliseconds;
- `sorted(set(subject)) == [10, 11, 12, 13, 14]` — the **original** OSIE subject ids, not the dense
  `0..4` the loader remapped to. This is the D4 roundtrip: `get_prediction_list()` takes the
  `args.fewshot_subject[subject_idx]` branch, so a correct run recovers 10–14 and an off-by-one in the
  subject axis surfaces here and nowhere else;
- `set(name)` has cardinality 70 and equals the set of `name`s on the `test` split of `fixations.json`;
- every `(name, subject)` pair occurs exactly once;
- `len(X) == len(Y) == len(T)` per record, with `1 <= len(X) <= 16` (`min_length` / `max_length`).

**FR15 — Coverage is asserted, not assumed (D7).**
Before reporting any number, assert on the `test` split that every image carries exactly the same number
of subjects — 15 before few-shot filtering, 5 after — the equal-subject invariant that keeps
`evaluation.py`'s `(subject × subject)` matrix square (TechStack §3.1). Assert the evaluator was handed 70
image entries. Report the count of **NaN MultiMatch rows dropped** by `is_eliminating_nan=True`: the
evaluator reduces `70 × 5 = 350` diagonal MultiMatch rows to
`collect_multimatch_diag_rlts.shape[0]` before taking the mean, and the difference must be re-derived
from the returned `score_details` (whose MultiMatch slots hold `NaN` where the metric failed and `-1`
where the cell never ran) — no edit to the frozen file. A dropped-row count that is not reported
alongside the MultiMatch mean violates D7.

**FR16 — An all-`-1` result is a failure, not a score.**
Uninitialised evaluator cells are `-1`, not `NaN` (TechStack §4). If any diagonal metric mean is exactly
`-1.0`, or if `score_details` is entirely `-1`, the run is treated as **failed** — the loop never
executed — and no number is recorded.

### Reporting

**FR17 — The baseline table is recorded in this spec folder.**
`results.md` holds: the five MultiMatch dimensions, ScanMatch with/without duration, `SED`, `STDE`,
`SED_best`, `STDE_best`, and the retrieval block (`pmrr`, `pr1`, `pr3`, `pr5`, `rsum`) — each as
**mean ± std** from the evaluator — for the seed-0 reference run; the FR12 seed spread; and the paper's
OSIE row alongside. `pr5` is annotated as structurally saturated at 100.0 (Finding F). The exact command,
the resolved env, and the resolved asset paths accompany the table.

**FR18 — Failure to reproduce is recorded, not worked around.**
If the numbers fall outside the FR12 spread of the paper's row, F1 is **not** declared done, and no
attempt is made to close the gap by changing metric parameters, the resize, or the evaluator. The
discrepancy is written up in `results.md` and escalated as a new roadmap open decision, since a setup that
cannot reproduce OSIE cannot license any conclusion about our data (D8).

---

## Public API Summary

No new modules. F1 adds one shell script, one print-only edit, and calls one existing function.

```python
# Stage B — invoked directly, NOT via `python preprocess/feature_extractor.py` (Finding D).
# Run from ISP/OSIE/GazeformerISP/ with the `isp` env active.
from preprocess.feature_extractor import image_data

image_data(
    dataset_path: str,   # parent of the hardcoded 'train/' subdir holding the OSIE .jpg   (FR4)
    output_path:  str,   # parent of an ALREADY-CREATED 'image_features/' subdir           (FR5)
    device:       torch.device = torch.device('cuda:0'),
    overwrite:    bool = True,   # True, so a partial earlier run cannot pass as complete  (FR7)
) -> None
```

```python
# FR8 — the only edit to test.py: the metric-logging region. Print-only.
logger.info("The metrics for best model performance are: ")
for metrics_key in cur_metrics.keys():
    for (metric_name, metric_value) in cur_metrics[metrics_key].items():
        std = cur_metrics_std.get(metrics_key, {}).get(metric_name)      # Finding G
        if std is None:
            logger.info("{:10}-{:15}: {:.4f}".format(metrics_key, metric_name, metric_value))
        else:
            logger.info("{:10}-{:15}: {:.4f} +/- {:.4f}".format(
                metrics_key, metric_name, metric_value, std))
# SM / MM / SED expressions unchanged; STDE added; all routed through logger (D5).
```

```bash
# FR10 — ISP/OSIE/GazeformerISP/bash/test.sh, invoked from ISP/OSIE/GazeformerISP/
#   SEED=0 sh bash/test.sh
```

---

## Dependencies

| direction | artefact | contract |
|---|---|---|
| reads | `ISP/environment.yml` | source of the `isp` env (FR1) |
| reads | `weights/OSIE-*/OSIE/checkpoint_best.pth` | staged to `src/assets/OSIE-ex-10to15/checkpoints/` (FR2) |
| reads | `weights/OSIE-*/OSIE/fewshot_user_embedding_10.pt` | staged to `SE-Net/assets/OSIE-ex-10to15/` (FR2, FR3) |
| reads | OSIE 800×600 `.jpg` stimuli (external — NUS-VIP) | staged to `<stage_b_root>/train/` (FR4) |
| reads | `src/data/fixations.json` (shipped) | TechStack §3.1; 10,500 records, 15 subjects, 70 test images |
| reads | `src/data/embeddings.npy` (shipped) | key `"free-viewing"`, dim 768; **not** regenerated (FR6) |
| reads | `result/git-osie-useremb-ex-10to15/log/prediction.json` (shipped) | schema reference for FR14 |
| reads (frozen, D1) | `utils/evaluation.py`, `utils/evaltools/*` | called, never edited |
| writes | `src/data/image_features/*.pth` | one per stimulus, `(768, 2048)` float32 (FR7) |
| writes | `src/assets/OSIE-ex-10to15/checkpoints/checkpoint_best.pth` | staged, git-ignored |
| writes | `ISP/OSIE/GazeformerISP/bash/test.sh` | new, committed (FR10) |
| modifies | `ISP/OSIE/GazeformerISP/src/test.py` | print-only, one region (FR8) |
| writes | `result/OSIE-ex-10to15/log/log_test_*.txt` | per-seed, renamed (FR13) |
| writes | `result/OSIE-ex-10to15/log/prediction.json` | F6's input (FR14) |
| writes | `spec/2026-09-08-osie-eval-baseline/{environment,results}.md` | FR1, FR17 |
| unblocks | **F6** (offline re-scorer) | FR14 fixes the schema it reads |
| unblocks | **F2 → F5** | proves env + metrics before EVE data enters |


---

## Amendments - 2026-09-09 (reinstatement)

Recorded here rather than edited into the body above, so the reinstated spec stays diffable against
the version withdrawn on 2026-09-08.

### A1 - The run script is `bash/test_osie.sh` at the **repo root**, not `ISP/OSIE/GazeformerISP/bash/test.sh`

Supersedes the in-scope bullet and FR10. The COCO-FreeView detour built a repo-root `sbatch` script
with an environment-overridable tunable block, a `#SBATCH` header, per-seed artefact preservation, and
an exit-code-driven skip-extraction guard; that is strictly more than the branch-local `test.sh` this
spec originally called for, and it is the shape `sbatch` wants. `bash/test_cocofv.sh` was renamed and
retargeted rather than rewritten. Everything FR10 required is preserved; only the path changed.

Retarget details: `--fewshot_subject 10 11 12 13 14`, `--subject_num 5`,
`--evaluation_dir src/assets/OSIE-ex-10to15`, `--width 512 --height 384` (passed explicitly - they
*are* the metric screen), `--user_emb_path` pointing into `weights/OSIE-*/OSIE/`, and `py -m pip`
corrected to `python -m pip` (the `py` launcher is Windows-only and does not exist on the cluster).

`origin_size` is deliberately **not** passed: `test.py` does not forward it and `OSIE_evaluation`'s
`(600, 800)` default is already correct for OSIE's native stimuli (D3). This is the one place where
inheriting the default is right - F5's EVE branch is where it becomes a bug.

Stage B is invoked by importing `image_data()` directly rather than running
`feature_extractor.py` as a module (Finding D: its `__main__` crashes on `text_data(dataset_path=args.p)`,
`args.p` being undefined), and a symlink farm supplies the hardcoded `<dataset_path>/train/`
subdirectory the function expects. Both are call-site workarounds; the file is untouched
(working convention 2).

### A2 - Preflight tooling: `tools/osie_prep/` (FR3.5, FR4.6)

New, and not in the original spec. Retargeted from `tools/cocofv_prep/`; 34 pytest tests pass on
Windows CPU (`py -m pytest tests/osie_prep -q`). Three changes and no others:

- `OSIE_evaluation` keys on `fixation['name']` **alone**, not `"{task}/{name}"` - so the equal-subject
  check groups on the bare name and the feature path is `name.replace('jpg','pth')` with no category
  component. The stimulus directory is flat.
- Coordinate bounds are the native **800x600** frame, not COCO-FreeView's 512x320, and are overridable
  via `--origin-width` / `--origin-height`.
- `normalize_fixations.py` **dropped entirely.** The branch's shipped `fixations.json` is already
  canonical, so there is no transform to apply - and a leftover COCO normaliser could only corrupt it.
  A test asserts the file is absent.

`check_fixations` keeps the hard/soft split: (a) length agreement, (b) query-subject coverage,
(c) the equal-subject invariant, (d) the unanchored `replace('jpg','pth')` trap, (f) stimulus presence
and **(g)** below all raise; out-of-range coordinates (e) and short scanpaths are counted, never
clamped. `check_features` remains the sole torch importer and its **exit code** still drives the run
script's skip-extraction guard.

### A3 - FR3.5(g), new hard invariant: `T` is in milliseconds, not duration bins

`max(T)` over the evaluated split must exceed 20. Nothing that is a fixation duration in milliseconds
can fail this; the binned file of Finding H fails it at `max(T) == 9`. The error message names the
offending file and says what to use instead, because "check failed" without an offender is not a loud
failure (D7). The counter dict gains `t_min_ms` / `t_max_ms` so the run log carries the evidence.

Verified on the real artefacts:

| `--fix` | result |
|---|---|
| `ISP/OSIE/GazeformerISP/src/data/fixations.json` | **exit 0** - 70 images x 5 subjects = 350 cells, `t_min_ms` 20, `t_max_ms` 1033, `oob` 0, `n_short` 0, lengths 4/10/18 |
| `data/osie_fixations_update_duration.json` | **exit 1** - FR3.5g, `max T == 9` |

The 350 cells match the shipped `prediction.json`'s 350 records exactly, which is Finding A's
independent confirmation that the published number covers the full test split.

`n_short == 0` on the query split is worth noting: the frozen evaluator's pad-to-length-3 path
(TechStack section 4) never fires for F1, so that particular divergence cannot explain any gap
against the published row.

### A4 - Finding A is confirmed, not merely asserted

Re-derived from the code on 2026-09-09: `OSIE_evaluation.__len__` returns `len(self.imgid)`, and the
`test` split groups to **70 distinct images** carrying exactly 15 subjects each. So the loader yields
70 batches at `--batch 1` (18 at the default `--batch 4`) and `if i_batch > 100: break` never fires.
`test.py` is left unmodified for F1. Note the unit: the cap counts **batches**, and one batch is one
image with all its subjects - the recurring "101 images" phrasing is right for OSIE only by accident,
and F5 must not inherit the reasoning.

### A5 - The subject embedding, closing an open cluster check

`fewshot_user_embedding_10.pt` is `(10, 384)` while the query set is 5 subjects, which invited the
worry that the dense remap `10..14 -> 0..4` might index the wrong rows. Resolved on the dev machine:
**rows 5-9 are exactly zero**; rows 0-4 carry the five query subjects (row norms 11.5, 7.5, 9.5, 13.2,
11.1). `train_user_embedding.pt` is a different tensor entirely - all ten rows populated, norms about
18, no row shared with the few-shot file. The `10` in the filename is `num_fewshot` (10-shot), not a
subject count. No cluster verification needed; `bash/test_osie.sh` still prints the non-zero row list
at run time, because a wrong tensor staged into that path would otherwise score silently.

### A6 - Still outstanding from the original body

Unchanged and still to do: **FR8**, the additive print-only edit to `test.py` that logs the full D6
block (means *and* stds, plus the retrieval block) instead of today's means-only subset - mind
Finding G's asymmetry. And every cluster-only item: build the `isp` env, stage the 700 OSIE stimuli,
resolve the on-disk case of `OSIE-ex-10to15`, run Stage B, the `--seed 0 1 2` sweep, and `notes.md`.
