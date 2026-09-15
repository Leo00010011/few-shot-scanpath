# F5 — EVE eval branch + run script

> Spec folder 1 of 3: **requirements** · [plan.md](plan.md) · [validation.md](validation.md)
> Roadmap feature: **F5**. Created 2026-09-15.
> Satisfies **D1**, **D3**, **D4**, **D5**, **D6**, **D7**, **D8**.
> Consumes F2, F3 and F4's artefacts; authors no new data.

---

## Goal

Stand up `ISP/EVE/GazeformerISP/` — a new dataset branch mirroring `ISP/OSIE/GazeformerISP/` —
and the single run script that drives it, so that one documented command on the cluster takes our
38-participant, 354-stimulus EVE cohort through Stage D (inference with the released ISP checkpoint
and F3's `(38, 384)` subject embedding) and Stage E (the authors' unmodified
`comprehensive_evaluation_by_subject()`), at three seeds, and writes per-seed artefacts from which
every reported number can be re-derived offline. This is the feature that turns four completed
preparation stages into the mission's actual measurement. It is also the feature where every
silent-corruption trap the constitution has accumulated becomes live at once: a mis-scaled
coordinate space, a permuted cohort, a truncated test split, a per-image feature standing in for a
per-trial one, and a `prediction.json` whose subject ids are positional slots rather than
participants. Each of those produces a plausible, wrong, publishable number. The requirements below
exist mostly to make each of them impossible or loud.

---

## Scope

**In scope**

- A new branch `ISP/EVE/GazeformerISP/`, created as a **full verbatim copy** of the OSIE branch's
  `src/` tree, with edits confined to `dataset/dataset.py` and `test.py`.
- `EVE_evaluation`, an adaptation of `OSIE_evaluation` that loads **one feature tensor per trial**
  keyed by `exp_key`, and exposes each sample's trial keys and ground-truth lengths.
- `test.py` adaptations: explicit `origin_size=(1080, 1920)`, a parameterised batch cap, correct
  D4 subject-id recovery into `prediction.json`, the NSS/CC/KLD heatmap block, and a
  machine-readable `metrics.json`.
- Artefact handshakes: sha256 gates against `bridge_report.json`, `senet_report.json` and
  `feature_report.json` before any of the three is consumed.
- A CPU preflight tool `tools/eve_eval/` whose exit code guards the run, including the re-run of
  F2's heatmap bitwise-parity check under the cluster's numpy.
- A single run script `bash/test_eve.sh`, seed-swept over `0 1 2`, preserving per-seed artefacts.
- A pytest suite under `tests/eve_eval/`, CPU-only.

**Explicitly out of scope**

- **Any edit to `utils/evaluation.py` or `utils/evaltools/*`** — in the EVE branch they are
  byte-identical copies of the OSIE files and stay frozen (D1). The call site changes; the function
  never does.
- **Any edit to `ISP/OSIE/GazeformerISP/`.** F1's baseline must stay reproducible bit-for-bit.
- **Re-running F2, F3 or F4.** Their artefacts are consumed read-only and sha-gated.
- **Logging `cur_metrics_std` (the per-cell standard deviation).** Decided 2026-09-15: it stays
  deferred to **F6**, which recomputes everything from `prediction.json` on CPU. `metrics.json`
  carries an explicit `per_cell_std: null` with a reason rather than omitting the field, so a reader
  cannot mistake absence for zero. D6's std is supplied for now by the **across-seed** spread, which
  is a different quantity (TechStack §3.5b).
- **Pooling the three seeds into a run record.** That is F6/F7's aggregator; F5 only guarantees the
  per-seed artefacts it needs exist and are distinguishable.
- **`--eval_repeat_num > 1`.** Pinned to 1; any other value raises `IndexError` inside the frozen
  evaluator (TechStack §5).
- Training, fine-tuning, the RL path, new metrics, model changes (D8).

---

## Functional Requirements

### FR1 — Environment

**FR1.1** F5 runs in the **ISP-side** env (`scanpath`, or any env satisfying FR1.2). It does **not**
use `senet`, Detectron2, MSDeformAttn, the SE-Net encoder init pickles, `align_stage_prefix()`, or
`module load CUDA/11.6`. F3's embedding is consumed as a **file**. TechStack §1.0.

**FR1.2** The run writes `versions.txt` per seed, resolving: `python`, `torch`, `torchvision`,
`numpy`, `scipy`, `skimage`, `cv2`, `multimatch_gaze`, `h5py`, `pandas`, plus `torch.version.cuda`
and `torch.cuda.get_device_name(0)`. **`multimatch_gaze.__version__` must be exactly `0.1.3`** or the
run aborts before inference — it is the one metric-critical pin (TechStack §1.1). Every other version
is recorded, not enforced.

**FR1.3** `evedataset` is **not** imported anywhere on the F5 path, and the EVE bundle is **not**
read. F4's declared D2 deviation is contained to `tools/eve_prep/`; F5 consumes only artefacts.

---

### FR2 — Branch creation and frozen-file provenance

**FR2.1** `ISP/EVE/GazeformerISP/src/` is created by copying `ISP/OSIE/GazeformerISP/src/` in full:
`dataset/`, `models/`, `preprocess/`, `utils/`, `assets/` layout and `data/`. Mirroring exactly is
working convention 3 — it keeps upstream diffs readable and avoids a cross-branch import surface.

**FR2.2** `src/utils/evaluation.py`, `src/utils/evaltools/scanmatch.py` and
`src/utils/evaltools/visual_attention_metrics.py` in the EVE branch must be **byte-identical** to
their OSIE counterparts. A test asserts sha256 equality for all three (D1). This is the branch that
must not drift: TechStack §4.2 documents what happened to COCO_FV's `evaluation.py`.

**FR2.3** `src/models/*`, `src/utils/logger.py`, `src/utils/data_postprocess.py` and
`src/preprocess/*` are likewise byte-identical copies. The only two files that may differ from OSIE
are `src/dataset/dataset.py` and `src/test.py`; a test asserts exactly that set.

**FR2.4** In the copied `dataset.py`, `OSIE` → `EVE`, `OSIE_rl` → `EVE_rl`, `OSIE_evaluation` →
`EVE_evaluation`. `EVE` and `EVE_rl` are **dead on the eval path** and are retained unmodified apart
from the rename, for diff readability. A test asserts `test.py` imports `EVE_evaluation` only.

**FR2.5** `EVE_evaluation.__getitem__` must contain **no `str.replace` call**. A test walks its AST
and raises on any `<expr>.replace(...)`. The upstream idiom `img_name.replace('jpg','pth')` is
unanchored and is what FR5 replaces with concatenation (TechStack §3.2, §3.9).

---

### FR3 — Preconditions and artefact handshakes

All of the following are checked **before** the model is constructed. Each failure raises with the
offending path and both hashes; none warns (D7).

**FR3.1** These artefacts must exist, or the run aborts naming the file and stating that `data/` is
git-ignored so it never arrives by `git pull`:

| path | source |
|---|---|
| `data/eve_bridge/fixations.json` | F2 |
| `data/eve_bridge/gt_heatmaps.h5` | F2 |
| `data/eve_bridge/subject_id_map.json` | F2 |
| `data/eve_bridge/bridge_report.json` | F2 |
| `data/eve_senet/seed0/eve_fewshot_user_embedding_10_seed0.pt` | F3 |
| `data/eve_senet/seed0/senet_report.json` | F3 |
| `data/eve_features/image_features/` (1062 `.pth`) | F4 |
| `data/eve_features/embeddings.npy` | F4 |
| `data/eve_features/feature_report.json` | F4 |
| `<weights>/checkpoints/checkpoint_best.pth` | released |

**FR3.2** `sha256(fixations.json)` must equal **all three** of
`bridge_report["fixations_sha256"]`, `senet_report["fixations_sha256"]` and
`feature_report["fixations_sha256"]`. Expected `46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b`.
This is what stops the three upstream features from silently discussing different builds. A merely
**reordered** JSON changes the hash and is correctly rejected — bridge artefacts are addressed
positionally (working convention 10).

**FR3.3** `sha256(eve_fewshot_user_embedding_10_seed0.pt)` must equal
`senet_report["embedding_sha256"]` (expected prefix `e904a165c985b33f…`), and
`senet_report["seed"]` must equal the seed in the filename (F3 FR8.2).

**FR3.4** For every trial the run will score, `sha256(image_features/<exp_key>.pth)` must equal
`feature_report["feature_sha256"][exp_key]`. Checked for all 1062 in the preflight (CPU, ~6.7 GB
read); the in-process loader re-checks nothing.

**FR3.5** `args.num_fewshot` (10) must be `<= bridge_report["min_support_per_subject"]` (10). The
**minimum**, never `support_pool_size` (20): the support pools are per-subject and share no image
names, so the thinnest pool binds (TechStack §3.6).

**FR3.6** `bridge_report["origin_size"] == [1080, 1920]` must equal the `(origin_height,
origin_width)` the run passes. `gt_heatmaps.h5`'s root attrs `origin_size`, `action_map` and
`max_length` must equal `[1080,1920]`, `[24,32]` and `16` respectively.

**FR3.7** `embeddings.npy` must load to a dict containing the key `"free-viewing"` with a
`(768,)` float32 value, and must be **byte-identical** to
`ISP/OSIE/GazeformerISP/src/data/embeddings.npy`. Using the file F1 scored with is what keeps F1's
baseline and F5's run comparable on this input (TechStack §3.9).

**FR3.8 — the cohort invariants.** Over the `test` split of `fixations.json`:
- exactly **1062** records, **354** distinct `name`s, **38** distinct `subject`s;
- every scored `name` carries exactly `args.subject_num` (**3**) records — the uniform-count rule,
  forced by `evaluation.py`'s `-1`-initialised collectors reduced by a bare `np.mean()` with no
  `!= -1` filter on this branch (Mission P4, TechStack §3.6). Subject **identities** may differ from
  image to image; the diagonal is positional. A ragged image raises, naming it and its count;
- subject ids are dense `0..37`;
- `max(T)` over the split is **> 20**, the §3.7 duration-bin guard — a decile-bin file passes every
  other invariant and corrupts `SM` and `MM` silently;
- `condition == "freeview"`, `task == "none"` on every record.

**FR3.9** Every `(name, subject)` on the scored split must resolve to an `exp_key` in
`gt_heatmaps.h5`, and `image_features/<exp_key>.pth` must exist. A missing tensor raises listing up
to 10 offenders and the total (D7) — it is never skipped into the mean.

**FR3.10 — heatmap parity under the run's numpy.** `tools/eve_bridge/heatmaps.py`'s construction was
pinned bitwise against `OSIE.__getitem__` under numpy 2.1.2 on the Windows dev machine. The preflight
re-runs that parity check **in the run's own env** over the same 200 seeded cases and requires
bitwise equality (`np.array_equal`), aborting on any difference. TechStack §1's dev-machine reality
check demands this before `gt_heatmaps.h5` is trusted on the cluster.

---

### FR4 — `EVE_evaluation`

**FR4.1 — signature.** Defaults change only where EVE differs:

```python
EVE_evaluation(args, stimuli_dir, feature_dir, fixations_dir, task_emb_dir,
               heatmaps_dir,                       # NEW - path to gt_heatmaps.h5
               exp_key_map,                        # NEW - {(name, subject): exp_key}
               action_map=(24, 32),
               origin_size=(1080, 1920),           # (H, W); OSIE default (600, 800) would mis-scale
               resize=(384, 512),
               type="test", transform=None)
```

`resizescale_x = 1920 / 512 = 3.75`, `resizescale_y = 1080 / 384 = 2.8125` — **non-uniform**, the
deliberate squash of OPEN-4. F7 must state it.

**FR4.2** `stimuli_dir` is accepted for signature parity and **never read**. `data/eve_bridge/stimuli/`
is unused downstream (OPEN-6): its one-`.jpg`-per-`stimulus_name` export cannot represent what every
participant saw. A test asserts `EVE_evaluation` opens no image file.

**FR4.3** `exp_key_map` is built **once at construction** by the caller via
`tools.eve_prep.trial_keys.load_trial_exp_keys(heatmaps_path, fixations_path)` — torch-free by design
— and passed in. It is never rebuilt per `__getitem__`.

**FR4.4 — the identity-remap assertion (D4).** After `select_fewshot_subject()` runs, every record's
`subject` must be unchanged from the file. `select_fewshot_subject()` remaps `args.fewshot_subject`
to a dense `0..N-1`; only an **ascending** `0 1 2 … 37` produces the identity, and only under the
identity does `exp_key_map`'s `(name, subject)` key still address the right trial and does
`subject_embed[subjects]` still put row *i* on participant *i*. Any other argument order silently
permutes the entire cohort and every metric still looks plausible. The constructor therefore captures
each record's subject before the call and raises on any change, naming the first offender.

**FR4.5 — `__len__` is the image count**, unchanged: `len(self.imgid)` = 354. One batch item yields
all 3 subjects for its image.

**FR4.6 — `__getitem__` return** extends the OSIE dict with two keys:

| key | type | contents |
|---|---|---|
| `trial_key` | `list[str]`, len 3 | `"{name}\|{dense subject}"` per subject, in `imgid_to_sub` order |
| `length` | `np.ndarray (3,) int32` | `min(fixation["length"], max_length)` per subject |

Existing keys (`image`, `fix_vectors`, `firstfix`, `img_name`, `subject`, `task`, `task_embedding`)
keep their OSIE semantics and order. `collate_func` concatenates the two new keys alongside the rest;
`trial_keys` stays a `list[list[str]]` (strings do not tensorise) and `lengths` becomes a
`(B, 3) int32` tensor.

**FR4.7** The fixation-vector construction is **unchanged from OSIE**: `X/resizescale_x`,
`Y/resizescale_y`, `T/1000.0`, `length = fixation["length"]`, assembled into a structured array
`dtype={'names': ('start_x','start_y','duration'), 'formats': ('f8','f8','f8')}`. Coordinates land in
the resized 512×384 space; duration is in **seconds** and `evaluation.py` multiplies by 1000 itself
for ScanMatch (D3, TechStack §4). No rounding, no clipping, no unit change.

---

### FR5 — Per-trial feature loading

**FR5.1** The `torch.load` moves **inside** the subject loop. For each `ids in
self.imgid_to_sub[img_name]`:

```python
exp_key = self.exp_key_map[(img_name, int(fixation["subject"]))]
image_ftrs = torch.load(join(self.feature_dir, exp_key + ".pth")).unsqueeze(0)
```

Built by **concatenation**, never `str.replace('jpg', 'pth')` (FR2.5). `exp_key_filename()` from
`tools/eve_prep/trial_keys.py` is reused so the `[A-Za-z0-9_]+` and no-`jpg` rules are re-asserted at
the call site rather than assumed.

**FR5.2** The loaded tensor must be `torch.float32` of shape `(768, 2048)`. Any other shape or dtype
raises naming the `exp_key`, the expected and the actual (D7).

**FR5.3** A missing `(img_name, subject)` in `exp_key_map` raises a `KeyError` naming both — never a
fallback that invents a key, and never a reuse of a sibling subject's tensor. This is the D4 gate for
Stage D: a mis-mapped trial hands a participant another participant's screen and every metric still
looks plausible.

**FR5.4** Nothing else changes. `images = torch.cat(images)` still yields `(3, 768, 2048)` per image;
`collate_func`'s `torch.stack` still yields `(B, 3, 768, 2048)`; `test.py`'s
`view(-1, *shape[2:])` still yields `(B*3, 768, 2048)`. The per-`(image, subject)` slot already
existed — upstream simply filled it with three copies of one tensor. `imgid_to_sub`, `__len__`, the
uniform 3-subjects-per-image count, the positional diagonal and the retrieval block are untouched, so
the evaluator sees no difference at all.

---

### FR6 — `test.py` adaptations

**FR6.1 — argument defaults.** Changed from OSIE: `--origin_width 1920`, `--origin_height 1080`,
`--im_h 24`, `--im_w 32`, `--subject_num 3`, `--num_fewshot 10`, `--batch 1`,
`--eval_repeat_num 1`. Unchanged: `--width 512`, `--height 384`, `--max_length 16`,
`--min_length 1`, `--action_map_num 4`, `--subject_feature_dim 384`, `--lm_hidden_dim 768`.

**FR6.2 — `origin_size` is passed explicitly.**

```python
EVE_evaluation(args, ..., action_map=(args.im_h, args.im_w),
               origin_size=(args.origin_height, args.origin_width),   # (1080, 1920)
               resize=(args.height, args.width), type="test", ...)
```

OSIE's `test.py` omits this and relies on the `(600, 800)` default while parsing
`--origin_width/--origin_height` into unused variables. Omitting it here mis-scales **every**
coordinate by 2.4× / 1.8× and every metric still returns a number (D3). A test asserts the
constructed dataset's `resizescale_x == 3.75` and `resizescale_y == 2.8125` exactly.

**FR6.3 — new arguments:**

| flag | default | meaning |
|---|---|---|
| `--max_batches` | `-1` | `-1` = no cap. `if args.max_batches > 0 and i_batch >= args.max_batches: break` replaces `if i_batch > 100: break`. |
| `--heatmap_dir` | `""` | path to `gt_heatmaps.h5`; empty disables the FR8 block |
| `--subject_map_path` | `""` | path to `subject_id_map.json`; **required** (FR7) |
| `--bridge_report` / `--senet_report` / `--feature_report` | `""` | the three FR3 handshake reports; all required |
| `--metrics_json` | `"metrics.json"` | filename under the log folder (FR9) |

**FR6.4 — the cap.** With `--max_batches -1` and `--batch 1`, all **354** batches run. The unit is
**batches, hence images** — never "101 images" by coincidence. The resolved value appears in the
logged arg namespace, so the log itself is the evidence that nothing was truncated (D5). A run whose
`i_batch` count at the end is not `len(test_loader)` raises rather than reporting (D7).

**FR6.5 — `--eval_repeat_num` is pinned to 1** and the parser rejects any other value with the
reason: `evaluation.py`'s collectors are shaped `(n_images, subject_num)` and any repeat drives
`row_idx` past `subject_num` into an `IndexError` (TechStack §5).

**FR6.6** Reproducibility lines are unchanged: `np.random.seed`, `torch.manual_seed`,
`torch.cuda.manual_seed_all`, `cudnn.benchmark = False`, `cudnn.deterministic = True` (D5).

**FR6.7** The headline `print('SM: …, MM: …, SED: …')` stays a bare `print()` to stdout, matching
OSIE, and the run script tees stdout into a per-seed artefact (TechStack §3.5a). It is additionally
written into `metrics.json` (FR9), so recovering it never requires parsing formatted text.

---

### FR7 — `prediction.json` and D4 subject recovery

**FR7.1 — the defect this requirement exists for.** Upstream `get_prediction_list()` writes
`args.fewshot_subject[subject_idx]` where `subject_idx` ranges over `range(args.subject_num)`. For
OSIE that is correct by accident: `fewshot_subject = [10,11,12,13,14]` has exactly `subject_num = 5`
entries, so slot *i* really is subject `10+i`, on every image. **For EVE it is wrong.**
`fewshot_subject` has 38 entries and `subject_num` is 3, so slot 0/1/2 would be written as
participant `0`/`1`/`2` for **every one of the 354 images** — while the actual three participants
differ per image. The metrics are unaffected (the evaluator's diagonal is positional), but
`prediction.json` would be unusable for F6 and would answer "which of my real subjects is row 3?"
with a lie. This is D4's live failure on this branch.

**FR7.2** The EVE `test.py` therefore takes the subject id from the **batch**, not from the argument
list. `batch["subjects"]` is `(B, 3)` in `imgid_to_sub` order and is flattened alongside the images,
so slot `idx * subject_num + subject_idx` carries the dense id of the participant actually being
predicted. That dense id is mapped through `subject_id_map.json`'s `to_eve` before it is written.

**FR7.3 — record schema.** `result/<evaluation_dir basename>/log/prediction.json`, a flat list:

```json
{"name": "<stimulus>.jpg", "subject": 12, "subject_eve": "train14",
 "X": [...ints...], "Y": [...ints...], "T": [...ints, ms...]}
```

`X`/`Y` are `int(...)` and `T` is `int(round(t*1000, 3))`, exactly as upstream (TechStack §3.5).
`subject` is the **dense** id, so F6 can key on it directly against `fixations.json`; `subject_eve`
is the real EVE participant string and is the additive part — it is what makes the artefact
self-describing without a second file (D4, D5).

**FR7.4** The file must hold exactly **1062** records = 354 × 3, and the `(name, subject)` key set
must equal the `test` split's key set of `fixations.json` exactly. A mismatch raises, listing up to
10 keys from each side (D7).

**FR7.5** `recover_subject_ids()` is **not** used — it is the `--ex_subject` path, and F5 uses
`--fewshot_subject`. `data/eve_bridge/subject_id_map.json`'s `to_eve` is the sole authority for the
dense → EVE direction (TechStack §3.6). A dense id absent from `to_eve` raises.

---

### FR8 — Heatmap metric block (NSS / CC / KLD)

**FR8.1** When `--heatmap_dir` is set, the run loads `GtHeatmapStore.load(path,
fixations_path=args.fix_dir)` once, before the loop. The `fixations_sha256` gate inside `load()` is
the safety net for its **positional** row addressing (TechStack §3.6).

**FR8.2** Per batch, after `predict = model(...)`:

```python
gt = store.get_batch(flat_trial_keys)              # (N, 16, 24, 32) float32
scores = score_step_heatmaps(all_actions_prob, torch.from_numpy(gt).to(device),
                             lengths=flat_lengths, max_length=args.max_length)
```

`flat_trial_keys` and `flat_lengths` come from FR4.6 and must be flattened in **exactly** the order
the images were — `for image in batch: for subject in image` — so row *k* of `gt` is the ground truth
for row *k* of `all_actions_prob`. A length mismatch between `N` and `len(flat_trial_keys)` raises.

**FR8.3** `score_step_heatmaps()` is imported from `tools/eve_bridge/heatmap_metrics.py` and used
unmodified. It loads `models/loss.py` by path via `importlib`, so the three functions are the
authors' own (TechStack §4.1). Argument order is **prediction first, ground truth second**;
KLD is not symmetric and reversing it yields a plausible wrong number.

**FR8.4 — accumulation.** The per-batch means are weighted by that batch's valid-timestep count
`M_b = Σ_b min(length, 16)` and reduced to one weighted mean over the run, so the reported value
equals the mean over all valid timesteps rather than a mean of batch means. The total `M` is reported
alongside each of NSS/CC/KLD.

**FR8.5 — reported separately, with the denominator stated.** The block never enters the
paper-comparable row, never influences `SM` or `MM`, and every place it appears carries the sentence:
*"NSS/CC/KLD average over valid timesteps (M = …), not over (image, subject) cells (1062) like the
scanpath metrics."* (D6, TechStack §4.1.)

**FR8.6** `metrics.json` carries the caveat that **NSS is ill-conditioned on a near-flat
prediction** — it standardises by `(x - mean)/(std + 1e-7)`, so on a nearly-uniform action map the
result is O(1) noise and an NSS near zero must not be read as "chance level" (TechStack §4.1).

**FR8.7** If `--heatmap_dir` is empty the block is skipped and `metrics.json` records
`"heatmap": null` with `"reason": "--heatmap_dir not set"`. Absence is stated, never implied.

---

### FR9 — `metrics.json`

**FR9.1** Written to the same log folder as `log_test_subject_*.txt`, unconditionally, on every run
including one that produced no predictions (D5).

**FR9.2** Contents:

| key | value |
|---|---|
| `seed`, `args` | `args.seed`, then the full resolved `vars(args)` |
| `versions` | FR1.2's resolved stack, including `cuda` and `gpu_name` |
| `sha256` | `{fixations, user_embedding, checkpoint, embeddings_npy}` |
| `counts` | `{n_images: 354, n_cells: 1062, n_subjects: 38, subject_num: 3, n_batches_run, n_predictions}` |
| `headline` | `{SM, MM, SED}` — the same three numbers the bare `print()` emits |
| `metrics` | `cur_metrics` verbatim: `MultiMatch` (5 dims), `ScanMatch` (w/, w/o), `VAME` (`SED`, `STDE`, `SED_best`, `STDE_best`), and the retrieval block |
| `per_cell_std` | **`null`**, with `"reason": "cur_metrics_std deferred to F6 (TechStack 3.5b)"` |
| `heatmap` | `{NSS, CC, KLD, M, denominator, nss_caveat}` or `null` per FR8.7 |
| `notes` | the FR13 annotations, as strings |

**FR9.3** `SED_best` and `STDE_best` are emitted **only** inside `metrics.VAME` (because that is what
the evaluator returns) and `notes` must state they are **aliases** of `SED`/`STDE` — `evaluation.py`
assigns them outright with no selection step, in every configuration and at any `eval_repeat_num`.
They are never reported as a second, corroborating result (TechStack §4).

**FR9.4** `metrics.json` is a *record*, not a re-score. It reports what the frozen evaluator returned;
it recomputes nothing. F6 is the re-scorer.

---

### FR10 — Seed sweep and per-seed artefacts

**FR10.1** The run is swept over `--seed 0 1 2`. Inference is stochastic (`Sampling.random_sample()`),
so one seed is a point, not a band (TechStack §3.5b).

**FR10.2** Because `log_test_subject_{num_fewshot}_{random_support}.txt` does not vary with the seed,
each invocation copies its outputs into `result/<eval>/log/seed$SEED/` before the next seed overwrites
them: `log_test_subject_*.txt`, `prediction.json`, `metrics.json`, `stdout.txt`, `versions.txt`,
`preflight.json`. Six files per seed.

**FR10.3** `metrics.json`'s `args.seed` must equal the seed of the directory it is copied into. The
copy step asserts it. This is the guard `aggregate_seeds.py` needed after the fact for F1 — a
mis-targeted copy pools non-replicates into a plausible, wrong band.

**FR10.4** F5 does **not** pool the seeds. The across-seed spread is F6/F7's, and the spec states
once more that it is **not** `cur_metrics_std`: across-seed is spread over *runs* (n = 3, a noisy
estimate, quote the range too), per-cell is spread over *(image, subject) cells* within one run. A
write-up labelling one as the other reports a quantity it did not measure.

---

### FR11 — `tools/eve_eval/` preflight

**FR11.1** A new package, CPU-only, Windows-importable, importing **no** frozen code and **no**
`evedataset`. It may import `tools/eve_prep/trial_keys.py` and `tools/eve_bridge/{store,heatmaps}.py`.
It does not import `tools/eve_bridge/heatmap_metrics.py` (the bridge's only torch importer) except in
the FR3.10 parity check, which needs numpy only.

**FR11.2** `check_eval.py` runs FR3.1–FR3.10 and exits `0` on success, `1` on any failure, printing
every failure rather than the first. Its exit code guards the run in `bash/test_eve.sh`.

**FR11.3** It writes `preflight.json`: resolved paths, the three sha256 handshakes, the cohort
counters from FR3.8, `min_support_per_subject`, `origin_size`, the feature coverage count, and the
FR3.10 parity result. This is the per-seed fingerprint F6's pooling guard compares across seeds.

**FR11.4** `--fast` skips only FR3.4's 1062-file hash sweep (existence is still checked), for a
quick re-run inside one allocation. `preflight.json` records `"feature_sha256_checked": false` when
it does — a skipped check is stated, never implied.

---

### FR12 — `bash/test_eve.sh`

**FR12.1** One script, run as `bash bash/test_eve.sh` (never `source` — `set -euo pipefail` would
then apply to the login shell and one failed precondition would drop the allocation). `sbatch` works
unchanged via the inert `#SBATCH` block. Seeds:

```
salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=08:00:00
bash bash/test_eve.sh                 # SEED=0
SEED=1 bash bash/test_eve.sh
SEED=2 bash bash/test_eve.sh
```

**FR12.2** Every path is a tunable read from the environment; no absolute path appears in any `.py`
(working convention 4). Defaults point at `/mnt/beegfs/home/<user>/projects/few-shot-scanpath`;
`/mnt/imagenes` is invisible to compute nodes and `/tmp` is node-local (TechStack §1.3).

**FR12.3** `set -u` is lifted **only** across `conda activate` and restored immediately; `conda.sh` is
sourced by **explicit path**, never `$(conda info --base)`. `-e` and `-o pipefail` are never lifted.

**FR12.4** `--fewshot_subject` is generated as `$(seq 0 37)` — **ascending, by construction**, never
hand-typed. The script asserts the generated list is strictly ascending, starts at 0, and has
`num_subjects` entries before passing it (FR4.4, D4).

**FR12.5** The script aborts if the preflight (FR11.2) exits non-zero. `stdout` is piped through
`tee` into `$SEED_DIR/stdout.txt`; `pipefail` is what makes that safe — without it the tee's status
would mask a failed run.

**FR12.6** Feature staging: `data/eve_features/image_features/` is ~6.7 GB of 1062 small tensors and
is read once per trial. The script stages it to `$LOCAL_SCRATCH` by default (`STAGE_FEATURES=1`) for
the same reason F4 staged the bundle — keeping random small reads off the network filesystem — and
falls back to reading from beegfs when set to `0`. Everything **written** still goes to beegfs.

**FR12.7** The script prints, and `preflight.json` records, the expected runtime shape: 354 batches at
`--batch 1`, 1062 cells, 3 subjects per image, 38 participants.

---

### FR13 — Reporting annotations

These strings must appear in `metrics.json`'s `notes` and in the log:

**FR13.1** *Retrieval is reported separately from the paper-comparable block.* At `subject_num = 3`
every rank lies in `{0,1,2}`, so **R@3 is structurally saturated at 100 %** and R@5 doubly so. Quote
**R@1 and MRR**. Note also that this branch's `p2g()` computes `r3` as `rank < 3`, so the field really
is R@3 here — the `rank < 2` defect is COCO_FV's (TechStack §4.2 row 4).

**FR13.2** *The cohort is 38 participants over 354 scored stimuli, 1062 cells, **3 subjects per image
with identities varying by image***. The evaluator's diagonal is positional, so this is legitimate —
but a reader will assume a fixed trio unless told.

**FR13.3** *Three independent distortions apply and all three reach F7*: the metric squash
1920×1080 → 512×384 (3.75 / 2.8125, OPEN-4); F4's feature squash 1920×1080 → 1024×768 (0.5333 /
0.7111); F3's SE-Net input squash to 512×320 (0.2667 / 0.2963).

**FR13.4** *Carried from `bridge_report.json`*: `short_scanpath` (padded to length 3 inside the frozen
evaluator, and the padded array then replaces the original for **all** subsequent metrics in that
cell), `clamped_coords` (5), `incomplete_stimulus` (602), `surplus_trial` (73).

**FR13.5** *The checkpoint is OSIE-trained.* Both the ISP checkpoint and the SE-Net checkpoint that
produced our embeddings were trained on OSIE. What transfers is the representation, not the subjects
(OPEN-2). And **OPEN-7**: our own OSIE reproduction lands ~1 % on the worse side of the published row
on all three headline metrics, so any comparison of EVE numbers to the paper inherits that offset.

**FR13.6** *The MultiMatch NaN-drop count is surfaced* (D7). `is_eliminating_nan=True` silently drops
NaN MultiMatch rows before the mean and changes the denominator; SED/STDE get no such treatment.
`score_details` is inspected and the dropped count reported per dimension.

---

### FR14 — Error conditions summary

| condition | behaviour |
|---|---|
| any FR3.1 artefact missing | abort, name the path, state that `data/` is git-ignored |
| any sha256 handshake mismatch | abort, print both hashes and both sources |
| `multimatch_gaze != 0.1.3` | abort before inference |
| a scored image with ≠ 3 records | abort, name the image and its count |
| `max(T) <= 20` on the scored split | abort — the §3.7 duration-bin file |
| `--fewshot_subject` not ascending / not `0..37` | abort in the run script, before python |
| `select_fewshot_subject()` changed any subject id | abort in `EVE_evaluation.__init__`, name the first offender |
| `(name, subject)` absent from `exp_key_map` | `KeyError` naming both — never a fallback key |
| `<exp_key>.pth` missing | abort in preflight listing ≤ 10 and the total; never skipped |
| feature tensor not `(768, 2048)` float32 | raise naming the `exp_key`, expected and actual |
| `--eval_repeat_num != 1` | parser rejects with the `IndexError` reason |
| batches run ≠ `len(test_loader)` | abort after the loop |
| `prediction.json` key set ≠ scored split key set | abort listing ≤ 10 from each side |
| dense id absent from `to_eve` | abort naming the id |
| FR3.10 heatmap parity not bitwise | abort |
| heatmap block: `N != len(trial_keys)` | raise from the call site |

An uninitialised evaluator cell is `-1`, **not** NaN — an all-`-1` result means the loop never ran, not
a bad score (TechStack §4).

---

## Public API Summary

```python
# ISP/EVE/GazeformerISP/src/dataset/dataset.py
class EVE_evaluation(torch.utils.data.Dataset):
    def __init__(self, args, stimuli_dir, feature_dir, fixations_dir, task_emb_dir,
                 heatmaps_dir, exp_key_map,
                 action_map=(24, 32), origin_size=(1080, 1920), resize=(384, 512),
                 type="test", transform=None): ...
    def __len__(self) -> int: ...                      # 354 (images, not records)
    def __getitem__(self, idx) -> dict: ...            # + "trial_key" (list[str]), "length" (int32 (3,))
    def collate_func(self, batch) -> dict: ...         # + "trial_keys" list[list[str]], "lengths" (B,3) int32

class EVE(Dataset): ...        # renamed copy, unused on the eval path
class EVE_rl(Dataset): ...     # renamed copy, unused on the eval path


# ISP/EVE/GazeformerISP/src/test.py   (new args on top of OSIE's)
--max_batches -1            # -1 = no cap; replaces `if i_batch > 100: break`
--heatmap_dir      PATH     # gt_heatmaps.h5; "" disables the NSS/CC/KLD block
--subject_map_path PATH     # subject_id_map.json  (required)
--bridge_report    PATH     # bridge_report.json   (required)
--senet_report     PATH     # senet_report.json    (required)
--feature_report   PATH     # feature_report.json  (required)
--metrics_json     NAME     # default "metrics.json"


# tools/eve_eval/check_eval.py           (CPU, Windows-importable, stdlib + numpy + h5py)
def check_eval(bridge_dir, senet_dir, feature_dir, weights_dir, *,
               subject_num=3, num_fewshot=10, fast=False) -> dict      # -> preflight.json
class EveEvalError(Exception): ...
# CLI: python tools/eve_eval/check_eval.py --bridge-dir ... --senet-dir ...
#          --feature-dir ... --weights-dir ... [--fast] [--out preflight.json]
#      exit 0 = every check passed; 1 = at least one failed (all are printed)


# tools/eve_eval/parity.py               (FR3.10, numpy only)
def check_heatmap_parity(n_cases=200, seed=0) -> dict                  # bitwise or raise


# bash/test_eve.sh
#   SEED=0|1|2   HOME_DIR  PROJECT_DIR  ISP_ENV  BRIDGE_DIR  SENET_DIR  FEATURE_DIR
#   WEIGHTS_DIR  LOCAL_SCRATCH  STAGE_FEATURES  MAX_BATCHES  SUBJECT_NUM  NUM_FEWSHOT
```

---

## Dependencies

| direction | artefact / module | contract |
|---|---|---|
| reads | `data/eve_bridge/fixations.json` | §3.1 schema; sha `46c6926f…`; 1062 scored records, 354 names, 38 subjects |
| reads | `data/eve_bridge/gt_heatmaps.h5` | §3.6 layout; positional rows; `fixations_sha256`-gated; `trials/{trial_key,exp_key,length}` + `trials/heatmaps` |
| reads | `data/eve_bridge/subject_id_map.json` | `to_eve` is the sole dense → EVE authority (D4) |
| reads | `data/eve_bridge/bridge_report.json` | `fixations_sha256`, `origin_size`, `min_support_per_subject`, the D7 counters |
| reads | `data/eve_senet/seed0/eve_fewshot_user_embedding_10_seed0.pt` | `(38, 384)` float32, row *i* = dense subject *i*, sha `e904a165…` |
| reads | `data/eve_senet/seed0/senet_report.json` | `embedding_sha256`, `fixations_sha256`, `seed` |
| reads | `data/eve_features/image_features/<exp_key>.pth` | `(768, 2048)` float32, **one per trial**, 1062 files ≈ 6.7 GB |
| reads | `data/eve_features/embeddings.npy` | dict with `"free-viewing"` → `(768,)` float32; byte-identical to OSIE's |
| reads | `data/eve_features/feature_report.json` | `fixations_sha256`, `feature_sha256` per tensor |
| reads | `weights/.../checkpoints/checkpoint_best.pth` | `{"model": OrderedDict}`; released ISP checkpoint |
| imports | `tools/eve_prep/trial_keys.py` | `load_trial_exp_keys`, `exp_key_filename` — torch-free by design |
| imports | `tools/eve_bridge/store.py` | `GtHeatmapStore.load` / `.get_batch` |
| imports | `tools/eve_bridge/heatmap_metrics.py` | `score_step_heatmaps` — the one bridge module F5 imports |
| copies | `ISP/OSIE/GazeformerISP/src/**` | verbatim into `ISP/EVE/GazeformerISP/src/`; the three frozen files sha-asserted (D1) |
| writes | `result/EVE-<tag>/log/seed{0,1,2}/` | `log_test_subject_*.txt`, `prediction.json`, `metrics.json`, `stdout.txt`, `versions.txt`, `preflight.json` |
| consumed by | **F6** | `prediction.json` + `fixations.json` → offline re-score, and the per-cell std F5 defers |
| consumed by | **F7** | `metrics.json` × 3 seeds, the FR13 annotations, the D7 counters |
