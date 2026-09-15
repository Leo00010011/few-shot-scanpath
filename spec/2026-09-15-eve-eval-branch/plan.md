# F5 — Implementation plan

> Spec folder 2 of 3: [requirements.md](requirements.md) · **plan** · [validation.md](validation.md)

---

## Context and Design Decisions

### Why a new branch rather than an argument on the OSIE one

Working convention 2 prefers additive over invasive, and convention 3 says a new dataset branch
mirrors the OSIE tree exactly. The deciding factor is F1: its baseline is the project's environment
proof and its numbers are quoted in every later comparison, so `ISP/OSIE/GazeformerISP/` must stay
reproducible bit-for-bit. Any argument-gated change to OSIE's `test.py` or `dataset.py` — however
carefully defaulted — makes "F1 is unchanged" a claim requiring proof instead of a fact. A separate
tree makes it a fact.

The copy is **full and verbatim** (decided 2026-09-15). The alternative — copy only the three frozen
files that D1 requires and reach `models/`, `sampling.py` and `logger.py` through `PYTHONPATH` into
the OSIE branch — saves duplication at the price of a cross-branch import surface, where a later edit
to OSIE's `models/` silently changes EVE's results. The duplication is the cheaper failure.

### Why the frozen files are copied and then hash-asserted

D1 freezes `utils/evaluation.py` and `utils/evaltools/*`. Copying them into a new branch is exactly
what created the COCO_FV drift documented in TechStack §4.2 — six real divergences in one file,
discovered long after the fact. The defence is not discipline, it is a test: FR2.2 asserts sha256
equality against the OSIE originals, so a drift is a red test rather than a quiet six-month-old
divergence. The same test covers `models/`, `logger.py`, `data_postprocess.py` and `preprocess/`, and
asserts that `dataset.py` and `test.py` are the *only* two files that differ.

### Why the feature load moves inside the subject loop

This is OPEN-6's resolution reaching Stage D. EVE presents each photograph at a per-trial display
scale, so one tensor per `stimulus_name` cannot represent what every participant saw; F4 therefore
wrote one `(768, 2048)` tensor per `(stimulus, participant)` trial, keyed by `exp_key`, and measured
the difference (within-name cosine median **0.775**, against a cross-stimulus **0.450** — the tensors
are genuinely different, not nearly-duplicates).

Structurally this changes nothing the evaluator sees. `OSIE_evaluation.__getitem__` already builds a
per-`(image, subject)` feature slot and simply fills it with three references to one tensor;
`collate_func`'s `torch.stack` and `test.py`'s `view(-1, *shape[2:])` already flatten it to
`(B*3, 768, 2048)`. Moving the `torch.load` inside the loop fills those three slots with three
different tensors. `imgid_to_sub`, `__len__` over images, the uniform 3-subjects-per-image count, the
positional diagonal and the retrieval block are all untouched.

The path is built by **concatenation**, `join(feature_dir, exp_key + ".pth")`, never the upstream
`img_name.replace('jpg', 'pth')` — that replace is unanchored and would corrupt any key containing
the substring. FR2.5's AST test enforces it rather than trusting the author.

### Why `origin_size` is passed explicitly, and why the scale is ugly

`OSIE_evaluation`'s default is `origin_size=(600, 800)`, and OSIE's own `test.py` does not override
it — `args.origin_width` / `args.origin_height` are parsed and unused. On EVE that default mis-scales
every coordinate by 2.4× horizontally and 1.8× vertically, and every metric still returns a plausible
number. D3 exists for this. The correct value, `(1080, 1920)`, is in `bridge_report.json` and in the
HDF5 attrs precisely so it can be asserted rather than remembered.

The resulting `3.75 / 2.8125` is **non-uniform** — a deliberate squash of 16:9 into the 512×384 metric
screen (OPEN-4), taken so every ScanMatch and MultiMatch parameter stays bit-identical to the
published configuration. We traded stimulus aspect ratio for metric comparability; F7 must say which
way that trade went, and must state all three squashes (FR13.3).

### Why `--max_batches` rather than deleting the cap

`if i_batch > 100: break` is inert for OSIE (70 batches) and **live** for EVE (354). Deleting the two
lines is safe but leaves nothing in the log positively asserting the full split ran, and no
smoke-test knob for a five-image dry run on a busy cluster. Parameterising with `-1 = no cap` puts
the value into the resolved arg namespace the logger already dumps (D5), so the log is the evidence.
FR6.4 adds the belt: a run whose final `i_batch` count is not `len(test_loader)` aborts.

### Why `prediction.json`'s subject id is taken from the batch

`get_prediction_list()` writes `args.fewshot_subject[subject_idx]`. On OSIE that is correct because
`len(fewshot_subject) == subject_num == 5`, so slot *i* is always subject `10+i`. On EVE
`fewshot_subject` has 38 entries and `subject_num` is 3, so it would stamp participant 0/1/2 onto
every one of the 354 images while the actual trio varies per image. The metrics are unharmed — the
evaluator's diagonal is positional — but the artefact would be a lie, F6 would re-score against the
wrong ground truth, and D4's question "which of my real subjects is row 3?" would get a confident
wrong answer.

`batch["subjects"]` already carries the right dense ids in the right positional order, and is already
flattened alongside the images. Taking the id from there and mapping it through
`subject_id_map.json`'s `to_eve` is a call-site fix (convention 2); `data_postprocess.py` stays a
byte-identical copy and is simply not used on this path.

### Why the ascending `--fewshot_subject` is generated, not typed

`select_fewshot_subject()` remaps the argument list to a dense `0..N-1` and the model indexes
`subject_embed[subjects]` with the result. Only ascending `0 1 2 … 37` yields the identity remap —
and the identity is what keeps `exp_key_map`'s `(name, subject)` key addressing the right trial and
row *i* of F3's tensor on participant *i*. Any other order silently permutes the whole cohort and
every metric still looks plausible. So: `$(seq 0 37)` in the script, an ascending-and-complete
assertion before it is passed, and an independent assertion inside `EVE_evaluation.__init__` that no
record's subject id actually changed. Three layers, because this one is unrecoverable after the fact.

### What is cached, what is computed, what is deferred

`exp_key_map` is built **once at construction** from `gt_heatmaps.h5` via
`load_trial_exp_keys()` — torch-free by design, reading `trials/{trial_key,exp_key}` only and never
the ~88 MB `trials/heatmaps`. The heatmap store itself is loaded once, in full, only when
`--heatmap_dir` is set; `(1804, 16, 24, 32)` float32 is ~88 MB resident, which is cheap next to the
model. Feature tensors are **not** cached — 1062 × 6.29 MB is 6.7 GB and each is read exactly once,
which is why FR12.6 stages them to node-local scratch instead.

Deferred on purpose: the per-cell `cur_metrics_std` (F6 recomputes it from `prediction.json` on CPU),
the pooling of the three seeds into a run record (F6/F7), and the write-up (F7).

### Constitution constraints this plan is written against

**D1** the three frozen files are copied and hash-asserted, never edited. **D2** F5 consumes only
canonical artefacts; no `evedataset`, no bundle. **D3** `origin_size` explicit and asserted;
structured-array dtype and second/millisecond contracts unchanged. **D4** three layers on the
ascending cohort, plus `subject_eve` in `prediction.json`. **D5** resolved namespace, `versions.txt`,
`metrics.json` and a re-scorable `prediction.json` per seed. **D6** the headline block reported, with
retrieval separated and annotated, and the std's provenance stated rather than implied. **D7** every
condition in FR14 raises or is counted; nothing is dropped into a mean. **D8** eval-only.

---

## Implementation Steps

### Step 1 — Create the branch tree

**Create** `ISP/EVE/GazeformerISP/` by copying `ISP/OSIE/GazeformerISP/src/` verbatim.

```
cp -r ISP/OSIE/GazeformerISP/src ISP/EVE/GazeformerISP/src
rm -rf ISP/EVE/GazeformerISP/src/**/__pycache__     # do NOT copy .pyc (convention 6)
```

Keep `src/data/embeddings.npy` and `src/data/fixations.json` in place as the OSIE reference copies —
the run points `--emb_dir` and `--fix_dir` at the EVE artefacts, so these are never read. Do not
delete them; they are what FR3.7's byte-identity check compares against without reaching across
branches at runtime.

No edits in this step. The tree must be a pure copy so Step 6's hash test has a meaningful baseline.

---

### Step 2 — `tools/eve_eval/__init__.py` and `parity.py`

**Create** `tools/eve_eval/__init__.py`:

```python
class EveEvalError(Exception):
    """Any F5 precondition or contract violation."""

SCORED_CELLS = 1062
SCORED_IMAGES = 354
N_SUBJECTS = 38
SUBJECTS_PER_IMAGE = 3
ORIGIN_SIZE = (1080, 1920)     # (H, W)
ACTION_MAP = (24, 32)
RESIZE = (384, 512)
FEATURE_SHAPE = (768, 2048)
EMBEDDING_SHAPE = (38, 384)
MAX_LENGTH = 16
```

**Create** `tools/eve_eval/parity.py` — FR3.10, numpy + scipy only, no torch:

```python
def check_heatmap_parity(n_cases=200, seed=0):
    """Re-run F2's bitwise parity check in THIS env's numpy (TechStack section 1).

    For each seeded random (length, coords) case, build the step heatmap two ways:
      a) tools.eve_bridge.heatmaps.build_step_heatmaps(...)
      b) a local transcription of OSIE.__getitem__'s construction:
           pos_discrete = ((pos - 1) / downscale).astype(np.int32)   # TRUNCATES
           scanpath[t, y, x] = 1
           scanpath[t] = gaussian_filter(scanpath[t], sigma=1)
           scanpath[t] /= scanpath[t].sum()
    Require np.array_equal (bitwise), not allclose.
    """
```

Return `{"n_cases": n, "bitwise_equal": True, "numpy": np.__version__,
"scipy": scipy.__version__}`; raise `EveEvalError` naming the first differing case, its length, its
coordinates and `np.abs(a - b).max()` otherwise.

> The truncating `.astype(np.int32)` is load bearing and is not rounding. Transcribe it exactly.

---

### Step 3 — `tools/eve_eval/check_eval.py`

**Create** the preflight. Stdlib + numpy + h5py; imports `tools/eve_prep/trial_keys.py` and
`tools/eve_eval/parity.py`. **No torch** except behind an optional flag, because FR3.4 needs only
file bytes, not tensor loads.

```python
def check_eval(bridge_dir, senet_dir, feature_dir, weights_dir, *,
               subject_num=3, num_fewshot=10, fast=False):
    failures = []          # collect ALL, print ALL, then exit 1 -- never first-failure-only
    report = {}

    # FR3.1 existence
    # FR3.2 sha256(fixations.json) == bridge/senet/feature report fixations_sha256   (3-way)
    # FR3.3 sha256(user_emb) == senet_report["embedding_sha256"]; seed in name == report seed
    # FR3.5 num_fewshot <= bridge_report["min_support_per_subject"]      (10 <= 10)
    # FR3.6 bridge origin_size == [1080,1920]; h5 attrs origin_size/action_map/max_length
    # FR3.7 embeddings.npy loads, has "free-viewing" (768,) f4, byte-identical to OSIE's
    # FR3.8 cohort invariants over the "test" split (below)
    # FR3.9 every scored (name, subject) -> exp_key -> <exp_key>.pth exists
    # FR3.4 sha256 of every scored .pth == feature_report["feature_sha256"][exp_key]  (skip if fast)
    # FR3.10 parity.check_heatmap_parity()
    return report
```

FR3.8's cohort block, written out because it is the one that catches a regenerated bridge:

```python
test = [r for r in fixations if r["split"] == "test"]
by_name = defaultdict(list)
for r in test: by_name[r["name"]].append(r)

ragged = {n: len(v) for n, v in by_name.items() if len(v) != subject_num}
if ragged: fail("ragged scored images (FR3.8): %s" % listing(ragged))
if len(test) != SCORED_CELLS: fail(...)
if len(by_name) != SCORED_IMAGES: fail(...)

subs = sorted({r["subject"] for r in test})
if subs != list(range(N_SUBJECTS)): fail("dense subject ids are not 0..37 (FR3.8)")

max_T = max(max(r["T"]) for r in test)
if max_T <= 20:
    fail("max(T) = %d over the scored split: this looks like the DECILE-BIN file "
         "(TechStack 3.7), not a duration file. SM and MM would be silently wrong." % max_T)

bad = [(r["name"], r["subject"]) for r in test
       if r["condition"] != "freeview" or r["task"] != "none"]
if bad: fail("condition/task constants violated (FR3.8): %s" % listing(bad))
```

CLI: `--bridge-dir --senet-dir --feature-dir --weights-dir [--subject-num 3] [--num-fewshot 10]
[--fast] [--out preflight.json]`. Prints every failure, writes `preflight.json` **either way** (with
`"ok": false` and the failure list when it fails — an absent report is worse than a failing one), and
exits `0`/`1`.

---

### Step 4 — `EVE_evaluation` in `ISP/EVE/GazeformerISP/src/dataset/dataset.py`

**Modify** the copied file. Rename the three classes (`OSIE` → `EVE`, `OSIE_rl` → `EVE_rl`,
`OSIE_evaluation` → `EVE_evaluation`) and change **only** `EVE_evaluation`.

**4a — `__init__`.** New parameters `heatmaps_dir` and `exp_key_map`; changed defaults per FR4.1.
After the existing `select_fewshot_subject()` call, insert the D4 identity assertion — capture the
subject ids *before* the call, since the function mutates the dicts in place:

```python
# BEFORE the ex_subject / fewshot_subject branch:
_pre = [int(r["subject"]) for r in fixations]
...
# AFTER it:
if len(_pre) != len(fixations):
    raise ValueError("select_fewshot_subject() changed the record count ...")
for i, (before, rec) in enumerate(zip(_pre, fixations)):
    if before != int(rec["subject"]):
        raise ValueError(
            "D4: select_fewshot_subject() remapped subject {} -> {} at record {} "
            "({}). --fewshot_subject must be ascending 0..N-1; any other order "
            "silently permutes the cohort's embeddings and every metric still looks "
            "plausible.".format(before, rec["subject"], i, rec["name"]))
```

Then store `self.exp_key_map = exp_key_map` and `self.heatmaps_dir = heatmaps_dir`. `stimuli_dir` is
stored and never used (FR4.2).

**4b — `__getitem__`.** Delete the two lines above the subject loop:

```python
# REMOVE:
# img_path = join(self.feature_dir, img_name.replace('jpg', 'pth'))
# image_ftrs = torch.load(img_path).unsqueeze(0)
```

and inside `for ids in self.imgid_to_sub[img_name]:`, after `fixation = self.fixations[ids]`:

```python
subject = int(fixation["subject"])
try:
    exp_key = self.exp_key_map[(img_name, subject)]
except KeyError:
    raise KeyError(
        "no exp_key for trial ({!r}, {}) -- gt_heatmaps.h5 and fixations.json "
        "disagree (FR5.3)".format(img_name, subject))
image_ftrs = torch.load(join(self.feature_dir, exp_key_filename(exp_key)))
if image_ftrs.shape != (768, 2048) or image_ftrs.dtype != torch.float32:
    raise ValueError(
        "feature tensor for exp_key {} is {} {}; expected (768, 2048) "
        "torch.float32 (FR5.2)".format(exp_key, tuple(image_ftrs.shape),
                                       image_ftrs.dtype))
image_ftrs = image_ftrs.unsqueeze(0)
```

`exp_key_filename` is imported from `tools.eve_prep.trial_keys` at module top, inside a try/except
that raises a clear message if `tools/` is not on `PYTHONPATH` (the run script puts it there).

Accumulate the two new per-subject lists and return them (FR4.6):

```python
trial_keys.append("{}|{}".format(img_name, subject))
lengths.append(min(int(fixation["length"]), self.max_length))
...
"trial_key": trial_keys,
"length": np.array(lengths, dtype=np.int32),
```

`EVE_evaluation` has no `max_length` attribute upstream — add it as a constructor parameter
defaulting to 16, matching `EVE`.

**4c — `collate_func`.** Append the two keys; `trial_keys` stays a plain nested list (the generic
`np.ndarray → tensor` conversion at the end of the function only touches ndarrays, so a list passes
through untouched):

```python
data["trial_keys"] = trial_key_batch          # list[list[str]], (B, 3)
data["lengths"] = np.stack(length_batch)      # (B, 3) int32 -> tensor by the existing conversion
```

---

### Step 5 — `ISP/EVE/GazeformerISP/src/test.py`

**Modify** the copied file.

**5a — arguments.** Change the FR6.1 defaults; add the FR6.3 flags. Reject `--eval_repeat_num != 1`
immediately after `parse_args()` with the `IndexError` reason (FR6.5).

**5b — preconditions, before the model is built.** Import and run the preflight in-process so a
direct `python src/test.py` invocation is as safe as the script path:

```python
sys.path.insert(0, join(PROJECT_ROOT, "tools"))
from eve_eval.check_eval import check_eval, EveEvalError
preflight = check_eval(args.bridge_report, args.senet_report, args.feature_report,
                       args.user_emb_path, args.fix_dir, args.emb_dir,
                       args.feat_dir, args.evaluation_dir,
                       subject_num=args.subject_num, num_fewshot=args.num_fewshot,
                       fast=True)      # the script already ran the full sweep
```

`fast=True` here because `bash/test_eve.sh` runs the full FR3.4 hash sweep once per allocation; the
in-process call re-checks everything cheap.

**5c — build `exp_key_map` and the dataset:**

```python
from eve_prep.trial_keys import load_trial_exp_keys
exp_key_map = load_trial_exp_keys(args.heatmap_dir, fixations_path=args.fix_dir)

test_dataset = EVE_evaluation(
    args, args.img_dir, args.feat_dir, args.fix_dir, args.emb_dir,
    heatmaps_dir=args.heatmap_dir, exp_key_map=exp_key_map,
    action_map=(args.im_h, args.im_w),
    origin_size=(args.origin_height, args.origin_width),      # FR6.2 -- explicit
    resize=(args.height, args.width), type="test", transform=transform)
assert test_dataset.resizescale_x == 3.75 and test_dataset.resizescale_y == 2.8125
```

**5d — the heatmap store**, loaded once when `--heatmap_dir` is set:

```python
from eve_bridge.store import GtHeatmapStore
store = GtHeatmapStore.load(args.heatmap_dir, fixations_path=args.fix_dir) \
        if args.heatmap_dir else None
from eve_bridge.heatmap_metrics import score_step_heatmaps
```

**5e — the loop.** Replace the cap, flatten the new keys, and accumulate the heatmap block:

```python
n_batches = 0
hm_sum = {"NSS": 0.0, "CC": 0.0, "KLD": 0.0}
hm_M = 0
for i_batch, batch in enumerate(test_loader):
    if args.max_batches > 0 and i_batch >= args.max_batches:
        break
    n_batches += 1
    ...
    # flatten in EXACTLY the image-then-subject order the tensors were flattened in
    flat_keys = [k for per_image in batch["trial_keys"] for k in per_image]
    flat_len  = batch["lengths"].reshape(-1).tolist()
    ...
    if store is not None:
        gt = torch.from_numpy(store.get_batch(flat_keys)).to(all_actions_prob.device)
        if gt.shape[0] != all_actions_prob.shape[0]:
            raise ValueError("heatmap batch misalignment: %d gt rows vs %d predictions"
                             % (gt.shape[0], all_actions_prob.shape[0]))
        s = score_step_heatmaps(all_actions_prob, gt, flat_len,
                                max_length=args.max_length)
        m = sum(min(int(v), args.max_length) for v in flat_len)   # valid timesteps
        for k in hm_sum: hm_sum[k] += s[k] * m                    # weighted (FR8.4)
        hm_M += m
```

and the D4-correct prediction record, replacing the `get_prediction_list` call:

```python
flat_subjects = batch["subjects"].reshape(-1).tolist()   # dense ids, positional
for idx in range(len(batch["img_names"])):
    for subject_idx in range(args.subject_num):
        slot = idx * args.subject_num + subject_idx
        dense = int(flat_subjects[slot])
        pred = sampling_random_predict_fix_vectors[slot]
        predict_results.append({
            "name": batch["img_names"][idx],
            "subject": dense,
            "subject_eve": to_eve[str(dense)],          # FR7.2; KeyError if absent
            "X": [int(p[0]) for p in pred],
            "Y": [int(p[1]) for p in pred],
            "T": [int(round(p[2] * 1000, 3)) for p in pred],
        })
```

After the loop (FR6.4):

```python
if args.max_batches <= 0 and n_batches != len(test_loader):
    raise RuntimeError("ran %d batches of %d -- the split was truncated (FR6.4)"
                       % (n_batches, len(test_loader)))
```

**5f — after `comprehensive_evaluation_by_subject()`.** Keep the existing logger loop over
`cur_metrics` unchanged. Then verify `prediction.json` against the scored key set (FR7.4), write it,
and write `metrics.json` (FR9) with `per_cell_std: null`, the FR13 `notes`, and the heatmap block:

```python
heatmap = None if store is None else {
    "NSS": hm_sum["NSS"] / hm_M, "CC": hm_sum["CC"] / hm_M,
    "KLD": hm_sum["KLD"] / hm_M, "M": hm_M,
    "denominator": "valid timesteps (M), NOT (image, subject) cells (1062) -- "
                   "the scanpath metrics use a different denominator",
    "nss_caveat": "NSS standardises by (x-mean)/(std+1e-7); on a near-flat action "
                  "map it is O(1) noise. An NSS near zero is not 'chance level'.",
}
```

`cur_metrics_std` is received and **not** logged — FR9.2 records `per_cell_std: null` with the
deferral reason instead, so a reader cannot mistake absence for zero.

**5g** Leave the headline `print()` exactly as it is (FR6.7); the script tees stdout.

---

### Step 6 — `tests/eve_eval/`

**Create** the CPU pytest suite. Fixtures are synthetic — a 6-image / 3-subject miniature
`fixations.json`, a matching `gt_heatmaps.h5` built through `GtHeatmapStore.build`, 18 tiny
`(768, 2048)` tensors, and a `(38, 384)` embedding — so nothing needs the cluster or the real 6.7 GB
cache. Real-artefact tests are marked `bridge` and skipped when `data/eve_bridge/` is absent, exactly
as `tests/eve_prep/` does.

The load-bearing ones:

- **`test_frozen_files_byte_identical`** — sha256 of the three D1 files in both branches, plus the
  assertion that `{dataset/dataset.py, test.py}` is *exactly* the set of files differing between the
  two `src/` trees (FR2.2, FR2.3).
- **`test_no_replace_in_getitem`** — walk `EVE_evaluation.__getitem__`'s AST for any
  `Call(func=Attribute(attr="replace"))` (FR2.5).
- **`test_origin_size_scales`** — constructed dataset has `resizescale_x == 3.75`,
  `resizescale_y == 2.8125` exactly (FR6.2).
- **`test_per_trial_features_differ`** — two subjects on one image receive **different** tensors;
  assert `not torch.equal(...)` (FR5.1).
- **`test_descending_fewshot_raises`** — a descending `--fewshot_subject` raises in
  `EVE_evaluation.__init__` naming the first offender (FR4.4). This is the D4 trap's test.
- **`test_prediction_subject_ids`** — over a synthetic run where image *k*'s trio is *k*, *k+1*,
  *k+2*, every record's `subject` matches the ground truth's, and `subject_eve` matches `to_eve`
  (FR7.2, FR7.4). Also assert the *upstream* formula would have failed here, so the test documents
  the defect rather than only guarding the fix.

---

### Step 7 — `bash/test_eve.sh`

**Create**, modelled on `bash/test_osie.sh` and `bash/extract_eve_features.sh`.

Order: tunables → `#SBATCH` block → preconditions on beegfs → optional env-image mount (non-fatal) →
optional feature staging to `$LOCAL_SCRATCH` → `set +u` / `source conda.sh` by explicit path /
`conda activate "$ISP_ENV"` / `set -u` → `versions.txt` → preflight (exit code is the gate) → the
cohort argument, generated and asserted → `python src/test.py … | tee` → per-seed artefact copy with
the FR10.3 seed assertion.

The cohort argument (FR12.4):

```bash
N_SUBJECTS="${N_SUBJECTS:-38}"
FEWSHOT_SUBJECTS="$(seq 0 $((N_SUBJECTS - 1)) | tr '\n' ' ')"
# generated, never typed: select_fewshot_subject() remaps this list to a dense
# 0..N-1 range, and ONLY the ascending order gives the identity remap that keeps
# row i on participant i. Any other order silently permutes the whole cohort.
python - "$FEWSHOT_SUBJECTS" "$N_SUBJECTS" <<'PY'
import sys
xs = [int(v) for v in sys.argv[1].split()]
assert xs == list(range(int(sys.argv[2]))), "fewshot_subject is not ascending 0..N-1 (D4)"
PY
```

The invocation:

```bash
CUDA_VISIBLE_DEVICES=0 python src/test.py \
  --fewshot_subject $FEWSHOT_SUBJECTS \
  --subject_num "$SUBJECT_NUM" --num_fewshot "$NUM_FEWSHOT" \
  --seed "$SEED" --eval_repeat_num 1 --batch 1 --max_batches "$MAX_BATCHES" \
  --width 512 --height 384 --origin_width 1920 --origin_height 1080 \
  --im_h 24 --im_w 32 \
  --fix_dir "$BRIDGE_DIR/fixations.json" \
  --feat_dir "$FEATURE_SRC/image_features" \
  --emb_dir  "$FEATURE_DIR/embeddings.npy" \
  --heatmap_dir "$BRIDGE_DIR/gt_heatmaps.h5" \
  --subject_map_path "$BRIDGE_DIR/subject_id_map.json" \
  --bridge_report "$BRIDGE_DIR/bridge_report.json" \
  --senet_report  "$SENET_DIR/senet_report.json" \
  --feature_report "$FEATURE_DIR/feature_report.json" \
  --user_emb_path "$SENET_DIR/eve_fewshot_user_embedding_10_seed0.pt" \
  --evaluation_dir "$EVAL_DIR" \
  2>&1 | tee "$SEED_DIR/stdout.txt"
```

Note `--feat_dir` points at `$FEATURE_SRC` (the staged copy) while `--feature_report` and
`--emb_dir` point at beegfs — the report travels with the record, the 6.7 GB does not.

`PYTHONPATH` must carry both `$PROJECT_DIR/tools` and the branch's `src`, because `test.py` imports
`eve_eval`, `eve_prep` and `eve_bridge`.

The per-seed copy, with the FR10.3 guard:

```bash
SEED_DIR="$RESULT_LOG/seed$SEED"; mkdir -p "$SEED_DIR"
cp "$RESULT_LOG"/log_test_subject_*.txt "$RESULT_LOG/prediction.json" \
   "$RESULT_LOG/metrics.json" "$SEED_DIR/"
cp "$WORK/versions.txt" "$WORK/preflight.json" "$SEED_DIR/"
python -c "import json,sys; m=json.load(open(sys.argv[1])); \
  assert m['args']['seed']==int(sys.argv[2]), 'seed mismatch: metrics.json says %s, \
  directory says %s (FR10.3)'%(m['args']['seed'],sys.argv[2])" \
  "$SEED_DIR/metrics.json" "$SEED"
```

Closing comment: run all three seeds, then the pooling belongs to F6/F7 — and the resulting spread is
the **across-seed** one, not `cur_metrics_std`.

---

### Step 8 — `.gitignore` check and spec housekeeping

Run `git status --ignored` and confirm that nothing F5 authors is swallowed by a broad rule. Two
precedents: `spec/**/*.json` was ignored until 2026-09-09 (F1's `paper_reference.json` silently
vanished) and `SE-Net/configs/*.json` until 2026-09-10 (F3's `eve_useremb.json`, same failure). F5
adds no new JSON *source*, but `ISP/EVE/GazeformerISP/src/data/*.npy|*.json` are copies of ignored
upstream files — confirm they are ignored deliberately and that the branch still works from a fresh
checkout plus hand-shipped `data/`.

---

## Implementation Order

1. **Step 1** — copy the OSIE `src/` tree verbatim into `ISP/EVE/GazeformerISP/`.
2. **Step 2** — `tools/eve_eval/{__init__,parity}.py`.
3. **Step 3** — `tools/eve_eval/check_eval.py` and its CLI.
4. **Step 4** — `EVE_evaluation`: renames, the D4 identity assertion, the per-trial feature load, the
   two new sample keys.
5. **Step 5** — `test.py`: arguments, preconditions, explicit `origin_size`, `--max_batches`, the D4
   prediction records, the heatmap block, `metrics.json`.
6. **Step 6** — `tests/eve_eval/` (must be green on Windows CPU before anything reaches the cluster).
7. **Step 7** — `bash/test_eve.sh`.
8. **Step 8** — `git status --ignored` sweep.
9. Cluster: ship `data/` by hand, run the preflight alone, then seed 0, then seeds 1 and 2.
10. Run [validation.md](validation.md)'s Data Validity and Data Architecture Integrity blocks against
    the real artefacts; record anything surprising in a `notes.md`.
