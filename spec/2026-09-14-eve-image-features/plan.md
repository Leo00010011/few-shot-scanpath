# F4 — Implementation plan

> Spec folder 2 of 3: [requirements.md](requirements.md) · **plan** · [validation.md](validation.md)

---

## Context and Design Decisions

### Why per-trial keying, and why it is free

The decision that shapes everything else is FR11.2: **the per-trial display scale is EVE's deliberate
augmentation and must be preserved.** That forces per-trial features, because one image per
`stimulus_name` cannot represent two display scales.

The reason this is affordable is a detail of the upstream loader that is easy to miss.
[`dataset.py:450-491`](../../ISP/OSIE/GazeformerISP/src/dataset/dataset.py#L450-L491) loads **one**
feature tensor per image name and then appends *that same tensor* once per subject:

```python
img_path   = join(self.feature_dir, img_name.replace('jpg', 'pth'))
image_ftrs = torch.load(img_path).unsqueeze(0)        # ONE load
...
for ids in self.imgid_to_sub[img_name]:
    ...
    images.append(image_ftrs)                          # appended N times
images = torch.cat(images)                             # (N, 768, 2048)
```

The per-`(image, subject)` slot already exists in the batch; upstream just fills it with duplicates.
Filling it with distinct tensors is a two-line change in F5's *adapted* `dataset.py` and is invisible
to `evaluation.py`, which receives fix vectors and never sees a path. So:

- `imgid_to_sub` grouping, `__len__` over images — unchanged.
- The uniform 3-subjects-per-scored-image count that F2 built the cohort around — unchanged.
- The positional diagonal (Mission P4, OPEN-5) — unchanged.
- The retrieval block, which is the only thing needing several subjects on one image — unchanged.
- **D1 is not engaged at all.** The frozen set is `utils/evaluation.py` and `utils/evaltools/*`.
  `dataset/dataset.py` is an adapted copy that the Roadmap already assigns to F5.

This is why FR11.3 can say OPEN-6 is resolved *by elimination* rather than by compromise: with one
`.pth` per trial there is no shared slot for two renderings to contend for.

### Why we read the bundle instead of re-running the bridge

This is a declared D2 deviation (FR12) and it was not the default choice. The bridge's
`export_stimuli()` writes one `.jpg` per `stimulus_name` and *counts* the collision
([`convert.py:278-302`](../../tools/eve_bridge/convert.py#L278-L302)); making it per-trial means
re-running F2, re-encoding 1804 PNGs to JPEG (a lossy step on data we want bit-exact), and reopening a
closed feature. Reading `bundle.get_stimulus(exp_key)` gives the presented screen capture with no
re-encode, and the deviation is contained: it lives only in `tools/eve_prep/`, F5 never touches the
bundle, and **FR2.4's cross-check re-anchors every bundle read to a bridge artefact** so the deviation
cannot silently drift from F2's index space.

### Why `gt_heatmaps.h5` is the authority for exp_key

`GtHeatmapStore` already persists `trials/exp_key` aligned to `trials/trial_key`
([`store.py:130-137`](../../tools/eve_bridge/store.py#L130-L137)), which is exactly the
`(name, subject) → exp_key` map F4 needs — written by the same code path that wrote
`fixations.json`, and hash-bound to it via the `fixations_sha256` attr. Re-deriving it from the
bundle alone would silently diverge if F2's selection logic (surplus-trial drop, support assignment)
ever changed. So the store is the authority and the bundle derivation is the **check**, not the
source. `GtHeatmapStore.load()` is deliberately *not* used: it materialises the full
`(1804, 16, 24, 32)` heatmap array (~88 MB) that F4 has no use for. A direct `h5py` read of two
string datasets is the right tool.

### Why the preprocessing chain is transcribed rather than called

`image_data()` cannot be reused as-is: it globs `*.jpg` from a hardcoded `<dataset_path>/train/` and
opens files, whereas F4 has uint8 arrays in memory. Staging 1804 PNGs as JPEGs to feed it would add a
lossy re-encode to the one thing FR11.2 says must be preserved exactly. So the three-line transform
chain is transcribed (FR4.3) while the **backbone class itself is imported unmodified** (FR4.1,
convention 2). Transcription is only acceptable because FR4.4 makes it a *provable* claim: a test
runs both paths on one file and asserts `torch.equal`. That test is unmarked and runs on CPU
everywhere (FR14.3) — it is the load-bearing assertion of this feature.

### Why the env is `scanpath`, not `senet`

Stage B's backbone is torchvision's `maskrcnn_resnet50_fpn(...).backbone.body`. It shares nothing with
SE-Net's `ImageFeatureEncoder` — no weights, no code, no init pickles. The Roadmap's dependency table
and F3's findings both say F4 "builds the same encoder"; **that is wrong** and FR1.1 records the
correction. F4 therefore inherits none of F3's detectron2 machinery: no `align_stage_prefix()`, no
`M2F_R50.pkl`, no MSDeformAttn, no `module load CUDA/11.6`. What it does inherit from
[TechStack §1.3](../constitution/TechStack.md) is the cluster facts: everything under
`/mnt/beegfs/home/<user>/`, never `/tmp`, and the `set +u`-across-`conda activate` guard (FR9.3),
kept even though `scanpath` has no MKL hook, because `ISP_ENV` is a tunable.

### Constitution constraints that shape the code

- **D1** — frozen files are not on this path. Nothing under `utils/` is imported.
- **D4** — FR2.4's cross-check is this feature's subject-identity guard. A trial mapped to the wrong
  exp_key hands a participant someone else's screen and every downstream metric still looks fine.
- **D5** — `feature_report.json` + `versions.txt`, written even on a fully-skipped run (FR8.3).
- **D7** — every anomaly raises. There is no counted-and-dropped path in F4; the `counters` block in
  the report exists to record zeros, not to absorb failures.
- **D8** — eval-only; nothing here trains anything.
- **Convention 2** — additive over invasive. `ResNetCOCO` imported, `sentence_transformers` stubbed at
  the call site, upstream untouched (FR14.4 asserts it).
- **Convention 5** — `data/eve_features/` is git-ignored, like `data/eve_bridge/` and
  `data/eve_senet/`.

---

## Implementation steps

### Step 1 — `tools/eve_prep/__init__.py`

Create the package with the single error type, mirroring `OsiePreflightError` / `EveSenetError`.

```python
class EvePrepError(RuntimeError):
    """Raised on any coverage, shape, identity or mapping violation (D7, FR13)."""
```

No imports beyond stdlib. Every other module in the package imports this and uses it for every raise
in the FR13 table.

### Step 2 — `tools/eve_prep/trial_keys.py`

Pure `h5py` + `numpy` + stdlib. **No torch** — this module must import on Windows and is what F5 will
reuse (FR10.3). Add a `sha256_file` helper or import the bridge's.

```python
TRIAL_KEY_SEP = "|"

def _as_str(v):                      # h5py returns bytes for vlen str
    return v.decode() if isinstance(v, bytes) else str(v)

def parse_trial_key(trial_key):
    """FR2.2. Split on the LAST separator: stimulus names may contain anything."""
    name, _, subject = trial_key.rpartition(TRIAL_KEY_SEP)
    if not name or not subject.isdigit():
        raise EvePrepError("malformed trial key {!r}".format(trial_key))
    return name, int(subject)

def load_trial_exp_keys(heatmaps_path, fixations_path=None):
    """FR2.1, FR2.3, FR2.5."""
    with h5py.File(heatmaps_path, "r") as f:
        stored_sha = _as_str(f.attrs["fixations_sha256"])
        grp = f["trials"]
        trial_keys = [_as_str(v) for v in grp["trial_key"][:]]
        exp_keys   = [_as_str(v) for v in grp["exp_key"][:]]
        # NOTE: grp["heatmaps"] is never touched -- ~88 MB we have no use for.
    if len(trial_keys) != len(exp_keys):
        raise EvePrepError(...)
    if len(set(exp_keys)) != len(exp_keys):
        raise EvePrepError("duplicate exp_key in {} (FR2.3): {}".format(...))
    if fixations_path is not None and sha256_file(fixations_path) != stored_sha:
        raise EvePrepError("fixations.json hash mismatch (FR2.5): store {} vs file {}")
    return {parse_trial_key(t): e for t, e in zip(trial_keys, exp_keys)}
```

```python
def derive_trial_exp_keys(samples_df, fixations, subject_id_map):
    """FR2.4 -- independent derivation. samples_df columns:
    exp_key, subject, stimulus_name, split, valid, stimulus_path."""
    to_eve = subject_id_map["to_eve"]                       # {"0": "train02", ...}
    index = {}                                              # (stimulus_name, eve_subject) -> exp_key
    for row in samples_df.itertuples():
        index.setdefault((row.stimulus_name, row.subject), []).append(row.exp_key)
    out = {}
    for rec in fixations:
        name, dense = rec["name"], int(rec["subject"])
        stem = name[:-4] if name.endswith(".jpg") else name
        eve  = to_eve[str(dense)]
        keys = index.get((stem, eve), [])
        if len(keys) != 1:
            raise EvePrepError(
                "{} trials in samples_df for stimulus {!r} subject {!r} (FR2.4)"
                .format(len(keys), stem, eve))
        out[(name, dense)] = keys[0]
    return out

def crosscheck_exp_keys(authoritative, derived):
    """FR2.4. Compare key sets AND values; raise naming up to 10 offenders."""
    # missing-in-derived, missing-in-authoritative, value disagreements -- three
    # separate messages, because they mean three different upstream problems.
    ...
    return {"source": "gt_heatmaps.h5", "derived_from": "samples_df",
            "agree": True, "n": len(authoritative)}
```

```python
EXP_KEY_RE = re.compile(r"[A-Za-z0-9_]+\Z")

def exp_key_filename(exp_key):
    """FR5.2, FR7.4."""
    if not EXP_KEY_RE.match(exp_key):
        raise EvePrepError("exp_key {!r} outside [A-Za-z0-9_]+ (FR5.2)".format(exp_key))
    if "jpg" in exp_key:
        raise EvePrepError(
            "exp_key {!r} contains 'jpg'; the OSIE loader's unanchored "
            "str.replace('jpg','pth') would corrupt it (TechStack 3.2)".format(exp_key))
    return exp_key + ".pth"
```

Also add `split_of(fixations)` returning `{(name, subject): split}`, used by both the extractor and
the checker to select trials — one derivation of that mapping, not two.

### Step 3 — `tools/eve_prep/extract_features.py`

The only torch importer in the package. Depends on Steps 1–2.

**3a. The upstream import, with the stub (FR4.1, FR4.2).** Lifted from `bash/test_osie.sh`, moved
into the module so the CLI works without the run script:

```python
def _import_backbone_class():
    if "sentence_transformers" not in sys.modules:
        try:
            import sentence_transformers          # noqa: F401
        except ImportError:
            stub = types.ModuleType("sentence_transformers")
            class _Unavailable:
                def __init__(self, *a, **k):
                    raise RuntimeError("text_data() is not part of F4; embeddings.npy is copied")
            stub.SentenceTransformer = _Unavailable
            sys.modules["sentence_transformers"] = stub
    from preprocess.feature_extractor import ResNetCOCO   # PYTHONPATH -> ISP/OSIE/GazeformerISP/src
    return ResNetCOCO
```

The caller is responsible for `PYTHONPATH`; the module raises a clear `EvePrepError` naming the
expected directory if the import fails.

**3b. Preprocessing (FR4.3) — the transcription under test.**

```python
RESIZE_INPUT   = (384 * 2, 512 * 2)      # (768, 1024) -- upstream's own expression
STIMULUS_SHAPE = (1080, 1920, 3)
_resize    = T.Resize(RESIZE_INPUT)
_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def preprocess(arr):
    if tuple(arr.shape) != STIMULUS_SHAPE or arr.dtype != np.uint8:
        raise EvePrepError("stimulus is {} {}, expected {} uint8 (FR3.2)".format(...))
    pil = PIL.Image.fromarray(arr)                       # identical to PIL.Image.open(...).convert('RGB')
    return _normalize(_resize(T.functional.to_tensor(pil))).unsqueeze(0)
```

Order is load-bearing: `to_tensor` → `resize` → `normalize`, matching upstream exactly. Resizing a
float tensor rather than a PIL image is what upstream does, and it changes the interpolation path —
do not "improve" it.

**3c. `extract_one` (FR4.5, FR4.8).**

```python
@torch.no_grad()
def extract_one(backbone, arr):
    out = backbone(preprocess(arr)).squeeze().detach().cpu()
    if tuple(out.shape) != FEATURE_SHAPE or out.dtype != torch.float32:
        raise EvePrepError("feature is {} {}, expected {} float32 (FR4.5)".format(...))
    return out
```

**3d. `extract_all` (FR4.7, FR8.1).** Iterates exp_keys in **sorted order** so a partial run is
resumable deterministically. Per exp_key: skip if the target exists and not `overwrite`
(`skipped_existing += 1`); else `bundle.get_stimulus(exp_key)` wrapped so a `ValueError` re-raises as
`EvePrepError` with the exp_key attached (FR3.3); `extract_one`; `torch.save` to a `.tmp` sibling then
`os.replace` (an interrupted run must not leave a truncated `.pth` that `check_features` later
"loads"); record the sha256. Returns counters plus `feature_sha256`.

`tqdm` over 1804 trials, and a progress line every 100 so the tee'd stdout is readable.

**3e. `copy_task_embeddings` (FR6).** `shutil.copyfile`, then re-open the **destination** and verify
dict / `"free-viewing"` / `(768,)` / float32, then compare `sha256(src) == sha256(dst)`. Verify the
copy, not the source — that is what catches a truncated write.

**3f. `main()` (FR8).** Resolve args; probe versions (Step 5's single source); load the mapping
(Step 2) with `fixations_path` passed so FR2.5 fires; open the bundle; derive and cross-check
(FR2.4); select trials by `--split`; assert the selected counts against `bridge_report.json`'s
`num_trials_test` / `num_trials_train` and raise on disagreement; extract; copy embeddings; write
`feature_report.json` **last and unconditionally** (FR8.3). Return `0`.

### Step 4 — `tools/eve_prep/check_features.py`

Modelled on `tools/osie_prep/check_features.py`, whose structure and `_listing()` helper are reused
verbatim in spirit. Depends on Steps 1–2.

```python
def check_features(fixations_path, heatmaps_path, feat_dir, split="both",
                   expected_shape=FEATURE_SHAPE):
    fixations = json.load(open(fixations_path))
    exp_of    = load_trial_exp_keys(heatmaps_path, fixations_path)   # FR2.5 fires here
    wanted    = [(r["name"], int(r["subject"])) for r in fixations
                 if split == "both" or r["split"] == split]
    missing, bad_shape = [], []
    for key in sorted(wanted):
        if key not in exp_of:
            raise EvePrepError("trial {} is in fixations.json but not in the store (FR2.1)".format(key))
        rel = exp_key_filename(exp_of[key])            # FR7.4 charset + no-'jpg'
        path = os.path.join(feat_dir, rel)
        if not os.path.isfile(path):
            missing.append(rel); continue
        t = torch.load(path, map_location="cpu")
        if tuple(t.shape) != expected_shape or t.dtype != torch.float32:
            bad_shape.append("{} {} {}".format(rel, tuple(t.shape), t.dtype))
    if missing:   raise EvePrepError(...)       # FR7.2 -- two distinct failures
    if bad_shape: raise EvePrepError(...)
    return {"n_checked": len(wanted), "n_test": ..., "n_train": ...,
            "missing": [], "bad_shape": []}
```

`main()` catches `EvePrepError`, writes it to stderr and returns `1`; on success prints the JSON
summary to stdout and returns `0` (FR7.3, FR7.5). **Note it does not import the bundle** — the guard
must be runnable before the bundle is staged.

### Step 5 — version probing, single-sourced

A small `versions()` in `extract_features.py` (or a shared `_versions.py`) returning
`{"python": ..., "torch": ..., "torchvision": ..., "numpy": ..., "PIL": ..., "h5py": ...,
"pandas": ..., "evedataset": ..., "device": ..., "cuda_available": ...}`, importing `evedataset`
whole (FR1.3). It is called once and its result is both printed (tee'd to `versions.txt`) and
embedded in the report — one source, the lesson TechStack §1.1 records from F1's wrong
skimage/opencv row.

### Step 6 — `bash/extract_eve_features.sh`

Structure mirrors `bash/test_osie.sh`; every tunable in the top block (FR9.2).

```bash
set -euo pipefail
HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/leonardo.ulloa}"
PROJECT_DIR="${PROJECT_DIR:-$HOME_DIR/projects/few-shot-scanpath}"
ISP_ENV="${ISP_ENV:-scanpath}"
BUNDLE_DIR="${BUNDLE_DIR:-$PROJECT_DIR/data/eve_bundle}"
BRIDGE_DIR="${BRIDGE_DIR:-$PROJECT_DIR/data/eve_bridge}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/data/eve_features}"
SPLIT="${SPLIT:-both}"
FORCE_FEATURES="${FORCE_FEATURES:-0}"
OSIE_EMB="${OSIE_EMB:-$PROJECT_DIR/ISP/OSIE/GazeformerISP/src/data/embeddings.npy}"
```

Order of business:

1. **FR9.4 preconditions** — every path above exists; `$BUNDLE_DIR/bundle.h5` and
   `$BUNDLE_DIR/stimuli` present; free space on `$OUT_DIR`'s filesystem ≥ 12 GB for `both`, 7 GB for
   `test` (`df -Pk`), failing with the FR5.4 budget in the message.
2. **Activate, with the guard (FR9.3):**
   ```bash
   source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"
   set +u; conda activate "$ISP_ENV"; set -u
   ```
   `-e` and `-o pipefail` stay on, so a genuine activation failure still stops the run.
3. **FR1.2/FR1.5 probe** — run the Step 5 probe, tee to `$OUT_DIR/versions.txt`, exit non-zero on any
   missing module or `cuda_available: false`.
4. **Guarded extraction (FR9.6):**
   ```bash
   export PYTHONPATH="$PROJECT_DIR/ISP/OSIE/GazeformerISP/src:${PYTHONPATH:-}"
   if [ "$FORCE_FEATURES" = "1" ] || ! python tools/eve_prep/check_features.py \
        --fix "$BRIDGE_DIR/fixations.json" --heatmaps "$BRIDGE_DIR/gt_heatmaps.h5" \
        --feat-dir "$OUT_DIR/image_features" --split "$SPLIT" ; then
     CUDA_VISIBLE_DEVICES=0 python tools/eve_prep/extract_features.py \
       --bundle-dir "$BUNDLE_DIR" --bridge-dir "$BRIDGE_DIR" --out-dir "$OUT_DIR" \
       --split "$SPLIT" --osie-embeddings "$OSIE_EMB"
   else
     echo "Feature cache complete; skipping (FORCE_FEATURES=1 to override)"
   fi
   python tools/eve_prep/check_features.py ... --split "$SPLIT"      # post-check, failure is fatal
   ```
5. Whole body tee'd to `$OUT_DIR/stdout.txt` (FR9.7).

Header comment carries: the `salloc` invocation, `bash`-never-`source`, **no `module load CUDA`
needed** (unlike F3 — nothing is compiled), and the FR5.4 disk budget.

### Step 7 — `tests/eve_prep/`

`conftest.py` provides: a synthetic `fixations.json` (a handful of trials across both splits), a
synthetic `gt_heatmaps.h5` written with `h5py` carrying only `trials/{trial_key,exp_key,subject,...}`
plus the attrs `load_trial_exp_keys` reads, a fake `samples_df` (a `pandas.DataFrame` with the six
real columns), and a `subject_id_map.json`. Markers `bundle` and `bridge` per FR14.2, registered in
`pytest.ini`/`pyproject` and wired with `--bundle-dir` in `conftest.addoption`.

Test modules:

- `test_trial_keys.py` — parse on names containing `|`; duplicate exp_key raises; sha mismatch
  raises; cross-check agreement, and each of the three disagreement kinds raising distinctly;
  `exp_key_filename` accepting a real key and rejecting `has_jpg_key` and `bad-key!`.
- `test_preprocess.py` — **FR4.4 bit-identity, unmarked** (Step 8 below); shape/dtype guards.
- `test_check_features.py` — complete cache returns 0 and the right counts; one missing file raises
  and the message names it; a `(700, 2048)` tensor raises as bad shape; `--split test` ignores train
  gaps; `main()` exit codes.
- `test_embeddings.py` — a synthetic `embeddings.npy` round-trips; a missing key, a `(384,)` value
  and a float64 value each raise.
- `test_no_upstream_edits.py` — FR14.4, `git status --porcelain ISP SE-Net` is empty.
- `test_bundle_integration.py` — `bundle`-marked: real bundle, 1804 trials, cross-check agrees,
  `get_stimulus` returns `(1080, 1920, 3)` uint8 on a sample.

### Step 8 — the FR4.4 bit-identity test, spelled out

This is the assertion that licenses the FR4.3 transcription, so it is written explicitly rather than
left to the implementer:

```python
def test_preprocess_matches_upstream_image_data(tmp_path):
    src = tmp_path / "train"; src.mkdir()
    tgt = tmp_path / "out" / "image_features"; tgt.mkdir(parents=True)
    arr = (np.random.RandomState(0).rand(1080, 1920, 3) * 255).astype(np.uint8)
    PIL.Image.fromarray(arr).save(src / "probe.jpg", quality=95, subsampling=0)

    # upstream path, unmodified
    image_data(dataset_path=str(tmp_path), output_path=str(tmp_path / "out"),
               device=torch.device("cpu"), overwrite=True)
    upstream = torch.load(tgt / "probe.pth")

    # our path, from the array the JPEG decodes to (NOT from `arr` -- JPEG is lossy)
    ours = extract_one(build_backbone(torch.device("cpu")),
                       np.array(PIL.Image.open(src / "probe.jpg").convert("RGB"), dtype=np.uint8))

    assert torch.equal(upstream, ours)
```

The comment on the last input matters: comparing against the pre-encode `arr` would fail for a reason
that has nothing to do with the transform chain, and would look like a real defect.

### Step 9 — `.gitignore` and the constitution

Add `data/eve_features/` to `.gitignore` (convention 5) — and check the blanket `*.json` rule the way
F3 had to for `eve_useremb.json`: `feature_report.json` lives under the ignored `data/` tree, so no
negation is needed, but confirm rather than assume.

Then update the constitution — a separate, deliberate step, not a side effect:

- **Roadmap** — F4 status; **OPEN-6 moved to resolved** with FR11's measurement and decision; the
  external-dependency table corrected to remove F4 from the SE-Net init-pickle row (FR1.1); F5
  unblocked.
- **TechStack** — a new §3.9 for the F4 artefacts (paths, `(768, 2048)`, the exp_key keying, the
  FR4.6 squash, the D2 deviation); the §3.2 note gains the fact that the EVE path never uses
  `str.replace('jpg','pth')`.
- **Mission** — §3's Stage B box gains "per-trial" and the OPEN-6 resolution; P1's "solved" note
  gains the D2 deviation pointer.

### Step 10 — `notes.md`

After the run, following the F2/F3 convention: what was measured on the bundle, what the run produced,
what F5 and F7 inherit, and anything that was not in the plan.

---

## Implementation Order

1. **Step 1** — `tools/eve_prep/__init__.py`, `EvePrepError`.
2. **Step 2** — `trial_keys.py`: mapping, derivation, cross-check, filename rule. *(no torch)*
3. **Step 4** — `check_features.py`. Built before the extractor so the guard exists first and the
   extractor can be validated against it.
4. **Step 3a-3c** — backbone import, `preprocess`, `extract_one`.
5. **Step 8** — the FR4.4 bit-identity test. **Nothing else proceeds until this passes**, on CPU, on
   the dev machine.
6. **Step 3d-3f** — `extract_all`, `copy_task_embeddings`, `main()`, `feature_report.json`.
7. **Step 5** — the version probe, single-sourced.
8. **Step 7** — the rest of `tests/eve_prep/`, green on Windows CPU.
9. **Step 6** — `bash/extract_eve_features.sh`.
10. **Step 9** — `.gitignore`; stage bundle + bridge artefacts to `/mnt/beegfs/...` (F3's lesson:
    `data/` is git-ignored, so nothing arrives on the cluster by `git pull`).
11. **The run**, under `salloc`, `SPLIT` per FR5.4.
12. **Step 9 (constitution)** and **Step 10 (`notes.md`)**, from the artefacts the run produced.
