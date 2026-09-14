# F4 — Per-trial image features for our stimuli

> Spec folder 1 of 3: **requirements** · [plan.md](plan.md) · [validation.md](validation.md)
> Roadmap feature: **F4**. Created 2026-09-14.
> Satisfies **D4**, **D5**, **D7**, **D8**. Deviates from **D2** — declared in FR12.
> **Resolves OPEN-6** (FR11).

---

## Goal

Produce the Stage B artefacts the EVE evaluation branch needs: one Mask R-CNN ResNet-50-FPN feature
tensor per **trial** — that is, per `(stimulus, participant)` pair — extracted from *the exact
1920×1080 screen capture that participant actually saw*, plus the `embeddings.npy` carrying the
`"free-viewing"` task key. Keying features per trial rather than per stimulus name is what resolves
**OPEN-6**: EVE presents the same photograph at a per-trial display scale (a deliberate augmentation
by the EVE team), so a single image per `stimulus_name` cannot represent what every participant saw,
and the bridge's first-exp_key-wins export would score 925 trials against a rendering the participant
never viewed. Per-trial keying removes the conflict rather than documenting around it, and it costs
nothing structurally: the evaluator never sees a feature path, and `OSIE_evaluation.__getitem__`
already carries a per-`(image, subject)` feature slot which upstream simply fills with a duplicate
tensor.

---

## Scope

**In scope**

- A new package `tools/eve_prep/` (CPU-importable on Windows; the extractor needs a GPU).
- Recovering the authoritative `(name, subject) → exp_key` mapping from `gt_heatmaps.h5`, and
  cross-checking it against an independent derivation from `bundle.samples_df`.
- Reading each trial's stimulus as a `(1080, 1920, 3)` uint8 array via
  `EveBundle.get_stimulus(exp_key)` — the images the EVE bundle stores, unmodified.
- Extracting a `(768, 2048)` float32 feature tensor per trial using the **upstream** `ResNetCOCO`
  backbone, with a preprocessing chain proven bit-identical to `feature_extractor.image_data()`.
- Writing `image_features/<exp_key>.pth`, one file per trial, for both splits.
- Producing `embeddings.npy` by copying and verifying the OSIE branch's shipped file.
- A preflight tool whose exit code guards re-extraction, a `feature_report.json` self-describing
  artefact, a cluster run script, and a pytest suite.

**Explicitly out of scope**

- **Any cropping, letterbox removal, rescaling, or coordinate remapping of the stimulus.** The
  per-trial display-scale variation is intentional augmentation and is preserved as presented.
  Decided 2026-09-14; see FR11.
- **Any change to `fixations.json`, `gt_heatmaps.h5`, `subject_id_map.json` or `bridge_report.json`.**
  F2's artefacts are consumed read-only and are not re-run. F3's `(38, 384)` embedding stays valid.
- **Any edit to `ISP/OSIE/GazeformerISP/`.** `ResNetCOCO` is *imported*, never modified (convention 2).
  The frozen files (D1) are not touched, read, or reached on this path.
- **The `ISP/EVE/GazeformerISP/` branch, its `dataset.py`, and the move of `torch.load` into the
  subject loop.** That is **F5**. F4 only states the contract F5 must implement (FR10).
- Training, RL, model changes, metric changes (D8).
- Regenerating the task embedding from `sentence-transformers`.

---

## Functional Requirements

### FR1 — Environment

**FR1.1** F4 runs in the **ISP-side** env — F1's `scanpath`, or any env satisfying FR1.2. It does
**not** use `senet`, Detectron2, MSDeformAttn, or the two SE-Net encoder init pickles.

> The Roadmap's external-dependency table lists the SE-Net init pickles against "F4 (same encoder)",
> and F3's findings say "F4 builds the same encoder". **Both are incorrect.** Stage B's backbone is
> `torchvision.models.detection.maskrcnn_resnet50_fpn(...).backbone.body`
> ([feature_extractor.py:16](../../ISP/OSIE/GazeformerISP/src/preprocess/feature_extractor.py#L16)),
> which shares no weights, no code and no init files with SE-Net's `ImageFeatureEncoder`. F4 therefore
> inherits neither the pickle derivation nor the `align_stage_prefix()` shim. It **does** inherit the
> `set -u` hazard only if a `senet`-style env is used; `scanpath` carries no such activation hook
> (TechStack §1.3), but FR9.3 lifts `set +u` across activation regardless.

**FR1.2** The run script MUST verify, before anything expensive, that the following import in the
target env: `torch`, `torchvision`, `numpy`, `PIL`, `h5py`, `pandas`, `evedataset`. A missing one
raises and exits non-zero. `h5py` and `pandas` are called out because `evedataset.bundle` imports both
at module scope and neither is guaranteed present in an env built for ISP eval.

**FR1.3** `import evedataset` transitively imports `evedataset.dataset`, which imports
`torchvision.transforms.v2`. The probe MUST therefore import `evedataset`, not
`evedataset.bundle`, so that a torchvision too old for `v2` fails at preflight rather than mid-run.

**FR1.4** CUDA MUST be available (`torch.cuda.is_available()`), and the resolved device is recorded.

**FR1.5** The resolved stack (python plus the version of every module in FR1.2) MUST be written to
`versions.txt` **and** embedded in `feature_report.json` (D5). One source, as `check_env.py` does for
F3.

### FR2 — The trial → exp_key mapping

**FR2.1** The authoritative mapping is `gt_heatmaps.h5`, which stores `trials/trial_key` and
`trials/exp_key` as aligned variable-length string datasets. F4 MUST read these two datasets
**directly**, without `GtHeatmapStore.load()`, which would materialise the ~88 MB heatmap array F4
never uses.

**FR2.2** `trial_key` has the form `"<name>|<subject>"`, e.g.
`"MIT-i05june05-static-street-boston-p1010800.jpg|25"`. The parse MUST split on the **last** `|` so a
stimulus name containing `|` cannot corrupt the subject id.

**FR2.3** On the realised cohort the mapping has **1804 entries with 1804 distinct exp_keys**
(1062 `test`, 742 `train`). A duplicate exp_key MUST raise — it would mean two trials sharing one
screen capture, which contradicts `duplicate_trial = 0` in `bridge_report.json`.

**FR2.4** The mapping MUST be **independently re-derived** from `bundle.samples_df` plus
`subject_id_map.json`: for each `fixations.json` record, the unique `samples_df` row with
`stimulus_name == name[:-4]` and `subject == to_eve[str(subject)]`. The two derivations MUST agree on
every trial; any disagreement raises and names the trials. This is the D4 check for this feature — an
off-by-one here hands a participant another participant's image and every metric still looks
plausible.

**FR2.5** The store's `fixations_sha256` root attr MUST match the sha256 of the `fixations.json`
being used (`46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b` on the realised
cohort). A mismatch raises: the mapping and the records would describe different builds.

### FR3 — Stimulus retrieval

**FR3.1** Each trial's image is obtained as `bundle.get_stimulus(exp_key)`, returning
`(1080, 1920, 3)` uint8 RGB. The array is used **as returned**. No crop, no resize, no colour
transform, no re-encode to JPEG.

**FR3.2** Any array whose shape is not `(1080, 1920, 3)` or whose dtype is not `uint8` MUST raise,
naming the exp_key (D7). `origin_size` in `bridge_report.json` is `[1080, 1920]` and the whole
coordinate contract depends on it.

**FR3.3** `get_stimulus` raises `ValueError` when `stimulus_path` is empty. F4 MUST let that
propagate with the exp_key attached rather than skipping the trial — a silently missing feature is
exactly the D7 failure mode.

### FR4 — Feature extraction

**FR4.1** The backbone is **imported** from upstream:
`from preprocess.feature_extractor import ResNetCOCO`, with `PYTHONPATH` including
`ISP/OSIE/GazeformerISP/src`. The class is used unmodified (convention 2).

**FR4.2** `feature_extractor.py` imports `SentenceTransformer` at module scope but uses it only
inside `text_data()`, which F4 never calls. A stub module MUST be registered in `sys.modules` before
the import if the real package is absent — the same call-site workaround `bash/test_osie.sh` uses.

**FR4.3** The preprocessing chain MUST be, in order, and with no other step:

```python
tensor = normalize(resize(T.functional.to_tensor(PIL.Image.fromarray(arr))))
# resize    = torchvision.transforms.Resize((384 * 2, 512 * 2))   # (768, 1024)
# normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
#                                              [0.229, 0.224, 0.225])
```

This is a literal transcription of
[feature_extractor.py:39-53](../../ISP/OSIE/GazeformerISP/src/preprocess/feature_extractor.py#L39-L53),
differing only in that the `PIL.Image` comes from a numpy array rather than `PIL.Image.open`.
`image_data()` itself cannot be reused: it globs `*.jpg` from a hardcoded `<dataset_path>/train/`
subdirectory, and staging 1804 PNGs as JPEGs to feed it would add a lossy re-encode for no benefit.

**FR4.4** Bit-identity with upstream MUST be **proven, not assumed**: given one image file, the
tensor produced by FR4.3 and the tensor produced by calling `image_data()` on a directory containing
that file MUST satisfy `torch.equal`. Verified by test (validation Group 1).

**FR4.5** The output tensor is `(768, 2048)` float32 on CPU — `24*32 = 768` spatial positions from
the `(768, 1024)` input, 2048 channels. Any other shape or dtype raises, naming the exp_key.

**FR4.6** The 1920×1080 → 1024×768 resize is a **non-uniform squash** (16:9 into 4:3), scale factors
`x = 1024/1920 = 0.5333`, `y = 768/1080 = 0.7111`. It MUST be recorded in `feature_report.json` and
reach F7. It is a **third**, independent distortion alongside F5's 512×384 metric screen (OPEN-4,
3.75 × 2.8125) and F3's SE-Net 512×320 input (0.2667 / 0.2963).

**FR4.7** Extraction MUST be resumable: an existing output file for an exp_key is skipped unless
`overwrite` is set. Mirrors upstream's own `overwrite=False` guard.

**FR4.8** The model is constructed **once** and run in `eval()` under `torch.no_grad()`.

### FR5 — Output layout

**FR5.1** Features are written to `<out_dir>/image_features/<exp_key>.pth`, one per trial.
`torch.save` of a bare CPU float32 tensor — the same object type `check_features` and the F5 loader
`torch.load`.

**FR5.2** `<exp_key>` is used **verbatim**, with `.pth` appended. On the realised cohort all 1804
exp_keys match `[A-Za-z0-9_]+` and **none contains the substring `jpg`** (verified 2026-09-14).
FR7.4 re-asserts both on every run.

> This keying sidesteps TechStack §3.2's unanchored `str.replace('jpg', 'pth')` trap entirely:
> F5 builds the path as `join(feature_dir, exp_key + ".pth")`, so no substring replacement occurs
> anywhere on the EVE path.

**FR5.3** Default `out_dir` is `data/eve_features/` (git-ignored, convention 5).

**FR5.4** **Disk budget, stated because it is not small.** Each tensor is
`768 × 2048 × 4 B = 6.29 MB`. Both splits = **1804 files ≈ 11.3 GB**; `test` alone = 1062 files
≈ 6.7 GB. `--split` selects `test`, `train`, or `both` (default `both`, matching the Roadmap's
877-image scope). `test.py` never builds a train-split loader (TechStack §3.6), so `test` alone is
sufficient for F5 and is the economical choice.

### FR6 — Task embeddings

**FR6.1** `embeddings.npy` is produced by **copying** `ISP/OSIE/GazeformerISP/src/data/embeddings.npy`
to `<out_dir>/embeddings.npy`. `text_data()` is not called and `sentence-transformers` is not
installed.

**FR6.2** The copy MUST be verified after writing: loads with `np.load(..., allow_pickle=True).item()`
to a `dict`, contains the key `"free-viewing"`, whose value is float32 with shape `(768,)`
(`args.lm_hidden_dim`). Any failure raises.

**FR6.3** The sha256 of both source and destination MUST be recorded in `feature_report.json` and
MUST be equal. Using the byte-identical file F1 scored with is what keeps F1's baseline and F5's run
comparable on this input.

### FR7 — Preflight (`check_features.py`)

**FR7.1** Given `fixations.json`, the trial→exp_key mapping and the feature directory, it asserts
that **every** trial in the selected split(s) has a loadable tensor at the path the F5 loader would
build.

**FR7.2** Each tensor MUST load and have shape `(768, 2048)` and dtype float32. Missing files and
wrong shapes are collected separately and reported as two distinct failures, each listing at most 10
offenders plus a total (the `_listing()` convention from `tools/osie_prep/check_features.py`).

**FR7.3** Exit code is load-bearing: `0` = complete, `1` = incomplete. The run script's
`if ! check_features` guard uses it to decide whether to spend GPU time (FR9.6).

**FR7.4** It re-asserts FR5.2 on the actual mapping: every exp_key matches `[A-Za-z0-9_]+` and
contains no `jpg` substring. This is cheap and catches a future bundle whose keys are less friendly.

**FR7.5** On success it prints a JSON summary to stdout: `n_checked`, `n_test`, `n_train`,
`missing: []`, `bad_shape: []`. `stderr` carries the human line.

### FR8 — `feature_report.json`

Written to `<out_dir>/feature_report.json`. MUST contain:

| key | content |
|---|---|
| `args` | the fully resolved argument namespace (D5) |
| `versions` | python + every module of FR1.2, and `device` |
| `created_utc` | ISO-8601 Z |
| `bundle_dir`, `fixations_path`, `heatmaps_path` | resolved absolute paths |
| `fixations_sha256` | the value asserted in FR2.5 |
| `n_trials`, `n_trials_test`, `n_trials_train` | 1804 / 1062 / 742 on this cohort |
| `n_extracted`, `n_skipped_existing` | this run's work |
| `feature_shape` | `[768, 2048]` |
| `resize_input` | `[768, 1024]` |
| `squash` | `{"x": 0.5333333333333333, "y": 0.7111111111111111, "uniform": false}` |
| `keying` | `"exp_key"` — the OPEN-6 resolution, recorded as data (FR11.3) |
| `embeddings` | `{"src", "dst", "sha256", "key": "free-viewing", "shape": [768], "dtype": "float32"}` |
| `exp_key_crosscheck` | `{"source": "gt_heatmaps.h5", "derived_from": "samples_df", "agree": true, "n": 1804}` |
| `counters` | every D7 counter below, always present, `0` when nothing fired |
| `feature_sha256` | `{exp_key: sha256}` for every written tensor |

**FR8.1** Counters: `missing_stimulus`, `bad_stimulus_shape`, `bad_feature_shape`,
`exp_key_mismatch`, `skipped_existing`.

**FR8.2** `feature_sha256` is what F5 asserts against before consuming the cache, the way F5 asserts
F3's `embedding_sha256` (Roadmap F5). It makes "are these the tensors that were checked?" answerable
from artefacts alone.

**FR8.3** The report MUST be written even when extraction is fully skipped because the cache was
already complete — otherwise a resumed run leaves no record (D5).

### FR9 — Run script (`bash/extract_eve_features.sh`)

**FR9.1** Single documented command, from the repo root, under `salloc` — the interactive path F1 and
F3 both use. `bash`, never `source` (TechStack §1 command conventions).

**FR9.2** Every tunable overridable from the environment: `HOME_DIR`, `PROJECT_DIR`, `ISP_ENV`,
`BUNDLE_DIR`, `BRIDGE_DIR`, `OUT_DIR`, `SPLIT`, `FORCE_FEATURES`.

**FR9.3** `set -euo pipefail` throughout, with `set +u` lifted **only** across `conda activate` and
restored immediately (TechStack §1.3's fourth cluster fact). Harmless for `scanpath`; correct if
`ISP_ENV` is ever pointed at an env with an MKL activation hook.

**FR9.4** Cheap preconditions before anything expensive: bundle dir, `bundle.h5`, `stimuli/`,
`fixations.json`, `gt_heatmaps.h5`, `subject_id_map.json`, the OSIE `embeddings.npy` source, and the
free space needed for FR5.4's budget.

**FR9.5** `versions.txt` teed into `<out_dir>` (D5), as `bash/test_osie.sh` does.

**FR9.6** Extraction guarded by `check_features.py`'s exit code; `FORCE_FEATURES=1` overrides.
`check_features.py` is re-run **after** extraction and its failure fails the job.

**FR9.7** stdout tee'd to `<out_dir>/stdout.txt` — the interactive path has no SLURM log.

### FR10 — The contract F5 must implement (stated here, built there)

**FR10.1** F5's `ISP/EVE/GazeformerISP/dataset/dataset.py` MUST move the feature load **inside** the
subject loop and key it by exp_key:

```python
# upstream OSIE (dataset.py:452-453, 465-491): ONE tensor, appended once per subject
img_path    = join(self.feature_dir, img_name.replace('jpg', 'pth'))
image_ftrs  = torch.load(img_path).unsqueeze(0)
for ids in self.imgid_to_sub[img_name]:
    ...
    images.append(image_ftrs)            # the same tensor, N times

# EVE: one tensor PER TRIAL
for ids in self.imgid_to_sub[img_name]:
    fixation = self.fixations[ids]
    exp_key  = self.exp_key_of[(img_name, fixation["subject"])]
    images.append(torch.load(join(self.feature_dir, exp_key + ".pth")).unsqueeze(0))
```

**FR10.2** This changes **nothing** the evaluator sees. `imgid_to_sub`, `__len__` over images, the
uniform 3-subjects-per-scored-image count, the positional diagonal and the retrieval block are all
untouched — `evaluation.py` receives scanpaths, never feature paths. **D1 is not engaged**: the frozen
set is `utils/evaluation.py` and `utils/evaltools/*`; `dataset/dataset.py` is an adapted copy the
Roadmap already lists as F5's to modify.

**FR10.3** F5 loads `exp_key_of` once at dataset construction, from `gt_heatmaps.h5` via
`tools/eve_prep`'s reader (FR2.1) — not per `__getitem__`.

**FR10.4** F5 MUST assert `feature_report.json`'s `fixations_sha256` against its own `fixations.json`
before consuming the cache.

### FR11 — OPEN-6 is resolved

**FR11.1 — The finding.** OPEN-6 recorded that 925 of 877 exported stimuli had renderings differing
between participants, described as "a brightness/scale/crop difference". Measured on the bundle
2026-09-14, the truth is narrower and benign: EVE presents each photograph **centred on a cream
`RGB(245,240,210)` page at a per-trial display scale**. For `MIT-i2273510179` the panel is 850 px
tall for `train24` and 1014 px for `train19`, aspect constant at 0.666; across 60 multi-viewer
stimuli the within-name scale ratio is a **median 1.175, max 1.358**, with aspect constant to 0.3 %
and the panel centred at `(959.5 ± 2, 540.5 ± 3)` in every one of a 200-trial sample. The apparent
"correlation 0.06 inside the differing box" is photo-against-background at displaced pixels, not
different content.

**FR11.2 — The decision (2026-09-14).** The scale variation is a **deliberate augmentation by the EVE
team to diversify gaze behaviour**. It is preserved exactly as presented. **No crop, no rescale, no
coordinate remapping.** `fixations.json` coordinates remain in native screen space and remain
correct — the screen capture and the fixations share one coordinate frame, which is precisely why no
remapping is needed.

**FR11.3 — The resolution.** Keying features by `exp_key` means there is no longer one image per
`stimulus_name` for two renderings to conflict over. **OPEN-6 is resolved by elimination.**
`stimulus_image_conflict = 925` becomes a property of the bridge's now-unused
`stimuli/<name>.jpg` export, not of anything F5 consumes.

**FR11.4** `data/eve_bridge/stimuli/` (877 `.jpg`, first-exp_key-wins) is **not used by F4 and must
not be used by F5.** It remains on disk for inspection. The report's `keying: "exp_key"` field is what
makes this checkable after the fact.

**FR11.5** F7 MUST state: the per-trial display-scale augmentation and its measured range; that
features are per-trial and therefore match what each participant saw; and the FR4.6 squash as the
third distortion in the stack.

### FR12 — Declared deviation from D2

**D2** states that everything downstream of Stage A sees only `fixations.json`,
`image_features/*.pth` and `embeddings.npy`, and that the native format never leaks past the bridge.
**F4 deviates: it reads `bundle.h5` and `bundle/stimuli/*.png` directly through `evedataset`.**

**Justification.** The per-trial screen captures are the input FR11.2 requires, and the bridge's
one-`.jpg`-per-`stimulus_name` export structurally cannot carry them. The alternatives were a bridge
re-run emitting 1804 per-trial JPEGs — which adds a lossy re-encode and re-opens F2 — or accepting
OPEN-6. Reading the bundle is the only option that preserves the presented stimulus bit-for-bit.

**Containment.** The blast radius stays at Stage B: the deviation is confined to `tools/eve_prep/`,
F5 consumes only `image_features/*.pth` + `feature_report.json` and never touches the bundle, and
FR2.4's cross-check re-anchors every bundle read to a bridge artefact. Decided 2026-09-14.

**Cost to record:** the cluster must hold `bundle.h5` (0.24 GB) and `bundle/stimuli/` (4.28 GB for
all 3095 PNGs; ~2.5 GB if subset to the 1804 trials in `fixations.json`), plus the `evedataset`
package and `h5py`/`pandas` in `ISP_ENV`.

### FR13 — Error conditions (D7)

Every one of these **raises `EvePrepError`** and exits non-zero. Nothing is counted-and-dropped.

| condition | FR |
|---|---|
| a module of FR1.2 missing, or `torch.cuda.is_available()` false | FR1.2, FR1.4 |
| `fixations_sha256` mismatch between store and `fixations.json` | FR2.5 |
| duplicate exp_key in the mapping | FR2.3 |
| the two exp_key derivations disagree on any trial | FR2.4 |
| a trial in `fixations.json` absent from the store | FR2.1 |
| `get_stimulus` raises, or returns a non-`(1080,1920,3)`/non-uint8 array | FR3.2, FR3.3 |
| an extracted tensor is not `(768, 2048)` float32 | FR4.5 |
| `embeddings.npy` missing the `"free-viewing"` key, or wrong shape/dtype, or sha mismatch | FR6.2, FR6.3 |
| any exp_key containing `jpg` or outside `[A-Za-z0-9_]+` | FR5.2, FR7.4 |
| any feature file missing or misshapen at preflight | FR7.2 |

### FR14 — Tests

**FR14.1** `tests/eve_prep/`, pytest, CPU, Windows-runnable, following the F2/F3 convention.

**FR14.2** Tests needing the real bundle are marked `bundle` and skipped unless `--bundle-dir` is
passed; tests needing F2's artefacts are marked `bridge` and skipped unless `data/eve_bridge/` exists.
Everything else runs on synthetic fixtures.

**FR14.3** The FR4.4 bit-identity test MUST run **unmarked** — it needs only a synthetic image and a
CPU backbone, and it is the single most load-bearing assertion in the feature.

**FR14.4** A test MUST assert no tracked file under `ISP/` or `SE-Net/` is modified (D1,
convention 2).

---

## Public API Summary

```python
# tools/eve_prep/__init__.py
class EvePrepError(RuntimeError): ...

# tools/eve_prep/trial_keys.py                       (stdlib + h5py + numpy; no torch)
def load_trial_exp_keys(heatmaps_path: str,
                        fixations_path: str | None = None) -> dict[tuple[str, int], str]:
    """FR2.1-2.3, FR2.5. Reads trials/{trial_key,exp_key} only -- never the heatmaps.
    Returns {(name, subject): exp_key}. Verifies fixations_sha256 when a path is given."""

def derive_trial_exp_keys(samples_df, fixations: list[dict],
                          subject_id_map: dict) -> dict[tuple[str, int], str]:
    """FR2.4. Independent derivation from the bundle's samples_df."""

def crosscheck_exp_keys(authoritative: dict, derived: dict) -> dict:
    """FR2.4. Raises EvePrepError naming the trials on any disagreement.
    Returns {"agree": True, "n": int}."""

def exp_key_filename(exp_key: str) -> str:
    """FR5.2. exp_key + '.pth', after asserting the charset and the no-'jpg' rule."""

# tools/eve_prep/extract_features.py                 (GPU; the only torch importer here)
FEATURE_SHAPE  = (768, 2048)
RESIZE_INPUT   = (768, 1024)          # (384*2, 512*2)
STIMULUS_SHAPE = (1080, 1920, 3)

def build_backbone(device) -> "ResNetCOCO":                              # FR4.1, FR4.2, FR4.8
def preprocess(arr: "np.ndarray") -> "torch.Tensor":                     # FR4.3 -> (1,3,768,1024)
def extract_one(backbone, arr: "np.ndarray") -> "torch.Tensor":          # FR4.5 -> (768,2048) f32 cpu
def extract_all(bundle, exp_keys: list[str], out_dir: str, device,
                overwrite: bool = False) -> dict: ...                    # FR4.7, FR8.1 counters
def copy_task_embeddings(src: str, dst: str) -> dict: ...                # FR6
def main(argv=None) -> int: ...                                          # CLI; writes feature_report.json

# tools/eve_prep/check_features.py                   (torch; exit code is load-bearing)
def check_features(fixations_path: str, heatmaps_path: str, feat_dir: str,
                   split: str = "both",
                   expected_shape: tuple = (768, 2048)) -> dict: ...     # FR7
def main(argv=None) -> int: ...                                          # 0 = complete, 1 = incomplete
```

**CLI**

```bash
py tools/eve_prep/extract_features.py \
    --bundle-dir DIR --bridge-dir data/eve_bridge --out-dir data/eve_features \
    [--split both|test|train] [--osie-embeddings PATH] [--overwrite] [--cuda 0]

py tools/eve_prep/check_features.py \
    --fix data/eve_bridge/fixations.json \
    --heatmaps data/eve_bridge/gt_heatmaps.h5 \
    --feat-dir data/eve_features/image_features [--split both]

bash bash/extract_eve_features.sh            # SPLIT=both by default
SPLIT=test bash bash/extract_eve_features.sh
```

---

## Dependencies

| direction | artefact | contract |
|---|---|---|
| reads | `data/eve_bridge/fixations.json` | TechStack §3.1; 1804 records, sha256 `46c6926f…` |
| reads | `data/eve_bridge/gt_heatmaps.h5` | `trials/{trial_key,exp_key}` + `fixations_sha256` attr only |
| reads | `data/eve_bridge/subject_id_map.json` | `to_eve` for FR2.4 |
| reads | `data/eve_bridge/bridge_report.json` | `num_trials*`, `origin_size` — cross-checked, not trusted blindly |
| reads | `<bundle>/bundle.h5` + `<bundle>/stimuli/*.png` | via `evedataset.EveBundle` — **the D2 deviation, FR12** |
| reads | `ISP/OSIE/GazeformerISP/src/data/embeddings.npy` | source for FR6, copied byte-identical |
| imports | `ISP/.../preprocess/feature_extractor.ResNetCOCO` | unmodified (convention 2) |
| **does not read** | `data/eve_bridge/stimuli/*.jpg` | FR11.4 — superseded by per-trial keying |
| **does not touch** | `utils/evaluation.py`, `utils/evaltools/*` | frozen (D1); not on this path at all |
| writes | `data/eve_features/image_features/<exp_key>.pth` | `(768, 2048)` float32, 1804 files ≈ 11.3 GB |
| writes | `data/eve_features/embeddings.npy` | byte-identical copy, `"free-viewing"` verified |
| writes | `data/eve_features/feature_report.json` | FR8 |
| writes | `data/eve_features/{versions.txt,stdout.txt}` | D5 |
| consumed by | **F5** | features + report; implements FR10 |
| consumed by | **F7** | FR4.6 squash, FR11 OPEN-6 resolution, FR12 deviation |
