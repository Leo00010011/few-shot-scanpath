# F4 — Notes

> Spec folder: [requirements.md](requirements.md) · [plan.md](plan.md) ·
> [validation.md](validation.md) · **notes**
> Written 2026-09-14, after implementation and before the cluster run.

---

## 0. The run — 2026-09-15, `SPLIT=test`, clean first time

**1062 extracted, 0 skipped, every D7 counter `0`.** The post-extraction
`check_features.py` exits `0` with `missing: []` / `bad_shape: []`, and
`exp_key_crosscheck.agree` is `true` over all 1804 trials. `fixations_sha256`
matches F2's `46c6926f…`; `embeddings.npy` copied at sha `a94bcf6534a2430e…`.

Four things the run itself established:

- **The exit-code guard works.** The pre-extraction `check_features.py` failed with
  all 1062 missing and exit 1 — which is what triggered extraction. Reading that
  `FATAL preflight failure` in the log as an error is a natural mistake; it is the
  guard doing its job.
- **`$LOCAL_SCRATCH` is scheduler-provided per job**: `bundle_dir` resolved to
  `/mnt/scratch/leonardo.ulloa/5522439/data/bundle`. The `/tmp/$USER` fallback never
  fired, and the staged tree is gone with the allocation — which is why
  `validate_features.py`'s two bundle-dependent checks skip unless `--bundle-dir`
  names a copy that still exists.
- **The env is `scanpath`, with F1's version drift**: python 3.11.14,
  torch 2.10.0+cu126, numpy 2.1.2, against a pin list of 3.8.18 / 1.13.1 / 1.24.3
  ([TechStack.md](../constitution/TechStack.md) §1.1). Stage B is a forward pass
  through a torchvision backbone, so this is far less exposed than the float metric
  paths — but it is recorded in `versions.txt`, per D5, rather than assumed harmless.
- **`evedataset` reported as `"unknown"`.** The package exposes no `__version__`;
  it does publish `0.1.0` as distribution metadata. `_version_of()` now falls back to
  `importlib.metadata`, so the D5 record stops discarding a version it could have
  had. Fixed after the run — the stored `versions.txt` from 2026-09-15 still says
  `unknown`.

---

## 1. Status: extraction done for the scored split; the checks are scripted

Everything in the plan's Implementation Order is built, tested and **run** for
`SPLIT=test`. What remains is executing `tools/eve_prep/validate_features.py` on the
cluster and transcribing its numbers (§8). Every CPU-side validation group passes on
the Windows dev machine:

| group | what it covers | result |
|---|---|---|
| 1 | preprocessing parity — **FR4.4 bit-identity** | 9/9 |
| 2, 3 | trial → exp_key mapping, filename rule | 26/26 |
| 4 | preflight `check_features.py` | 9/9 |
| 5 | task embeddings, report, atomic write | 9/9 |
| 6 | **bundle integration, on the real bundle** | 10/10 |
| 7 (static) | run-script invariants that are greppable | 12/12, 1 skipped |
| — | upstream untouched, keying greps | 4/4 |
| — | the post-run validator, incl. 8 deliberate corruptions | 10/10 |

**89 passed, 1 skipped** (`bash -n`, see §5). Run them with:

```
py -m pytest tests/eve_prep -q --bundle-dir <path to EveDataset/bundle>
```

**FR4.4 passed on the first run.** `torch.equal(upstream, ours)` holds between
`image_data()` and the transcribed chain, so the FR4.3 transcription is proven rather
than believed — the assertion the plan called load-bearing, discharged.

**Group 6 did not need the cluster.** The EVE bundle is present on the dev machine
(`eve_shared/EveDataset/bundle/bundle.h5`, 0.22 GB) and `evedataset` is installed, so
the D4 cross-check gate ran here: **`crosscheck_exp_keys()` agrees on all 1804
trials**, 1804 distinct exp_keys, 1062 test / 742 train, all filename-safe, dense ids
exactly `0..37`, and `origin_size == [1080, 1920]` agrees with the observed stimulus
shape. The cross-check was also shown to be **non-vacuous**: corrupting one
`samples_df` row makes it raise and name that trial.

---

## 2. The finding that was not in the plan: **the display scale is visible in the features, and validation's threshold is wrong**

Validation's falsifiable prediction was that two participants' tensors for the same
photograph are different but **cosine ≥ 0.7**. Measured on a CPU sample of 20
multi-viewer stimuli (573 stimuli have ≥ 2 viewers):

| | value |
|---|---|
| within-name cosine | min **0.4566**, p10 0.4987, median **0.7214**, max **0.9680** |
| pairs below 0.7 | **9 of 20** |
| cross-name cosine | min 0.2921, median **0.3821**, max **0.5088** (n = 37) |
| max cross ≥ min within | **yes** — the extremes overlap |

**The ≥ 0.7 floor does not hold, and what replaces it is better evidence, not worse.**
Sorting the 20 pairs by cosine sorts them almost exactly by the per-trial
display-scale ratio:

| scale ratio | 1.004 | 1.008 | 1.020 | 1.046 | 1.077 | 1.107 | 1.117 |
|---|---|---|---|---|---|---|---|
| within-name cosine | 0.968 | 0.962 | 0.904 | 0.787 | 0.606 | 0.459 | 0.457 |

That is FR11.1's measured augmentation (median ratio 1.175, max 1.358) showing up in
the feature space exactly as it should. A flat floor was the wrong shape for the
prediction: the similarity is a **function of the scale difference**, so a threshold
chosen without reference to the ratio was always going to fail on the widely-scaled
pairs. The two things OPEN-6 actually needs are both decisively true:

- **No pair is bit-identical.** Per-trial keying did not collapse back to per-name.
- **Within-name beats cross-name by a wide margin on the medians** — 0.721 vs 0.382.

`test_per_trial_keying_actually_did_something` therefore asserts *those*, with the
reasoning and the measured numbers written into its docstring.
**`validation.md` is deliberately left unedited**, carrying the prediction as it was
made — the F3 convention for a stated expectation that did not hold.

**For F7:** this is the quantitative statement about the display-scale augmentation
that OPEN-6 always lacked. Quote the range and the mechanism, not a single number.

---

## 3. Two more stated expectations, one off and one confirmed

- **Sparsity runs higher than validation's band.** The predicted fraction of exact
  zeros was "roughly 0.3–0.8"; measured **0.752–0.858, median 0.819**. Above the
  band, far below the 0.95 threshold validation set for "worth investigating". These
  stimuli are a photograph centred on a large uniform cream page, and a big constant
  background region is a plausible reason for post-ReLU features to be sparser than
  a full-bleed photograph would be — **plausible, not verified**; recorded so the
  real run's figure has something to be compared against.
- **Post-ReLU sanity confirmed.** `t.min() == 0.0` exactly, `std > 0`, all finite, on
  every tensor sampled. The right module is tapped.
- **Fixations land on the stimulus, not the page.** Spot-checked on 3 trials: 5/5,
  7/7 and 7/7 fixations fall on non-cream pixels. The trials are mapped to the
  capture the participant actually saw (D4, end to end).

---

## 4. Two checks validation specified that cannot be written as specified

Both are recorded because the specified form looks reasonable and silently does not
work — a later reader would otherwise "fix" them back.

1. **"Assert `heatmaps` appears nowhere in the module source" is impossible.** The
   file is *called* `gt_heatmaps.h5`, so the substring is unavoidable in the
   docstring, in the parameter name and in every path. The test does the real thing
   instead: it monkeypatches `h5py.Dataset.__getitem__` to raise on any dataset whose
   name contains `heatmaps`, and asserts `load_trial_exp_keys` still completes. That
   is what F5's reuse (FR10.3) actually depends on.
2. **`grep "replace('jpg'"` flags the documentation, not the defect.**
   `exp_key_filename`'s error message quotes the banned idiom on purpose, to explain
   what it protects against. The test walks the **AST** for a call
   `<expr>.replace("jpg", ...)` instead, so it is about what executes. Stripping
   string literals wholesale would have been worse than useless — it would delete the
   very argument the grep is looking for and the check would pass unconditionally.

---

## 4a. The run script was wrong in four ways, caught by reading a sibling project

`~/projects/EyeNet-Pipeline/whole_train.sh` — another project of ours on the same
cluster, against the same EVE bundle — was checked after the script was written, and
it contradicted four of its assumptions. All four are fixed; all four would have cost
a failed allocation to discover.

1. **The bundle belongs on node-local scratch, not beegfs.** `whole_train.sh` rsyncs
   a tar from beegfs to `$LOCAL_SCRATCH/data/`, extracts it there, and points at
   `${LOCAL_SCRATCH}/data/bundle`. Its config template states the reason outright:
   *"data paths point at the rsync'd local-scratch copy, not the shared eve_shared
   mount, to keep training I/O off the network filesystem."* F4 does **1804 random
   reads of ~1.3 MB PNGs**, which is precisely that workload. `BUNDLE_DIR` now
   defaults to `$LOCAL_SCRATCH/data/bundle` and the script stages into it
   (`STAGE_BUNDLE=0` opts out).

   **This does not contradict [TechStack.md](../constitution/TechStack.md) §1.3's
   "`/tmp` is node-local" warning**, and the distinction is worth stating because the
   two look contradictory: §1.3 is about anything that must *persist* — envs, source
   trees, artefacts, the D5 record — where node-local storage silently loses things.
   A **read-only data copy re-staged per job** is what scratch is for. Everything F4
   *writes* still goes to beegfs.

2. **The existing archive is used as-is — and what is slow is the *expansion*, not
   the copy.** *(Corrected 2026-09-15 on the user's instruction; the first version of
   this note had it wrong.)* The bundle ships as a tar on beegfs, and unpacking
   thousands of small files onto a network filesystem is the expensive part. Copying
   the whole archive to scratch and expanding it **there** is what `whole_train.sh`
   does and why. My first fix proposed repacking a narrowed
   `bundle_stimuli.tar` (`bundle.h5` + `stimuli/`, skipping the 11 GB
   `face_crops/`); that optimises the wrong axis — it saves local disk that is sized
   for it, at the cost of a repack every time the bundle changes, and it makes the
   run depend on a hand-built artefact. The whole archive is extracted and the parts
   F4 does not read are simply never opened. Narrowing is left as a named fallback on
   the `tar -xf` for a tight-scratch day.

   **The beegfs archive is only ever read.** `rsync` copies it; nothing moves,
   renames or deletes it. The script reclaims the *scratch copy* it made after
   extraction, under a guard against `BUNDLE_TAR` itself living on the staging
   filesystem — where the two paths would be the same file and the `rm` would take
   the original. Two tests pin this: one that no `rm`/`mv` line mentions
   `$BUNDLE_TAR` and that the removal is guarded, one that the copy precedes the
   expansion and that it is the scratch copy being expanded.

3. **`conda activate scanpath` needs the env image mounted first.** `scanpath` lives
   inside `my_env.ext4`; `bash/test_osie.sh` mounts it with `sudo mount_image.py
   my_env.ext4 --rw` and treats a failure as **non-fatal**, because a second mount in
   one allocation exits non-zero and `set -e` would abort before any precondition
   ran. The script had no mount at all and would simply have failed to activate. F3's
   `embed_eve_subjects.sh` legitimately has none — `senet` is a beegfs prefix, not an
   image — which is why the omission looked reasonable.

4. **`$(conda info --base)` cannot run before conda is on PATH.** Both
   `whole_train.sh` and `test_osie.sh` spell the path out:
   `source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"`. The substitution form was
   inherited from `embed_eve_subjects.sh`, where it happens to work because that
   script is run after a manual `module load`. Now explicit, with a test.

**One thing the sibling project confirmed rather than corrected:**
`EveBundle.load(bundle_dir)` is exactly the API `scripts/train.py` uses, so
`_open_bundle()` matches established practice.

---

## 5. Smaller things the next feature inherits

- **`data/eve_features/` needs no `.gitignore` change.** The existing `data/` rule
  covers it, `feature_report.json` included — **confirmed with
  `git check-ignore -v`**, not assumed, the way the `spec/` and `SE-Net/configs/`
  negations had to be learned twice (TechStack convention 5). The rule's comment now
  names all three generated trees.
- **FR1.1's correction is real and worth repeating: F4 shares nothing with F3.** No
  detectron2, no MSDeformAttn, no `M2F_R50.pkl`, no `align_stage_prefix()`, and
  **no `module load CUDA/11.6`** — nothing is compiled. `test_no_module_load_cuda`
  asserts the run script stays that way. The Roadmap's dependency table and F3's
  notes both said "F4 builds the same encoder"; they are wrong, and §7 below fixes
  the table.
- **The `set +u` guard is kept even though `scanpath` does not need it.** `ISP_ENV`
  is a tunable, so the moment anyone points it at `senet` the MKL activation hook
  aborts the script before a line of our code runs (TechStack §1.3).
- **`bash -n` is skipped on Windows** when the `bash` on PATH is a WSL one that
  cannot read a Windows path. The test detects that case and skips rather than
  reporting a syntax failure; it runs for real on the cluster.
- **`pipefail` is what makes the `tee` safe.** Without it the tee's exit status masks
  a failed post-check and the job reports success on an incomplete cache.

---

## 6. What F5 inherits

- Features at `data/eve_features/image_features/<exp_key>.pth`, `(768, 2048)` float32,
  path built by **concatenation** — `join(feature_dir, exp_key + ".pth")`. No
  `str.replace('jpg', 'pth')` anywhere on the EVE path.
- `tools/eve_prep/trial_keys.load_trial_exp_keys()` is the reader to call **once at
  dataset construction** (FR10.3), not per `__getitem__`. It is torch-free.
- FR10.1's two-line change: move the `torch.load` **inside** the subject loop. The
  per-`(image, subject)` slot already exists — upstream fills it with duplicates.
- Assert `feature_report.json`'s `fixations_sha256` before consuming the cache
  (FR10.4), and `feature_sha256` for the tensors themselves (FR8.2).
- **`data/eve_bridge/stimuli/` must not be used** (FR11.4). A test greps for it.

## 7. What F7 inherits

- The **third squash**: 1920×1080 → 1024×768, x = 0.5333, y = 0.7111, non-uniform.
  Alongside F5's 512×384 metric screen (3.75 / 2.8125, OPEN-4) and F3's SE-Net
  512×320 input (0.2667 / 0.2963). Three independent distortions; all three must be
  stated.
- **OPEN-6 resolved by elimination**, with §2's numbers as the evidence: the
  per-trial display scale is EVE's deliberate augmentation, preserved exactly, and
  features are keyed per trial so no two renderings contend for one slot.
  `stimulus_image_conflict = 925` becomes a property of the bridge's now-unused
  per-name `.jpg` export, not of anything F5 consumes.
- The **declared D2 deviation** (FR12): F4 reads `bundle.h5` directly. Contained to
  `tools/eve_prep/`, and every bundle read is re-anchored to a bridge artefact by the
  FR2.4 cross-check, which agrees on all 1804 trials.

---

## 8. Still to do

1. **Run the post-run validator on the cluster** — CPU-only, login node, no GPU:
   ```
   python tools/eve_prep/validate_features.py        --features data/eve_features --bridge-dir data/eve_bridge        --senet-report data/eve_senet/seed0/senet_report.json
   ```
   It writes `data/eve_features/validation_report.json` and exits non-zero on any
   failure. Add `--bundle-dir` only while a staged bundle still exists (inside the
   allocation, `$LOCAL_SCRATCH/data/bundle`), otherwise its two bundle-dependent
   checks skip — and a skip is reported as a skip, never as a pass.
2. **Transcribe its numbers here** — the realised sparsity band and the within-/
   cross-name cosine medians from the *real* tensors, to sit beside the pre-run CPU
   estimates in §2 and §3. Then flip F4 to ✓ DONE.
3. **Optional: `SPLIT=train`.** Only if F5 ever needs support-split features;
   `test.py` builds no train-split loader. The extractor is resumable, so it would
   fill the 742 gaps and leave the 1062 alone.
4. **Not exercised, and honestly so:** validation Group 7's four deliberate failure
   runs (missing `bundle.h5`, missing `stimuli/`, missing `gt_heatmaps.h5`,
   insufficient space) and the `FORCE_FEATURES=1` determinism check. The clean run
   exercised the happy path only.
