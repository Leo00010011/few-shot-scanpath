# F3 — Notes

> Spec 4 of 4, the F2 convention. Read with [requirements.md](requirements.md),
> [plan.md](plan.md) and [validation.md](validation.md).
> Written 2026-09-14, from the run's own artefacts.

What the first real run of Stage C found. Five things cost time; none of them were in the plan, and
**four of the five are inherited by F4 and F5**, which build the same encoder in the same env.

---

## 1. The run

| | |
|---|---|
| node / env | `hpc-gpu3`, `senet` (`/mnt/beegfs/home/leonardo.ulloa/envs/senet`), `check_env.py` **11/11 ok** |
| command | `bash bash/embed_eve_subjects.sh`, then `SEED=1 …`, then `bash bash/pin_duration_channel.sh` |
| cohort | 38 participants × 10-shot = **380 support scanpaths**, 380 forward passes |
| output | `data/eve_senet/seed{0,1}/eve_fewshot_user_embedding_10_seed{0,1}.pt`, `(38, 384)` float32 |
| seed 0 `embedding_sha256` | `e904a165c985b33fde68f5de78ef83acf13bbe9494e8d96a1fe3e0edc755166e` |
| checkpoint | `ckp_11999.pt`, sha256 `4c385b748e168e5a…` |

F5 consumes the seed-0 tensor and **must assert that hash** against `senet_report.json` before using
it (FR8.2), so the two features cannot silently discuss different artefacts.

---

## 2. Five things that were not in the plan

### 2.1 Two init pickles that are distributed nowhere — and both loads are unguarded

`ImageFeatureEncoder.__init__` **strict**-loads `data/M2F_R50.pkl` (the ResNet) and
`data/M2F_R50_MSDeformAttnPixelDecoder.pkl` (the deformable pixel decoder). Neither is in this
repository, the checkpoint bundle, or the READMEs' Drive folders, and the authors' own
`if os.path.exists(...)` guards are **commented out** above both — so a missing file is a bare
`FileNotFoundError` several frames inside SE-Net, not a skip.

They are *initialisation* files: `load_model()`'s `load_state_dict(ckp["model"], strict=False)`
overwrites every one of their tensors on the next line, and the released checkpoint carries **all**
of them — 265 under `encoder.backbone.*`, 117 under `encoder.pixel_decoder.*`. So
`tools/eve_senet/make_backbone_init.py` lifts both subtrees **out of that checkpoint** rather than
downloading third-party pickles whose provenance cannot be checked against the checkpoint we
actually run. The encoder then reaches the released weights by two independent routes, and the two
strict loads are themselves the completeness check.

**Recorded as a Roadmap §6 dependency row, flagged for F4.**

### 2.2 detectron2 0.6 names the ResNet stages differently from the checkpoint ◀ the dangerous one

The checkpoint stores `encoder.backbone.stages.res{2..5}.*`; **0.6**'s `build_resnet_backbone`
registers the stages directly and wants bare `res{2..5}.*`. Same 265 tensors, identical shapes,
**zero** structural difference — the diff is the string `stages.` on 260 keys.
`models.py`'s own rename loop (`"stages." + k` for leading-`res` keys) is written for the authors'
detectron2 and cannot produce 0.6's names from any input, so under 0.6 the encoder cannot be
constructed as shipped.

`embed.py::align_stage_prefix()` translates the names — never the values — in whichever direction
the *installed* detectron2 needs, so it is a no-op under the authors' version. It applies twice:
through a wrapper on `src.models.build_backbone` for the strict init load, and to the checkpoint
before `load_state_dict`.

**The checkpoint-side one is what matters.** Without it all 260 stage keys land in `unexpected_keys`,
`strict=False` swallows them, and the encoder runs on its *initialisation* while producing
embeddings that look entirely normal — no error, no zero rows, plausible norms. That is why
`load_model()` now raises on any remaining unexpected `.backbone.` key instead of trusting the shim.
The realised run reports `renamed 260 / direction "strip"` in both places, and `unexpected_keys` is
**empty**.

Nothing under `SE-Net/` was edited (convention 2); the wrapper is installed on the module object for
the duration of construction and removed after.

### 2.3 `set -euo pipefail` versus `conda activate`

The `senet` env ships `etc/conda/activate.d/libblas_mkl_activate.sh`, which reads
`MKL_INTERFACE_LAYER` before assigning it. Under `nounset`, `conda activate` therefore aborted the
whole script before a single line of F3 ran, with a message that reads like a broken env. It is not.
Both F3 scripts now lift `set +u` across the activation and restore it immediately; `-e` and
`-o pipefail` stay on, so a genuine activation failure still stops the run. **F4 and F5 hit this the
moment they activate `senet`** — `bash/test_osie.sh` has the same construct but the `scanpath` env
has no such hook, which is why F1 never saw it.

### 2.4 `data/` is git-ignored, so F2's artefacts were not on the cluster

The bridge ran on Windows (F2) and `data/` is ignored by convention 5, so `fixations.json`,
`subject_id_map.json`, `bridge_report.json`, `gt_heatmaps.h5` and the 877 stimuli (~310 MB) had to be
shipped separately — `tar -cf - data/eve_bridge | ssh … tar -xf -`. The sha256 gate (FR11.1) is what
makes that transfer safe to trust rather than something to re-verify by hand.

`data/osie_embeddings.npy` (FR6.5's task embedding, read from a path `Siamese_Triplet_Gaze`
hardcodes) did **not** need shipping: it is byte-identical to the **tracked**
`ISP/OSIE/GazeformerISP/src/data/embeddings.npy`, so a `cp` materialises it.

### 2.5 `MISSING_KEY_ALLOWLIST` stays empty — as an observation, not a guess

It shipped empty on purpose, to be resolved by the first real run. The run reports
`missing_keys = []` and `unexpected_keys = []`. Nothing needed excusing, and in particular no
`subject_predictor.*` key is missing, so `Data.num_subjects = 10` agrees with the checkpoint and the
subject head is not randomly initialised (FR6.2). **The allowlist is correct as empty and should stay
that way** — a future non-empty `missing_keys` is a real finding, not a nuisance.

---

## 3. Validation outcomes

### Group 4 — model loading and forward pass ✓

- `check_env.py` 11/11 `ok` on `hpc-gpu3`, `cuda_available: True`.
- `missing_keys` 0, `unexpected_keys` 0; no `subject_predictor.*` key.
- **`n_forward == 10` for all 38 subjects.** This is the check that catches upstream's
  `drop_last=True` at `bs = 8` silently discarding 20 % of a 10-scanpath support set; it would have
  read `8`. It reads `10`.
- `n_records = 380 = 38 × 10`, `forward_passes = 380`.

### Group 5 — the duration-deadness pin ✓ **pinned**

All three arms — decile `bins`, raw milliseconds, absurd `1e6` — produced **byte-identical files**:
sha256 `e904a165c985b33f…` for every one, `torch.equal` true, `max_abs_diff = 0.0`.

The absurd arm is what makes this conclusive: a tolerance-based check could hide a small live
contribution, but no live duration channel can consume `1e6` and return bit-identical output. The
observation in `SE-Net/src/models.py` — `duration_encoding` added into `ventral_pos`, then
`ventral_pos.fill_(0)` on the next line, after `ventral_embs += ventral_pos` — is now evidence, not
inference. Recorded in `seed0/duration_pin.json` and stamped into `senet_report.json` as
`duration_channel_pinned: true`.

FR3's decile binning therefore stays a *faithful input contract*, not a load-bearing computation.

### Data Validity ✓, with one expectation corrected

| check | expected | observed |
|---|---|---|
| tensor | `(38, 384)` float32 | ✓ |
| zero rows | 0 | 0 |
| duplicate rows | none | none |
| row-norm max/min | below ~3 | **1.74** (8.5851 … 14.9418) |
| off-diagonal cosine | mean well below 1.0, visible spread | seed 0 **0.0623 / 0.7572 / 0.9719**; seed 1 0.1735 / 0.7628 / 0.9679 |
| union over seeds 0+1 > 10 | 37 of 38 | **37 of 38**; subject 34 (pool exactly 10) draws the same 10 |
| support/query disjoint | empty | empty |
| `truncated_scanpaths` | 0 | 0 |
| duration bin edges | `[100, 131, 151, 169, 188, 210, 234, 263, 308, 390, 1144]` | exactly |
| bin occupancy | ≈ 424 each | 409 … 432 |
| bridge cross-check | 38 / 10 / 742 | 38 / 10 / 742, and 742 train records in the JSON |
| SE-Net input + rescale | 512×320, 0.2667 / 0.2963 | ✓ |

**`oob_fixations_dropped = 0`, where validation predicted a small non-zero.** The prediction came
from EVE's `Y` reaching exactly 1080.0, which rescales to exactly 320.0 and is dropped. It does not
apply to the *support* subset: the whole train split holds **one** frame-edge fixation and the seed-0
draw does not include it (3 more sit in the test split). The expectation was about the dataset, the
counter is about 380 selected scanpaths. Nothing is wrong with the rescale — and a value in the
hundreds would still mean what validation says it means.

`len1_scanpaths = 2` — two support scanpaths carry a single fixation. Counted, not dropped (D7).

---

## 4. What F5 inherits

- **`--user_emb_path data/eve_senet/seed0/eve_fewshot_user_embedding_10_seed0.pt`**, and assert
  `embedding_sha256 = e904a165c985b33f…` from `senet_report.json` before consuming it.
- **`--fewshot_subject 0 1 2 … 37`, ascending.** `select_fewshot_subject()` remaps the argument list
  to a dense range and indexes `subject_embed[subjects]` with it; only ascending order gives the
  identity remap that keeps row *i* on subject *i*. Any other order permutes the whole cohort's
  embeddings and every metric still looks plausible. F3's rows are in `subject_id_map.json`'s dense
  order, verified by `verify_embedding.py`'s printed `dense_id → eve_id → ||row||` table.
- **A second, different squash**, for F7: SE-Net's input is 512×**320** (0.2667 / 0.2963) while F5's
  metric screen is 512×**384** (OPEN-4). Two independent distortions; both get stated.
- Seed 1 exists and is a genuinely different draw (37 of 38 subjects move), so OPEN-3's
  "average over `--random_support` repeats" remains answerable.

## 5. For F7

- The embeddings are **ours**; the encoder that produced them is **OSIE-trained**. What transfers is a
  representation learned on OSIE subjects, applied to EVE support scanpaths (OPEN-2).
- **Mean off-diagonal cosine 0.757** (min 0.062, max 0.972). The encoder does discriminate our
  participants — but 0.757 is high, and the claim to make is "distinguishable", not "well separated".
  Quote the number; the near-duplicate pair at 0.972 is worth naming.
- `len1_scanpaths = 2`, `oob_fixations_dropped = 0` for the realised support set.
- The encoder's weights reached the model through a **name translation** (§2.2) and its init pickles
  were **derived from the checkpoint** (§2.1). Neither changes a value, and `unexpected_keys = []` is
  the evidence — but both are deviations from the authors' own load path and belong in the write-up.
