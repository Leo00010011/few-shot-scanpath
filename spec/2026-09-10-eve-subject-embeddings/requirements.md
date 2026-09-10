# F3 — Subject embeddings for our EVE participants

> Spec 1 of 3. Read with [plan.md](plan.md) and [validation.md](validation.md).
> Constitution: [Mission](../constitution/Mission.md) · [TechStack](../constitution/TechStack.md) · [Roadmap](../constitution/Roadmap.md)
> Created: 2026-09-10 · Implements **Stage C** · Satisfies **D4, D5, D7, D8**; constrained by **D1**

---

## Goal

Produce a `(38, 384)` float32 subject-embedding tensor for our 38 EVE participants, computed by running
the **released SE-Net checkpoint** (`weights/OSIE-20260904T121550Z-1-001/OSIE/ckp_11999.pt`) over each
participant's own 10-shot support set from the F2 bridge, so that `ISP/EVE/.../src/test.py --user_emb_path`
can condition on *our* subjects rather than borrowing OSIE subjects 10–14. This resolves **OPEN-2 toward
option 1** — the embeddings are genuinely ours, so the "personalization" F5 measures is our participants'
and not somebody else's. Row *i* of the tensor is EVE participant `subject_id_map.json["to_eve"][str(i)]`,
and that correspondence is asserted end-to-end rather than assumed (D4).

---

## Scope

### In scope

- Building the `senet` conda environment on the cluster, **including the Detectron2 and MSDeformAttn
  source installs** — the project's highest-risk install (TechStack §1), specced as numbered steps rather
  than left as an operational footnote.
- A new top-level package `tools/eve_senet/` that drives the released SE-Net checkpoint over EVE support
  scanpaths and writes the embedding tensor. Modelled on `tools/eve_bridge/` and `tools/osie_prep/`.
- Deriving EVE's own **decile duration bins** over the support split, matching the input regime the
  released checkpoint was trained under (`osie_fixations_update_duration.json`, TechStack §3.7).
- One embedding row per participant, computed from **exactly 10 support scanpaths** each, drawn
  deterministically from that participant's F2 support pool.
- Verification tooling: shape, dtype, row-order-vs-`subject_id_map.json`, no-zero-row, and the
  duration-deadness pin.
- A single documented run script under `bash/`.

### Explicitly out of scope

- **Training or fine-tuning SE-Net** (D8). `--eval-only` semantics only: forward passes, no optimiser.
- **Editing any file under `SE-Net/`.** Working convention 2 (additive over invasive). The one shared
  behaviour we need to change — triplet sampling — is obtained by *subclassing*, not by patching
  (FR5.2). If a future need cannot be met additively, it stops and gets escalated.
- **Editing frozen files** (D1): `ISP/*/GazeformerISP/src/utils/evaluation.py` and `utils/evaltools/*`.
  F3 does not touch the ISP tree at all.
- **`ISP/EVE/GazeformerISP/`** — creating the eval branch is F5's. F3 produces an artefact F5 consumes.
- **Feature extraction** for our stimuli — F4, and still blocked on OPEN-6.
- **Deciding how many `--random_support` repeats F5 averages over** (OPEN-3's residue). F3 makes repeats
  *possible* by keying every artefact on `--seed`; choosing the count is F5's.
- **Any change to `fixations.json` or `gt_heatmaps.h5`.** F3 is a pure reader of F2's artefacts.
  Working convention 10 (bridge artefacts are addressed positionally) means regenerating one would
  require regenerating the other; F3 never does either.

---

## Functional Requirements

### FR1 — Environment

**FR1.1** The `senet` environment is created from `SE-Net/environment.yml` (TechStack §1 `senet` table:
python 3.8.0, pytorch 1.11.0+cu113, torchvision 0.12.0, numpy 1.23.5, scipy 1.10.0, timm 0.6.13,
**multimatch-gaze 0.1.3**). Unlike `ISP/environment.yml`, this file is not known to be internally
inconsistent, so it is created as written first; only if the solve fails is a minimal env built from the
pin table, and any such deviation is recorded per D5.

**FR1.2** Detectron2 is installed from source per the HAT repo instructions, into the activated `senet`
env, and the install is verified by `import detectron2` succeeding.

**FR1.3** MSDeformAttn is built via `cd SE-Net/src/pixel_decoder/ops && sh make.sh`, and verified by the
same import the pixel decoder performs (`MultiScaleDeformableAttention` / `MSDeformAttnFunction`)
succeeding. On a GCC version error the documented remedy is `conda install -c conda-forge gxx=9`.

**FR1.4** `tools/eve_senet/check_env.py` reports each of `torch`, `torch.cuda.is_available()`,
`detectron2`, MSDeformAttn, `timm`, `numpy`, `scipy` as `ok` or `FAIL` with the resolved version, prints
the dict as JSON on **stdout** and commentary on stderr, and **exits non-zero** if any required item
fails. Its exit code drives the run script's guard — the same pattern as
`tools/osie_prep/check_features.py`.

**FR1.5** Every run writes a `versions.txt` capturing the resolved stack (`python`, `torch`,
`torchvision`, `numpy`, `scipy`, `detectron2`, `timm`) alongside its outputs (D5, and the point
TechStack §1.1 insists on: F1 proved a hand-written version table stays wrong until a tool reads it).

**FR1.6** F3 **must not** be run on the Windows dev machine. `check_env.py`, `durations.py` and
`verify_embedding.py` are the only parts that run there; anything importing `torch.cuda`, Detectron2 or
MSDeformAttn is cluster-only (TechStack §1).

### FR2 — Inputs and their contracts

**FR2.1** F3 reads exactly these and writes back to none of them:

| input | path (default, repo-root relative) | contract |
|---|---|---|
| `fixations.json` | `data/eve_bridge/fixations.json` | TechStack §3.1 / §3.6 |
| `subject_id_map.json` | `data/eve_bridge/subject_id_map.json` | `{"to_dense": {...}, "to_eve": {...}}` |
| `stimuli/` | `data/eve_bridge/stimuli/` | 877 × 1920×1080 RGB `.jpg` |
| `bridge_report.json` | `data/eve_bridge/bridge_report.json` | D7 counters + resolved args |
| SE-Net checkpoint | `weights/OSIE-20260904T121550Z-1-001/OSIE/ckp_11999.pt` | `{"model": ..., "step": ...}` |
| task embedding | `data/osie_embeddings.npy` | pickled `dict[str, ndarray]`, dim 768 |

**FR2.2** Only records with `split == "train"` are read. This is the F2 support pool: **742 records,
38 subjects, 10–20 records each, disjoint by stimulus name from the scored `test` split**. Reading a
`test` record into the support set silently invalidates every cell of F5's score matrix, so FR11.4
makes it an error rather than a filter.

**FR2.3** The tool asserts, from `bridge_report.json`, that `num_subjects == 38` and
`min_support_per_subject >= num_fewshot` (default 10). It asserts the **minimum**, not
`support_pool_size` — the pools are per-subject and share no image names, so the thinnest pool binds
(TechStack §3.6, OPEN-3).

**FR2.4** The tool asserts `sha256(fixations.json) == bridge_report["fixations_sha256"]`
(`46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b` for the current run). A mismatch
raises — the same safety net `GtHeatmapStore` uses, for the same reason (working convention 10).

### FR3 — Duration binning

**FR3.1** The released SE-Net OSIE checkpoint was trained with `Data.fix_path =
"osie_fixations_update_duration.json"`, whose `T` is a **decile bin index 0–9**, not milliseconds
(TechStack §3.7). EVE's `fixations.json` carries milliseconds. F3 therefore converts EVE durations to
EVE's own decile bins before they reach SE-Net.

**FR3.2** `decile_bins(T_values) -> (edges, bin_of)` computes **10 equal-count buckets** over the
multiset of all per-fixation durations on the **`train` split only** (4,240 fixations in the current
run), using `numpy.quantile` at `q = i/10, i = 0..10`. Bin assignment is
`np.searchsorted(edges[1:-1], t, side="right")`, giving `bin ∈ {0..9}`. The realised edges for the
current bridge run are `[100, 131, 151, 169, 188, 210, 234, 263, 308, 390, 1144]` ms; they are
**computed, never hardcoded**.

**FR3.3** The edges are written to the run's `senet_report.json` as `duration_bin_edges` (list of 11
floats), so F7 can state what EVE's duration distribution was and how it differs from OSIE's (EVE min
100 ms vs OSIE min 20 ms — different distributions, which is exactly why the bins are re-derived rather
than borrowed).

**FR3.4** **The duration channel is dead in the released SE-Net, and this must be pinned, not assumed.**
`SE-Net/src/models.py` L676–678 computes `duration_encoding`, adds it into `ventral_pos`, and then calls
`ventral_pos.fill_(0)` on the next line — `ventral_embs += ventral_pos` has already happened, and
`ventral_pos` is subsequently reused as an accumulator for the indicator embeddings. The duration
therefore never reaches the network. F3 still feeds decile bins (FR3.1: it is the faithful input
contract, it costs nothing, and it is correct if a future SE-Net variant consumes the channel), and
validation asserts that swapping raw milliseconds for bins produces a **bitwise-identical** embedding
tensor (validation Group 5). If that assertion ever fails, the dead path has come alive and every
previously generated embedding must be regenerated.

### FR4 — Support-set selection

**FR4.1** For each dense subject `s ∈ 0..37`, the candidate set is every `train`-split record with
`subject == s`. Records are addressed by their `(name, subject)` key, not by position.

**FR4.2** Exactly `num_fewshot` (default **10**) records are selected per subject, drawn **without
replacement** from that subject's candidate set by a per-subject-seeded RNG:
`rng = random.Random(f"{seed}:{s}")` over the subject's candidate `name`s **sorted lexicographically**,
so the draw is reproducible and independent of dict iteration order. A subject with exactly 10
candidates yields all 10.

> **Why exactly 10 and not "all available".** The paper's comparable row is **n = 10** and F2 was
> deliberately tuned so `num_fewshot = 10` fits every subject (`train23` was dropped to buy it,
> Roadmap F2). Using a subject's whole pool would give some participants 20 shots and others 10, making
> the cohort's embeddings unequally informed and the comparison to the n = 10 row unsound. This is a
> stated assumption, not an inference: if F5 later wants a different n, it is a CLI argument.

**FR4.3** **SE-Net's own `select_fewshot_subject()` is NOT used.** It draws `num_fewshot` image names
from the **union** across the fewshot subjects and keeps whichever subjects have each
(`SE-Net/common/utils.py:1118`). Our support pools are per-subject **disjoint by stimulus**, so a single
union draw of 10 names would give each subject ≈ 10/38 scanpaths, unequally — the failure Roadmap F3
already flags. FR4.2's per-subject draw replaces it.

**FR4.4** The realised selection — `{dense_subject: [name, ...]}`, 38 × 10 names — is written to
`senet_report.json` as `support_selection`, so any embedding row is traceable to the exact scanpaths
that produced it (D5).

**FR4.5** Every selected record must have `split == "train"`. A record with any other split raises
(FR11.4).

### FR5 — Dataset construction

**FR5.1** Fixation preprocessing reuses SE-Net's own functions, imported and not copied:
`common.utils.preprocess_fixations` followed by `common.utils.filter_scanpath`. `filter_scanpath` keeps
only `is_last` labels, i.e. the terminal `stop_label`, whose `dura` field is the record's **full `T`
list** — which is where FR3's bins enter.

**FR5.2** The per-item tensors are produced by **subclassing** `common.data.Siamese_Triplet_Gaze` and
overriding `__getitem__` alone:

```python
class EveSupportDataset(Siamese_Triplet_Gaze):
    def __getitem__(self, idx):
        return {"anchor": self.process_data(idx)}
```

Nothing else is overridden, so image loading, resizing, normalisation, fixation normalisation, padding
and duration truncation are byte-for-byte the authors' own.

> **Why subclassing is mandatory here, not stylistic.** Embedding construction needs only the anchor —
> the positive and negative exist for the **training** triplet loss, and `evaluate_user_siamese` does
> `batch = batch['anchor']` and never reads the other two. But `Siamese_Triplet_Gaze` is the same class
> `builder.py` constructs for both loaders, and its `__getitem__` builds the full triplet regardless of
> eval mode, selecting a negative via
> `random.choice([i for i, d in enumerate(self.fix_labels) if d[-3] != anchor_sid])`. With a single
> subject in the dataset that comprehension is **empty** and `random.choice` raises `IndexError`, so
> per-subject invocation of `train.py --eval-only` is impossible as shipped.
>
> Upstream does have an escape hatch — `num_fewshot == 1` short-circuits both draws (`positive = anchor`,
> `negative = positive`) — but it is gated on the **shot count**, not on eval mode, so at the paper's
> `n = 10` the branch is live. The authors never hit this because OSIE is invoked with all five unseen
> subjects at once (`--fewshot_subject 10 11 12 13 14`), leaving the negative list always non-empty. Our
> support pools are disjoint by stimulus, which forces per-subject invocation (FR4.3), which empties it.
> This is our data shape meeting their sampler, not a defect in either.
>
> The override therefore restores the paper's own semantics for this stage — take the anchor, drop the
> rest — and removes two thirds of the image loads along with the crash.

**FR5.3** Coordinates are rescaled from EVE's native 1920×1080 to SE-Net's `im_w × im_h = 512 × 320`
before `preprocess_fixations`, mirroring `common/dataset.py::process_data`'s rescale block:
`ratio_w = 512/1920 = 0.266667`, `ratio_h = 320/1080 = 0.296296`.

> **This is a second, different squash.** F5's metric screen is 512×384 (4:3, `resizescale = 3.75 /
> 2.8125`, OPEN-4); SE-Net's input is 512×320 (8:5). The two distortions are independent and both must
> reach F7. F3 records `senet_input_size = [320, 512]` and `senet_rescale = [0.266667, 0.296296]` in
> `senet_report.json` so neither is inferred later.

**FR5.4** `process_data` in `common/dataset.py` is **not** called — it raises `NotImplementedError` for
any `Data.name` outside `{OSIE, COCO-Search18, COCO-Freeview, MIT1003, CAT2000}`, and calling it would
also build the val loader F3 has no use for. F3 performs FR5.3's rescale itself and calls
`preprocess_fixations` directly. `Data.name` is set to `"EVE"`; the only place the dataset class
branches on it is the image path, and EVE's `task == "none"` takes the `cat_name == 'none'` branch,
which is the identical `"{image_path}/{img_name}"` form.

**FR5.5** `preprocess_fixations` **silently drops out-of-bound fixations** (`X >= im_w or Y >= im_h or
X < 0 or Y < 0`) for every fixation after the first. EVE coordinates are 1-indexed with `Y` reaching
exactly 1080.0, which rescales to exactly 320.0 and is therefore dropped. F3 **counts** these drops per
subject and reports them as `oob_fixations_dropped` in `senet_report.json` (D7 — counted and surfaced,
never silently folded in). The first fixation of each scanpath is exempt from the bound check by
upstream design and is not counted.

**FR5.6** Scanpaths of length 1 are present in the support split (`min length == 1` in the current run).
They survive `preprocess_fixations` as a single `stop_label` with one fixation and are valid input.
They are counted as `len1_scanpaths` and reported, not dropped.

**FR5.7** `max_traj_length` is SE-Net's `20`; the support split's longest scanpath is 9, so no
truncation occurs. The tool asserts `max(length) <= max_traj_length` and reports `truncated_scanpaths`
(expected `0`).

### FR6 — Model construction and checkpoint loading

**FR6.1** `SE-Net/src/builder.py::build()` is **not** used — it calls `process_data` (FR5.4) and
constructs two DataLoaders. F3 constructs `src.models.UserEmbeddingNet` directly with the same arguments
`build()` passes, read from the config: `num_decoder_layers=6, hidden_dim=384, nhead=4, ntask=1,
num_output_layers=3, train_encoder=False, train_pixel_decoder=True, dropout=0.0, dim_feedforward=512,
num_encoder_layers=3`.

**FR6.2** The checkpoint is loaded as `model.load_state_dict(ckp["model"], strict=False)` — the same
call `build()` makes. **`strict=False` is a live hazard here**: `self.subject_predictor` is an MLP whose
output width is `pa.num_subjects`, so a `num_subjects` differing from the checkpoint's silently leaves
that head randomly initialised. F3 therefore (a) sets `num_subjects` to the checkpoint's own value
(**10**, OSIE's), and (b) **reports the returned `missing_keys` / `unexpected_keys` lists in full** and
raises if any key outside a documented allowlist is missing (FR11.6).

**FR6.3** `num_subjects` is irrelevant to the output F3 needs: `out['user_emb'] = cls_token`
(`SE-Net/src/models.py:757`) is produced before and independently of `subject_predictor`. F3 reads
`user_emb` only and never reads `pred_subject_id`. Keeping `num_subjects = 10` is what makes FR6.2's key
check clean.

**FR6.4** `model.eval()` and `torch.no_grad()` for every forward pass. No optimiser is constructed (D8).

**FR6.5** The task embedding is `list(task_emb_dict.values())[0]` from `data/osie_embeddings.npy` —
upstream `Siamese_Triplet_Gaze.__init__` hardcodes that file for every non-COCO-Search18 dataset and
`process_data` in `common/data.py` discards the key, so the free-viewing task vector is **positional**.
F3 asserts the file exists and that the loaded dict is non-empty, and records `task_emb_source` and the
selected key in `senet_report.json`.

### FR7 — Embedding computation

**FR7.1** For each dense subject `s`, the embedding is the **arithmetic mean of `user_emb` over that
subject's `num_fewshot` support scanpaths**:

```
row[s] = mean_over_k( model(...)['user_emb'][k] )     # k = 1..num_fewshot
```

This reproduces `evaluate_user_siamese`'s accumulate-then-divide
(`class_embeddings.index_add_` followed by `class_embeddings[nonzero] /= class_counts[nonzero]`) for the
single-subject case, without inheriting its two defects: the `num_fewshot == 1` early return that saves
an **un-normalised sum**, and the `drop_last=True` loader.

**FR7.2** **`drop_last` is `False`.** `builder.py` builds the eval loader with `drop_last=True` at
`batch_size // 2 == 8`; a 10-scanpath support set would yield one batch of 8 and **silently discard 2
scanpaths** (20 % of the support evidence). F3's loader uses `drop_last=False`, and asserts that the
number of forward-passed items equals `num_fewshot` exactly (FR11.5).

**FR7.3** The output tensor is `torch.float32` of shape `(38, 384)` — `(num_subjects_eve,
Model.embedding_dim)`. `384` is also `args.subject_feature_dim` on the ISP side, and the two are
asserted equal.

**FR7.4** Row `i` corresponds to dense subject `i`, which is EVE participant
`subject_id_map.json["to_eve"][str(i)]` (D4). Rows are written in ascending dense-id order by
construction — never by dict iteration order.

**FR7.5** **No row may be all-zero.** An all-zero row means a subject produced no forward pass, which
`evaluate_user_siamese` would have silently left as the `torch.zeros` initialiser — exactly what makes
the released `(10, 384)` tensor's rows 5–9 zero. F3 raises (FR11.5).

**FR7.6** Determinism (D5): `--seed` (default 0) seeds `random`, `numpy`, `torch` and CUDA, and pins
`cudnn.deterministic=True` / `benchmark=False` before any forward pass. The seed appears in the output
filename so multiple draws coexist, which is what lets F5 answer OPEN-3's repeat question later without
re-speccing F3.

### FR8 — Outputs

Written into `--out-dir` (default `data/eve_senet/`, git-ignored per working convention 5):

| artefact | contract |
|---|---|
| `eve_fewshot_user_embedding_{num_fewshot}_seed{seed}.pt` | `torch.float32` tensor `(38, 384)`, saved with `torch.save`. This is F5's `--user_emb_path`. |
| `senet_report.json` | resolved args, `duration_bin_edges`, `support_selection`, `senet_input_size`, `senet_rescale`, `task_emb_source`, `embedding_sha256`, `fixations_sha256`, `checkpoint_sha256`, `missing_keys`/`unexpected_keys`, and every D7 counter — always present, `0` when nothing fired |
| `versions.txt` | FR1.5's resolved stack |

**FR8.1** The filename deliberately mirrors the released `fewshot_user_embedding_10.pt` naming so F5's
`--user_emb_path` reads the same way, but is prefixed `eve_` and suffixed with the seed so it can never
be confused with the OSIE tensor on a shared filesystem.

**FR8.2** `senet_report.json` records `embedding_sha256` of the saved tensor's bytes so F5 and F7 can
assert they are discussing the same artefact.

### FR9 — Verification

**FR9.1** `tools/eve_senet/verify_embedding.py` is CPU-only, torch-importing but CUDA-free, and runnable
on a login node or the Windows dev machine. It loads the tensor and `subject_id_map.json` and checks:
shape `(38, 384)`, dtype `float32`, all-finite, no all-zero row, no two identical rows, and that
`len(to_eve) == 38` with keys exactly `{"0".."37"}`.

**FR9.2** It prints per-row `dense_id → eve_id → ||row||₂` so D4's question "which of my real subjects is
row 3?" has a literal answer in an artefact.

**FR9.3** It reports the pairwise cosine-similarity matrix's off-diagonal min/mean/max. This is a
**sanity signal, not a metric**: a cohort whose 38 embeddings are all near-identical would indicate the
model is not discriminating subjects, and F7 needs to know that before interpreting any personalization
claim. No threshold is enforced; the numbers are reported (see validation Data Validity).

**FR9.4** Its **exit code** is non-zero on any FR9.1 failure, so the run script can gate on it.

### FR10 — Frozen-code and convention compliance

**FR10.1** No file under `ISP/*/GazeformerISP/src/utils/` is read, imported or modified (D1). F3 does not
touch the ISP tree.

**FR10.2** No file under `SE-Net/` is modified. `tools/eve_senet/` imports from it (`common.utils`,
`common.data`, `common.config`, `src.models`) and subclasses it. Validation Group 6 asserts this by
comparing tracked-file hashes before and after a run.

**FR10.3** `tools/eve_senet/` does not import `tools/eve_bridge/` code — it reads the bridge's
*artefacts* (D2). The two packages share no module.

**FR10.4** No new heavy dependency (working convention 7). `tools/eve_senet/` imports only what the
`senet` env already provides plus stdlib; `durations.py` is numpy-only and runs on Windows.

**FR10.5** Any `.pyc` dropped into `SE-Net/` directories by our imports is deleted after the run
(working convention 6 — the 89 tracked `__pycache__` directories stay, new artefacts do not).

### FR11 — Error conditions (D7 — fail loudly)

| # | condition | behaviour |
|---|---|---|
| **FR11.1** | `fixations.json` sha256 ≠ `bridge_report["fixations_sha256"]` | raise `EveSenetError` |
| **FR11.2** | `min_support_per_subject < num_fewshot` | raise `EveSenetError`, naming the thinnest subject |
| **FR11.3** | a stimulus named in a selected record is absent from `stimuli/` | raise `EveSenetError` with the name |
| **FR11.4** | a selected record has `split != "train"` | raise `EveSenetError` — a support/query leak, never a filter |
| **FR11.5** | forward-passed item count ≠ `num_fewshot` for any subject, or any output row is all-zero | raise `EveSenetError` with the subject's dense and EVE ids |
| **FR11.6** | `load_state_dict` returns a missing key outside the documented allowlist | raise `EveSenetError` listing the keys |
| **FR11.7** | `subject_id_map.json` `to_eve` key set ≠ `{"0".."37"}`, or disagrees with `fixations.json`'s dense subject set | raise `EveSenetError` |
| **FR11.8** | embedding dim ≠ 384, or `max(length) > max_traj_length` | raise `EveSenetError` |
| **FR11.9** | `check_env.py` reports any required import as FAIL | non-zero exit; the run script aborts before any GPU work |
| — | out-of-bound fixation dropped (FR5.5); length-1 scanpath (FR5.6) | **counted and reported**, not raised — these are legitimate data, and silence is the failure mode |

---

## Public API Summary

```python
# tools/eve_senet/__init__.py
class EveSenetError(RuntimeError): ...

# tools/eve_senet/durations.py           (numpy only — Windows-runnable, unit-testable)
def decile_bins(durations: Sequence[float]) -> tuple[np.ndarray, Callable[[float], int]]:
    """10 equal-count buckets. Returns (edges (11,) float64, bin_of)."""

def bin_scanpath_durations(records: list[dict], bin_of) -> list[dict]:
    """Returns copies whose T (ms) is replaced by its decile index 0..9."""

# tools/eve_senet/dataset.py             (imports SE-Net; cluster-only)
class EveSupportDataset(Siamese_Triplet_Gaze):
    def __getitem__(self, idx) -> dict: ...   # {"anchor": self.process_data(idx)}

def build_fix_labels(
    records: list[dict], im_h: int = 320, im_w: int = 512,
    max_traj_length: int = 20,
) -> tuple[list, dict]:
    """Rescale -> preprocess_fixations -> filter_scanpath. Returns (fix_labels, counters)."""

# tools/eve_senet/embed.py               (cluster-only)
def select_support(
    records: list[dict], num_fewshot: int, seed: int,
) -> dict[int, list[str]]: ...

def embed_subject(
    model, records: list[dict], pa, device, image_dir: str,
) -> tuple[torch.Tensor, dict]:
    """Returns (user_emb mean (384,), counters)."""

def build_embeddings(
    fixations_path: str, subject_map_path: str, report_path: str,
    image_dir: str, checkpoint: str, config: str, out_dir: str,
    num_fewshot: int = 10, seed: int = 0, device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """Full pipeline. Returns ((38, 384) tensor, senet_report dict)."""

# tools/eve_senet/check_env.py           — __main__, JSON on stdout, exit code
# tools/eve_senet/verify_embedding.py    — __main__, CPU-only, exit code
```

### CLI

```bash
# preflight (login node)
py tools/eve_senet/check_env.py

# the run (GPU node, senet env)
python tools/eve_senet/embed.py \
    --fixations   data/eve_bridge/fixations.json \
    --subject-map data/eve_bridge/subject_id_map.json \
    --report      data/eve_bridge/bridge_report.json \
    --image-dir   data/eve_bridge/stimuli \
    --checkpoint  weights/OSIE-20260904T121550Z-1-001/OSIE/ckp_11999.pt \
    --config      SE-Net/configs/eve_useremb.json \
    --out-dir     data/eve_senet \
    --num-fewshot 10 --seed 0

# verification (anywhere, CPU)
py tools/eve_senet/verify_embedding.py \
    --embedding   data/eve_senet/eve_fewshot_user_embedding_10_seed0.pt \
    --subject-map data/eve_bridge/subject_id_map.json

# one documented command
bash bash/embed_eve_subjects.sh          # SEED=0; SEED=1 bash ... for a second draw
```

---

## Dependencies

| direction | item | what F3 does with it |
|---|---|---|
| reads | `data/eve_bridge/fixations.json` | `train`-split records only (FR2.2); sha256-checked (FR2.4) |
| reads | `data/eve_bridge/subject_id_map.json` | D4 row-order authority (FR7.4) |
| reads | `data/eve_bridge/stimuli/*.jpg` | model input, resized to 512×320 (FR5.3) |
| reads | `data/eve_bridge/bridge_report.json` | `num_subjects`, `min_support_per_subject`, `fixations_sha256` (FR2.3–2.4) |
| reads | `weights/OSIE-.../OSIE/ckp_11999.pt` | released SE-Net weights (FR6.2) |
| reads | `data/osie_embeddings.npy` | positional free-viewing task vector (FR6.5) |
| imports | `SE-Net/common/{utils,data,config}.py`, `SE-Net/src/models.py` | unmodified; subclassed (FR10.2) |
| **never touches** | `ISP/*/GazeformerISP/src/utils/**` | D1 (FR10.1) |
| **never touches** | `SE-Net/**` | working convention 2 (FR10.2) |
| **never touches** | `data/eve_bridge/**` | pure reader; working convention 10 |
| writes | `data/eve_senet/eve_fewshot_user_embedding_10_seed0.pt` | **F5's `--user_emb_path`** |
| writes | `data/eve_senet/senet_report.json` | F5 asserts against it; F7 quotes it |
| writes | `data/eve_senet/versions.txt` | D5 |
| creates | `SE-Net/configs/eve_useremb.json` | new config file, additive; no existing config edited |
| unblocks | **F5** (with F4) | Stage C complete |
| colours | **F7** | OPEN-2 resolved toward option 1; the SE-Net squash (FR5.3); the dead duration channel (FR3.4) |
