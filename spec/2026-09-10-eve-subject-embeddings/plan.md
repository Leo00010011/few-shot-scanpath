# F3 — Implementation plan

> Spec 2 of 3. Read with [requirements.md](requirements.md) and [validation.md](validation.md).
> Created: 2026-09-10

---

## Context and Design Decisions

### Why we generate our own embeddings (OPEN-2 → option 1)

The released `fewshot_user_embedding_10.pt` is a `(10, 384)` tensor whose rows 0–4 carry OSIE subjects
10–14 and whose rows 5–9 are exactly zero (verified during F1). It cannot address 38 subjects, and even
tiled or truncated it would encode *somebody else's* viewing behaviour. The mission's whole claim is that
prediction `i` is meaningful because it is personalised to subject `i` (P2); borrowing an embedding
severs that link and reduces F5 to measuring how well an OSIE viewer's habits predict an EVE viewer's.
We therefore pay the Detectron2 + MSDeformAttn install cost and produce genuine embeddings. **F7 still
inherits an honesty obligation**: the SE-Net *checkpoint* is OSIE-trained even though the embeddings are
ours, so what is transferred is the encoder, not the subjects.

### Why a new `tools/eve_senet/` package rather than `train.py --eval-only`

Three independent blockers make the shipped invocation path unusable for our data, and all three are
call-site problems rather than upstream defects to fix:

1. **`process_data` has no EVE branch.** `common/dataset.py` maps a dataset name to `(ori_h, ori_w)` and
   raises `NotImplementedError` otherwise. Adding a branch would edit a shared upstream file — working
   convention 2 says prefer additive.
2. **Single-subject invocation crashes.** Embedding construction needs only the anchor — the positive and
   negative serve the *training* triplet loss, and `evaluate_user_siamese` reads `batch['anchor']` alone.
   But `Siamese_Triplet_Gaze` is shared between the train and val loaders and its `__getitem__` builds the
   full triplet regardless of mode, drawing a negative from a different subject; with one subject in the
   dataset that candidate list is empty and `random.choice` raises `IndexError`. The `num_fewshot == 1`
   short-circuit upstream would avoid it, but it is gated on the shot count rather than on eval mode, so
   at the paper's `n = 10` the draw is live. The authors never hit this because OSIE is invoked with all
   five unseen subjects at once; our pools are disjoint by stimulus, which forces per-subject invocation
   (otherwise `select_fewshot_subject`'s union draw gives each subject ≈ 10/38 scanpaths), which empties
   the list. Per-subject invocation and the shipped sampler are mutually exclusive.
3. **`drop_last=True` at `bs=8` eats the tail of a 10-scanpath support set.** `builder.py` builds the
   eval loader with `drop_last=True` and `batch_size // 2`; 10 items yield one batch of 8, discarding
   20 % of the evidence for the subjects at `min_support_per_subject`. Silent, plausible, wrong — the
   exact D7 failure mode.

A thin driver of our own resolves all three by **composition**: we call the authors' `preprocess_fixations`
and `filter_scanpath`, we *subclass* `Siamese_Triplet_Gaze` so its `process_data()` (image load, resize,
normalise, pad) stays byte-for-byte theirs, and we build the model class directly rather than through
`build()`. Nothing under `SE-Net/` is edited. This mirrors how `tools/eve_bridge/` handled the same
tension for `OSIE.__getitem__` (re-derive, then pin with a parity test).

### Why decile-bin the durations, given the channel is dead

`SE-Net/src/models.py` L676–678:

```python
ventral_embs += ventral_pos                                    # (a) pos is folded in HERE
duration_encoding = get_duration_positional_encoding(...)      # (b) computed
ventral_pos += duration_encoding                               # (c) added to ventral_pos
ventral_pos.fill_(0)                                           # (d) ...and wiped on the next line
```

`ventral_pos` is then reused as an accumulator for `ventral_ind_emb` and `fix_ind_emb`. Between (c) and
(d) nothing reads `ventral_pos`, so the duration never reaches the network. The same
add-then-`fill_(0)` shape appears two lines earlier for `dorsal_pos`, where it is deliberate (the tensor
is being recycled); for duration it is almost certainly not.

We bin anyway, for three reasons: it is the input contract the checkpoint was trained under
(`osie_fixations_update_duration.json`, TechStack §3.7), it costs one numpy call, and it is what a
corrected SE-Net would need. What we do **not** do is assume the deadness — validation Group 5 asserts
raw-ms and binned runs produce a **bitwise-identical** tensor. That converts an observation into a
regression test: if a future env or checkpoint makes the channel live, the test fires and every
embedding produced under the assumption is invalidated. Reporting the bin edges (FR3.3) means F7 can
state EVE's duration distribution regardless.

### Why exactly 10 shots per subject

F2 dropped participant `train23` specifically to raise `min_support_per_subject` from 9 to 10, buying the
paper's `num_fewshot = 10` row (Roadmap F2). Honouring that means every subject contributes exactly 10
scanpaths — using each subject's whole pool would give some 20 and others 10, so the cohort's rows would
be unequally informed and the comparison to the n = 10 row would not hold. The draw is seeded per
subject so a second seed produces a genuinely different support set inside the same pool, which is what
`support_pool_size = 20 > num_fewshot = 10` was engineered for (OPEN-3).

### Constitution constraints that shape the code

- **D1** — F3 never reads or imports the frozen files. It does not touch the ISP tree at all.
- **D4** — row order is `subject_id_map.json`'s dense ids, ascending, asserted twice: at write time
  (FR7.4) and by an independent verifier (FR9).
- **D5** — seed pinned, versions recorded from the resolved stack rather than transcribed, every input
  hashed, the support selection itself written out so a row is traceable to its scanpaths.
- **D7** — out-of-bound drops and length-1 scanpaths are *counted*, not swallowed; every other anomaly
  raises.
- **D8** — no optimiser is ever constructed.
- **Working convention 5** — `data/eve_senet/` is git-ignored; it holds subject-level derived data.
- **Working convention 6** — new `.pyc` under `SE-Net/` is cleaned up after each run.

---

## Implementation Steps

### Step 1 — Build the `senet` environment (cluster, GPU node with `nvcc`)

No repo files change. Record every deviation for D5.

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00

conda env create -f SE-Net/environment.yml -n senet     # FR1.1
conda activate senet
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the solve fails or is unsatisfiable on this base image, build minimally from TechStack §1's `senet`
table (python 3.8.0, pytorch 1.11.0 cu113, torchvision 0.12.0, numpy 1.23.5, scipy 1.10.0, timm 0.6.13,
opencv-python 4.7.0.72, `multimatch-gaze==0.1.3`) and record the resolved `conda list` / `pip freeze` as
an FR1.1 deviation.

> `multimatch-gaze` is not on F3's code path (no metric is computed here) but is pinned in the env for
> parity with `isp`; install it with `--no-deps` so pip cannot move numpy/scipy underneath the env, the
> lesson TechStack §1.1 records from F1.

### Step 2 — Detectron2 from source (FR1.2)

```bash
conda activate senet
python -c "import torch; print(torch.version.cuda)"     # expect 11.3
gcc --version                                            # need a version torch 1.11/cu113 accepts
# on a version error:  conda install -c conda-forge gxx=9

git clone https://github.com/facebookresearch/detectron2 "$SCRATCH/detectron2"
python -m pip install -e "$SCRATCH/detectron2"
python -c "import detectron2; print(detectron2.__version__)"
```

Clone **outside the repo** (`$SCRATCH` or `work/`, which is git-ignored) — Detectron2 is a third-party
source tree, not project code.

### Step 3 — MSDeformAttn (FR1.3)

```bash
conda activate senet
cd SE-Net/src/pixel_decoder/ops && sh make.sh && cd -
```

`make.sh` compiles in place inside the repo. `SE-Net/src/pixel_decoder/ops/build/` and any `*.so` it
produces are **build artefacts, not source** — confirm `.gitignore` covers them (add `*.so`,
`**/ops/build/`, `*.egg-info/` if not) before the first `git status`. This is the one step that writes
into the `SE-Net/` tree, and it is a compiler output, not an edit to a tracked file; validation Group 6
hashes tracked files only, so it does not trip the frozen-tree check.

Verify with whichever import the pixel decoder performs, then guard against a stale build:

```bash
python -c "from src.pixel_decoder.ops.functions import MSDeformAttnFunction; print('ok')"
```

### Step 4 — `tools/eve_senet/__init__.py` and `check_env.py`

*Depends on nothing; write first so Steps 1–3 can be verified by tooling rather than by eye.*

`__init__.py`:

```python
class EveSenetError(RuntimeError):
    """Raised on any F3 invariant violation (FR11)."""
```

`check_env.py` (FR1.4) — stdlib + optional imports only, so it runs even in a broken env:

```python
CHECKS = [                       # (label, import expression, required?)
    ("python",        sys.version,                         True),
    ("torch",         "torch.__version__",                 True),
    ("cuda_available","torch.cuda.is_available()",         True),
    ("torchvision",   "torchvision.__version__",           True),
    ("numpy",         "numpy.__version__",                 True),
    ("scipy",         "scipy.__version__",                 True),
    ("timm",          "timm.__version__",                  True),
    ("detectron2",    "detectron2.__version__",            True),
    ("msdeformattn",  <the pixel_decoder import>,          True),
]
# each wrapped in try/except -> {"status": "ok"|"FAIL", "detail": version or repr(exc)}
# JSON dict -> stdout;  human commentary -> stderr;  sys.exit(1) if any required FAIL
```

`versions.txt` (FR1.5) is produced by the same resolution code, so the record cannot disagree with the
check — the §1.1 lesson made structural.

### Step 5 — `tools/eve_senet/durations.py`

*Depends on Step 4. Pure numpy — developed and unit-tested on Windows.*

```python
def decile_bins(durations):
    d = np.asarray(list(durations), dtype=np.float64)
    if d.size == 0:
        raise EveSenetError("no durations to bin")
    edges = np.quantile(d, [i / 10 for i in range(11)])       # (11,) FR3.2
    inner = edges[1:-1]
    def bin_of(t):
        return int(np.searchsorted(inner, float(t), side="right"))
    return edges, bin_of

def bin_scanpath_durations(records, bin_of):
    """Copy each record with T (ms) replaced by decile index 0..9. Never in place —
    the caller may still need the milliseconds for the report."""
```

Ties at an edge collapse into one bucket, so counts need not be exactly equal; the function does **not**
force equal counts. Edges are returned for FR3.3 and asserted monotone non-decreasing.

### Step 6 — `SE-Net/configs/eve_useremb.json`

*Depends on nothing. A new file — no existing config is edited (FR10.2).*

Copied from `osie_useremb.json` with only these fields changed:

| field | OSIE | EVE | why |
|---|---|---|---|
| `Data.name` | `"OSIE"` | `"EVE"` | FR5.4 — steers the image-path branch to the `cat_name == 'none'` form |
| `Data.fix_path` | `osie_fixations_update_duration.json` | `""` | F3 loads fixations itself; `build()` is not used |
| `Data.image_path` | `data/OSIE` | set from `--image-dir` at runtime | never hardcode a path (working convention 4) |
| `Data.num_fewshot` | `10` | `10` | unchanged; the paper's n |
| `Data.num_subjects` | `10` | `10` | **deliberately unchanged** — FR6.2/6.3: it sizes `subject_predictor`, whose weights come from the checkpoint, and F3 never reads that head |

Everything else — `im_w: 512`, `im_h: 320`, `patch_num: [32, 20]`, `patch_size: [16, 16]`,
`max_traj_length: 20`, `TAP: "FV"`, `pad_idx: 0`, `Model.embedding_dim: 384`, `Model.n_dec_layers: 6`,
`Model.n_enc_layers: 3`, `Model.hidden_dim: 512`, `Model.n_heads: 4`, `Model.checkpoint: "ckp_11999.pt"`
— is left **exactly** as OSIE's, because those values are what the released checkpoint's weights were
shaped by. Changing any of them silently mismatches the state dict under `strict=False`.

### Step 7 — `tools/eve_senet/dataset.py`

*Depends on Steps 4–6. Imports SE-Net; cluster-only.*

```python
sys.path.insert(0, <repo>/SE-Net)          # SE-Net expects to be importable as its own root
from common.utils import preprocess_fixations, filter_scanpath
from common.data import Siamese_Triplet_Gaze


class EveSupportDataset(Siamese_Triplet_Gaze):
    """FR5.2 — anchor only. No triplet sampling: with one subject the negative
    candidate list is empty and random.choice raises IndexError."""
    def __getitem__(self, idx):
        return {"anchor": self.process_data(idx)}


def build_fix_labels(records, im_h=320, im_w=512, max_traj_length=20):
    counters = {"oob_fixations_dropped": 0, "len1_scanpaths": 0,
                "truncated_scanpaths": 0, "n_records": len(records)}

    ratio_w, ratio_h = im_w / 1920.0, im_h / 1080.0            # FR5.3
    scaled = []
    for r in records:
        if r["split"] != "train":                              # FR11.4
            raise EveSenetError(...)
        if r["length"] > max_traj_length:                       # FR11.8
            raise EveSenetError(...)
        if r["length"] == 1:
            counters["len1_scanpaths"] += 1
        s = dict(r)
        s["X"] = np.asarray(r["X"], float) * ratio_w
        s["Y"] = np.asarray(r["Y"], float) * ratio_h
        s["rescaled"] = True
        scaled.append(s)

    # count what preprocess_fixations will silently drop (FR5.5) BEFORE calling it,
    # so the counter is independent of upstream behaviour rather than inferred from it
    for s in scaled:
        for i in range(1, len(s["X"])):
            if not (0 <= s["X"][i] < im_w and 0 <= s["Y"][i] < im_h):
                counters["oob_fixations_dropped"] += 1

    fix_labels = preprocess_fixations(
        scaled,
        patch_size=[16, 16], patch_num=[32, 20],
        im_h=im_h, im_w=im_w,
        truncate_num=max_traj_length,
        has_stop=True,                    # required: filter_scanpath keeps only is_last
        sample_scanpath=False,
        min_traj_length_percentage=0,
        discretize_fix=False,
        remove_return_fixations=False,
        is_coco_dataset=False,            # EVE keeps its own first fixation
    )
    return filter_scanpath(fix_labels), counters
```

Two upstream behaviours this deliberately relies on and must not "fix":

- `has_stop=True` is what makes `filter_scanpath` return anything at all — it keeps only `is_last`
  labels, and the `stop_label` is the only one flagged `is_last`. Its `dura` slot holds `traj['T']`
  entire, which is where the FR3 bins enter.
- `is_coco_dataset=False` prevents the first fixation being forced to the image centre. EVE is
  free-viewing with a real first fixation; forcing the centre is a COCO-Search18 convention.

Assert `len(fix_labels) == len(records)` — one terminal label per scanpath — and raise otherwise.

### Step 8 — `tools/eve_senet/embed.py`, part 1: selection and model

*Depends on Step 7.*

```python
def select_support(records, num_fewshot, seed):
    by_subject = defaultdict(list)
    for r in records:
        by_subject[r["subject"]].append(r)
    out = {}
    for s in sorted(by_subject):
        names = sorted(r["name"] for r in by_subject[s])        # FR4.2 — deterministic order
        if len(names) < num_fewshot:                            # FR11.2
            raise EveSenetError(f"subject {s} has {len(names)} < {num_fewshot}")
        rng = random.Random(f"{seed}:{s}")
        out[s] = sorted(rng.sample(names, num_fewshot))
    return out


def load_model(config, checkpoint, device):
    pa = JsonConfig(config).Data ; mp = JsonConfig(config).Model ; tp = JsonConfig(config).Train
    model = UserEmbeddingNet(
        pa,
        num_decoder_layers=mp.n_dec_layers, hidden_dim=mp.embedding_dim,
        nhead=mp.n_heads, ntask=1, num_output_layers=mp.num_output_layers,
        train_encoder=tp.train_backbone, train_pixel_decoder=tp.train_pixel_decoder,
        dropout=tp.dropout, dim_feedforward=mp.hidden_dim,
        num_encoder_layers=mp.n_enc_layers,
    ).to(device)
    ckp = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckp["model"], strict=False)   # FR6.2
    # allowlist: keys the released checkpoint legitimately omits, established by
    # running once and inspecting. Anything outside it raises (FR11.6).
    model.eval()
    return model, pa, list(missing), list(unexpected)
```

> **Establishing the allowlist is a real step, not a formality.** Run `load_model` once, print
> `missing_keys`, and decide per key whether the checkpoint genuinely does not carry it (e.g. a
> backbone buffer rebuilt at construction) or whether our construction arguments are wrong. A missing
> `subject_predictor.*` would mean `num_subjects` disagrees with the checkpoint and must be fixed, not
> allowlisted. Write the resolved list into the source with a comment naming why each key is there.

### Step 9 — `tools/eve_senet/embed.py`, part 2: per-subject forward pass

*Depends on Step 8.*

```python
@torch.no_grad()
def embed_subject(model, records, pa, device, image_dir):
    fix_labels, counters = build_fix_labels(records, pa.im_h, pa.im_w, pa.max_traj_length)
    ds = EveSupportDataset(root_dir=<repo>/data, fix_labels=fix_labels, bbox_annos={},
                           pa=pa, transform=transform, catIds={"none": 0},
                           device=device, blur_action=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=True)   # FR7.2

    embs, n_seen = [], 0
    for batch in loader:
        b = batch["anchor"]
        inp_seq, inp_seq_high = transform_fixations(
            b["normalized_fixations"], b["is_padding"], pa, False, return_highres=True)
        logits = model(b["true_state"].to(device), inp_seq.to(device),
                       (inp_seq == pa.pad_idx).to(device), inp_seq_high.to(device),
                       b["duration"].to(device), b["task_emb"].to(device))
        embs.append(logits["user_emb"].detach().cpu())
        n_seen += b["true_state"].size(0)

    if n_seen != len(records):                                   # FR11.5
        raise EveSenetError(...)
    row = torch.cat(embs, dim=0).mean(dim=0)                     # FR7.1
    if not torch.isfinite(row).all() or bool((row == 0).all()):  # FR11.5
        raise EveSenetError(...)
    return row.to(torch.float32), counters
```

`transform` is built once and reused: `Resize((pa.im_h, pa.im_w))`, `ToTensor()`, ImageNet
`Normalize` — identical to `process_data`'s `transform_test`. `pa.image_path` is set from `--image-dir`
before the dataset is constructed, since `Siamese_Triplet_Gaze.process_data` reads it directly.

`num_workers=0` is deliberate: the support sets are 10 items, worker startup dominates, and a single
process keeps the seeded draw and any exception traceable.

### Step 10 — `tools/eve_senet/embed.py`, part 3: the pipeline and CLI

*Depends on Step 9.*

```python
def build_embeddings(...):
    set_seeds(seed)                                    # random, numpy, torch, cuda; cudnn deterministic
    fixations   = json.load(...)                       # FR2.1
    assert_sha256(fixations_path, report["fixations_sha256"])          # FR2.4 / FR11.1
    subject_map = json.load(...)
    assert set(subject_map["to_eve"]) == {str(i) for i in range(report["num_subjects"])}   # FR11.7

    train = [r for r in fixations if r["split"] == "train"]            # FR2.2
    assert report["min_support_per_subject"] >= num_fewshot             # FR2.3 / FR11.2

    edges, bin_of = decile_bins([t for r in train for t in r["T"]])     # FR3.2
    binned = {(r["name"], r["subject"]): r
              for r in bin_scanpath_durations(train, bin_of)}

    selection = select_support(train, num_fewshot, seed)                # FR4
    assert_stimuli_exist(selection, image_dir)                          # FR11.3

    model, pa, missing, unexpected = load_model(config, checkpoint, device)
    pa.image_path = image_dir
    assert pa.__dict__.get("embedding_dim", 384) == 384                 # FR11.8

    rows, counters = [], Counter()
    for s in range(report["num_subjects"]):                             # FR7.4 — ascending, always
        recs = [binned[(n, s)] for n in selection[s]]
        row, c = embed_subject(model, recs, pa, device, image_dir)
        rows.append(row); counters.update(c)

    emb = torch.stack(rows, dim=0)                                      # (38, 384) FR7.3
    torch.save(emb, out_path)
    write_report(...)   # FR8: args, edges, selection, squash, hashes, key lists, counters
    write_versions(...) # FR1.5
    return emb, report_dict
```

CLI flags exactly as requirements' **CLI** block. `--device` defaults to `cuda` and refuses `cpu`
unless `--allow-cpu` is passed, so a silently CPU-bound run cannot be mistaken for a normal one.

### Step 11 — `tools/eve_senet/verify_embedding.py`

*Depends on Step 10's artefact format; can be written in parallel. CPU-only, Windows-runnable.*

Implements FR9.1–9.4. Prints a `dense_id → eve_id → ||row||₂` table and the off-diagonal cosine
min/mean/max, JSON on stdout, table on stderr, non-zero exit on any structural failure.

### Step 12 — `bash/embed_eve_subjects.sh`

*Depends on Steps 4–11. Modelled on `bash/test_osie.sh`.*

- `set -euo pipefail`; **`bash script`, never `source script`** (TechStack §1 command conventions —
  under `source`, a failed preflight kills the login shell and drops the allocation).
- Every tunable overridable from the environment: `SENET_ENV`, `SEED`, `NUM_FEWSHOT`, `BRIDGE_DIR`,
  `OUT_DIR`, `CKPT`, `CONFIG`.
- Order: `conda activate "$SENET_ENV"` → `check_env.py` (exit code gates, FR11.9) → `embed.py` teed to
  `$OUT_DIR/seed$SEED/stdout.txt` → `verify_embedding.py` (exit code gates) → clean new `.pyc` under
  `SE-Net/` (FR10.5).
- Per-seed artefacts under `$OUT_DIR/seed$SEED/` so a second seed cannot overwrite the first — the
  lesson `bash/test_osie.sh` learned when three seeds landed on one filename (TechStack §3.5a).
- `#SBATCH` block retained and inert when executed directly, matching `test_osie.sh`.

### Step 13 — Tests (`tests/eve_senet/`)

*Depends on Steps 5, 7, 8. Written alongside, run on Windows CPU.*

`py -m pytest tests/eve_senet -q`. Self-contained fixtures — synthetic records, a 4×4 stub image
directory — and no cluster data, mirroring `tests/osie_prep/`. Groups 1, 2, 3 and 6 of
[validation.md](validation.md) are pytest; Groups 4, 5 and Data Validity need the GPU and run on the
cluster.

### Step 14 — Run, verify, and record

*Depends on everything.*

1. `bash bash/embed_eve_subjects.sh` under `salloc`, seed 0.
2. Inspect `senet_report.json`: bin edges, counters, missing/unexpected keys, support selection.
3. `SEED=1 bash bash/embed_eve_subjects.sh` — a second draw from the same pools, which is the artefact
   F5 needs if OPEN-3's repeat question resolves toward averaging.
4. Run validation Group 5 (the duration-deadness pin) and Data Validity once, on the cluster.
5. Update Roadmap F3's checklist and, if anything surprising surfaced, add a `notes.md` to this spec
   folder — the F2 convention, not F1's generated-record one, since F3's findings are structural rather
   than numeric.

---

## Implementation Order

1. **Step 1** — `senet` env from `SE-Net/environment.yml`
2. **Step 2** — Detectron2 from source
3. **Step 3** — MSDeformAttn `make.sh`
4. **Step 4** — `__init__.py` + `check_env.py` (verifies 1–3 by tool, not by eye)
5. **Step 5** — `durations.py` (numpy only, Windows)
6. **Step 6** — `SE-Net/configs/eve_useremb.json` (new file)
7. **Step 7** — `dataset.py` (`EveSupportDataset`, `build_fix_labels`)
8. **Step 8** — `embed.py` selection + model loading; **establish the missing-key allowlist**
9. **Step 9** — `embed.py` per-subject forward pass
10. **Step 10** — `embed.py` pipeline + CLI
11. **Step 11** — `verify_embedding.py`
12. **Step 12** — `bash/embed_eve_subjects.sh`
13. **Step 13** — `tests/eve_senet/`
14. **Step 14** — run seeds 0 and 1, verify, record

Steps 4, 5, 11 and the pytest half of 13 are Windows-developable and do not wait on Steps 1–3. Step 8's
allowlist is the only step that needs a real GPU forward pass before its code can be finalised.
