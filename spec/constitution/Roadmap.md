# Roadmap

> Constitution file 3 of 3. Read together with [Mission.md](Mission.md) and [TechStack.md](TechStack.md).
> Last updated: 2026-09-07
>
> **Current phase: F1 — Reproduce the OSIE baseline.** Nothing else starts until F1 produces numbers.

---

## 1. Status board

| ID | Feature | Status | Blocked by |
|---|---|---|---|
| F0 | Constitution + spec workflow | ✓ DONE (2026-09-07) | — |
| F1 | Reproduce OSIE eval baseline on the cluster | ▶ NEXT | — |
| F2 | Dataset bridge: our data → `fixations.json` | ⏸ TODO | F1, **OPEN-1** |
| F3 | Subject embeddings for our subjects | ⏸ TODO | F2, **OPEN-2** |
| F4 | Feature extraction for our stimuli | ⏸ TODO | F2 |
| F5 | Our-dataset eval branch + run script | ⏸ TODO | F2, F3, F4 |
| F6 | Offline re-scorer (`prediction.json` → metrics) | ⏸ TODO | F1 |
| F7 | Results write-up + validity statement | ⏸ TODO | F5, F6 |

Legend: ✓ DONE · ▶ IN PROGRESS/NEXT · ⏸ TODO · ✗ DROPPED

---

## 2. Dependency graph

```
        F0 constitution
             │
             ▼
        F1 OSIE baseline ──────────────┐
             │                         │
             │  (proves env + metrics) │
             ▼                         ▼
   OPEN-1 ─► F2 data bridge        F6 offline re-scorer
             │                         │
      ┌──────┴────────┐                │
      ▼               ▼                │
 F4 features    OPEN-2 ─► F3 subj emb  │
      │               │                │
      └───────┬───────┘                │
              ▼                        │
        F5 eval branch ────────────────┤
              │                        │
              └────────────┬───────────┘
                           ▼
                   F7 write-up
```

---

## 3. Features

### F1 — Reproduce the OSIE eval baseline on the cluster ▶ NEXT

**Why first:** it is the only way to prove the environment, the checkpoints, and the frozen metric code all
work *before* our data is in the picture. Every later failure then has an unambiguous cause.

- [ ] Build the `isp` conda env on the cluster from `ISP/environment.yml`; record what deviates from the
      pin list in [TechStack.md](TechStack.md) §1.
- [ ] Stage the OSIE stimuli (800×600 `.jpg`) into the path `--img_dir` points at.
- [ ] Place `weights/OSIE-*/OSIE/checkpoint_best.pth` at
      `ISP/OSIE/GazeformerISP/src/assets/OSIE-ex-10to15/checkpoints/checkpoint_best.pth`, and the
      `*_user_embedding.pt` files where `--user_emb_path` expects them. **Verify actual on-disk case**
      of `OSIE-ex-10to15` vs `osie-ex-10to15` and `user` vs `subject` in the embedding filenames.
- [ ] Run Stage B (`preprocess/feature_extractor.py`) to produce `src/data/image_features/*.pth` and
      `src/data/embeddings.npy`.
- [ ] Run the query-set demo:
      `CUDA_VISIBLE_DEVICES=0 python src/test.py --fewshot_subject 10 11 12 13 14`
- [ ] Record SM / MM / SED and the full metric block; compare against the paper's OSIE numbers.
- [ ] **Decide and document** what to do about the `if i_batch > 100: break` cap in `test.py`
      (TechStack §5) — is the published number computed over 101 images or the full test split?
- [ ] Confirm the shipped `result/git-osie-useremb-ex-10to15/log/prediction.json` matches the schema our
      re-scorer (F6) will assume.

**Done when:** OSIE numbers are reproduced within sampling noise and the exact command + env are recorded.

---

### F2 — Dataset bridge: our data → `fixations.json` ⏸

Depends on **OPEN-1**. Implements Stage A; satisfies D2, D3, D4, D7.

- [ ] Resolve OPEN-1 (integration strategy for the external dataloader).
- [ ] Write the converter producing a `fixations.json` conforming byte-for-byte to the schema in
      [TechStack.md](TechStack.md) §3.1.
- [ ] Emit an explicit `subject_id_map.json` (our real subject ids ↔ dense `0..N-1`) — D4.
- [ ] Record our stimulus resolution and carry it into `origin_size=(H, W)`; do **not** rely on the
      `OSIE_evaluation` default `(600, 800)` (D3).
- [ ] Validator (runnable on Windows/CPU): every image has the same subject count; `len(X)==len(Y)==len(T)
      ==length`; coordinates inside `[1, W]` / `[1, H]`; `T` positive integer ms; every `name` resolves to a
      file on disk; report the count of scanpaths shorter than 3 fixations (they get padded — D7).
- [ ] Define the `split` assignment for our subjects and record the rationale.

---

### F3 — Subject embeddings for our subjects ⏸

Depends on **OPEN-2**. Implements Stage C.

- [ ] Resolve OPEN-2 (whose subject embeddings do our subjects get?).
- [ ] If generating our own: build the `senet` env including Detectron2 + MSDeformAttn (the highest-risk
      install in the project — [TechStack.md](TechStack.md) §1), add a config under `SE-Net/configs/`, run
      `train.py --eval-only --fewshot_subject ...`.
- [ ] If reusing a released embedding: document precisely which subjects' embeddings are being borrowed
      and what that does to the interpretation of the numbers (feeds F7).
- [ ] Verify the produced tensor's shape matches `args.subject_feature_dim = 384` and its row order matches
      the dense subject indices from F2.

---

### F4 — Feature extraction for our stimuli ⏸

- [ ] Point `feature_extractor.image_data()` at our stimuli (note the hardcoded `<dataset_path>/train/`
      subdirectory) and produce one `.pth` per image.
- [ ] Confirm each tensor is `(768, 2048)`; a different stimulus aspect ratio still resizes to 768×1024, so
      record whether our images are letterboxed or distorted by that resize.
- [ ] Generate `embeddings.npy` containing the `"free-viewing"` key.
- [ ] Check no stimulus filename contains the substring `jpg` outside its extension
      ([TechStack.md](TechStack.md) §3.2).

---

### F5 — Our-dataset eval branch + run script ⏸

Implements Stage D+E for our data; satisfies D1, D3, D4, D5, D8.

- [ ] Create `ISP/<OurDataset>/GazeformerISP/` mirroring the OSIE tree, with `utils/evaluation.py` and
      `utils/evaltools/*` **copied verbatim** (D1).
- [ ] Adapt only: `dataset/dataset.py` (class name, `origin_size`, subject handling) and `src/test.py`
      (pass `origin_size` explicitly, set `--subject_num` to our unseen-subject count, remove or
      parameterise the `i_batch > 100` cap).
- [ ] Add a single documented run script under `bash/` that takes our data end-to-end.
- [ ] Verify `get_prediction_list()` / `recover_subject_ids()` write our *real* subject ids into
      `prediction.json` (D4).

---

### F6 — Offline re-scorer ⏸

A small CPU-only tool that reads a `fixations.json` (ground truth) + a `prediction.json` and calls the
frozen `comprehensive_evaluation_by_subject()`. Lets us re-derive every reported number on a laptop from
artefacts alone, and is the natural test harness for F1/F5 (D5).

- [ ] Build the `[image][subject]` nested structured-array inputs, in the resized 512×384 space with
      durations in **seconds** ([TechStack.md](TechStack.md) §4).
- [ ] Reproduce F1's numbers from F1's artefacts as the correctness test.
- [ ] Report the NaN-dropped row count alongside every mean (D7).

---

### F7 — Results write-up + validity statement ⏸

- [ ] Table of MultiMatch (5 dims) / ScanMatch (w/, w/o) / SED / STDE / retrieval, mean ± std, ours vs the
      paper's OSIE row.
- [ ] Explicit statement of what the comparison licenses, given the domain gap between the checkpoint's
      training data and our stimuli/subjects, and given the OPEN-2 decision.

---

## 4. Explicitly out of scope

Recorded so they are not silently re-litigated (D8). Reopening any of these is a constitution change.

- ✗ **Training or fine-tuning** ISP or SE-Net on our dataset. Eval-only with released checkpoints.
- ✗ The **RL / policy-gradient** path (`OSIE_rl`, `--start_rl_epoch`).
- ✗ The **COCO-FreeView** and **COCO-Search18** branches, and anything task/search-conditioned.
- ✗ **Modifying, re-implementing, or "improving"** any metric (D1).
- ✗ New metrics not in the paper's suite.
- ✗ Making the repo run on Windows GPUs.

---

## 5. Open decisions

These block specific features. Each must be resolved by the user before its dependent feature starts.

### OPEN-1 — How does our dataset get into this repo? *(blocks F2)*
Deferred by the user on 2026-09-07 ("specified later"). The three candidates:
1. **Converter → `fixations.json`** *(presumed default)*. Least code, least risk, keeps upstream Dataset
   classes untouched, and the output is directly diffable against the shipped OSIE JSON.
2. **New `Dataset` subclass** ported into this repo, mirroring `OSIE` / `OSIE_evaluation`.
3. **Import the other repo as a dependency** (submodule / `pip -e`), adapting at the collate boundary.

Until this is resolved, F2 cannot be specced. Everything in F1 is independent of it.

### OPEN-2 — Whose subject embeddings do our subjects get? *(blocks F3, colours F7)*
The released `fewshot_user_embedding_10.pt` encodes *OSIE subjects 10–14*, not ours. Options:
1. Run SE-Net over a support set of our subjects' real scanpaths to produce genuine embeddings for them
   — scientifically correct, but requires the Detectron2 + MSDeformAttn install.
2. Reuse a released embedding as a stand-in — cheap, but the "personalization" being measured is then
   somebody else's, and F7 must say so plainly.

### OPEN-3 — Support/query split for our subjects *(blocks F2, F5)*
`select_fewshot_subject()` samples `--num_fewshot` images as the support set using `random.shuffle` seeded
by `--random_support`. We must decide `num_fewshot` (paper demos use 10), how many unseen subjects
(`--subject_num`), and how many `random_support` repeats to average over.

### OPEN-4 — Stimulus resolution and resize policy *(blocks F2, F4)*
Our stimuli are almost certainly not 800×600. Decide whether to letterbox, crop, or accept the aspect-ratio
distortion of the 768×1024 feature-extractor resize — and note that MultiMatch and ScanMatch are both
parameterised on the 512×384 screen (`Xbin=16, Ybin=12`), so a different aspect ratio changes the *shape*
of the spatial bins and thus comparability to the paper's numbers.

---

## 6. External dependencies to obtain

| item | source | needed for |
|---|---|---|
| ISP-SENet checkpoints | already in `weights/` (git-ignored) | F1, F5 |
| OSIE stimulus images | NUS-VIP `predicting-human-gaze-beyond-pixels` repo | F1 |
| Detectron2 | source install, per HAT repo | F3 (option 1 only) |
| MSDeformAttn | `SE-Net/src/pixel_decoder/ops/make.sh` | F3 (option 1 only) |
| `stsb-roberta-base-v2` | sentence-transformers hub | F1, F4 |
| Mask R-CNN R50-FPN COCO weights | torchvision download | F1, F4 |
| our dataset + its dataloader | our external repo | F2 |
| cluster allocation with an NVIDIA GPU | — | F1, F3, F4, F5 |
