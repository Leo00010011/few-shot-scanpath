# F3 — Validation

> Spec 3 of 3. Read with [requirements.md](requirements.md) and [plan.md](plan.md).
> Created: 2026-09-10

Groups 1, 2, 3 and 6 run as `py -m pytest tests/eve_senet -q` on Windows CPU with synthetic fixtures.
Groups 4, 5 and everything under **Data Validity** need the `senet` env and a GPU, and run on the
cluster against the real bridge artefacts.

---

## Code Correctness

### Group 1 — Duration binning (`durations.py`, CPU, no fixtures)

- [ ] `decile_bins(range(100))` returns `edges` of shape `(11,)`, dtype `float64`, and
      `np.all(np.diff(edges) >= 0)` is `True`. A decreasing edge means the quantile call was
      mis-parameterised.
- [ ] For a uniform input `0..999`, every returned `bin_of(t)` lies in `{0..9}` and each bin holds
      `100 ± 1` values. Any value outside `{0..9}` means `searchsorted` was given the full `edges`
      rather than `edges[1:-1]` — an off-by-one that silently creates an 11th bin.
- [ ] Boundary behaviour is pinned: `bin_of(edges[0]) == 0` and `bin_of(edges[-1]) == 9`. The maximum
      duration must land in the top bin, not overflow it.
- [ ] A tied input (`[5] * 100`) does not raise and returns all values in one bin. Equal counts are
      **not** required — ties legitimately collapse buckets, and forcing equality would misassign.
- [ ] `decile_bins([])` raises `EveSenetError`, not `IndexError` from numpy (FR11 — our errors, named).
- [ ] `bin_scanpath_durations(records, bin_of)` returns **copies**: the input records' `T` lists are
      unchanged after the call (asserted by identity and by value), so the caller can still report
      milliseconds.
- [ ] Applied to the real support split's durations, the edges equal
      `[100, 131, 151, 169, 188, 210, 234, 263, 308, 390, 1144]` to within `1e-9`. Recomputed from
      `data/eve_bridge/fixations.json`, never hardcoded in the implementation.

### Group 2 — Support selection (`select_support`, CPU, synthetic records)

- [ ] With 38 synthetic subjects × 20 records each, `select_support(recs, 10, 0)` returns exactly 38
      keys `0..37`, each a list of exactly 10 **distinct** names.
- [ ] The same `(records, num_fewshot, seed)` returns an identical selection across two calls **and**
      across two processes (the seed is a string fed to `random.Random`, not the interpreter's hash
      seed — a `PYTHONHASHSEED`-dependent draw would be irreproducible).
- [ ] `seed=0` and `seed=1` produce **different** selections for at least 30 of the 38 subjects. If a
      seed change moves nothing, `support_pool_size = 20 > num_fewshot = 10` is buying nothing and
      OPEN-3's averaging plan is void.
- [ ] Shuffling the input record list does not change the output — the draw is over lexicographically
      sorted names (FR4.2), so input order cannot leak in.
- [ ] A subject with 9 records raises `EveSenetError` whose message names that subject's dense id
      (FR11.2). Not an assertion error, not a silent short draw.
- [ ] Every selected name resolves back to a record with `split == "train"`; a planted `test` record in
      the candidate set raises `EveSenetError` (FR11.4).

### Group 3 — Fix-label construction (`build_fix_labels`, CPU, synthetic records + stub images)

- [ ] Input of `N` records returns exactly `N` fix-labels. `filter_scanpath` keeps only `is_last`
      entries and `has_stop=True` produces exactly one per scanpath; any other count means
      `has_stop` was dropped and the terminal label — the only one carrying the full `T` list — is gone.
- [ ] Each returned label unpacks to the 9-tuple
      `(img_name, cat_name, condition, fixs, action, is_last, sid, dura, dataset)` with `is_last is True`
      and `cat_name == "none"`, `condition == "freeview"`.
- [ ] `dura` is the record's **full `T` list** (post-binning: every element in `{0..9}`), not a scalar
      cumulative sum. A scalar here means an intermediate label leaked past `filter_scanpath` and
      `process_data`'s `len(dura)` would raise later.
- [ ] Rescaling: a record with `X = [1920.0]`, `Y = [1080.0]` yields scaled `X ≈ 512.0`,
      `Y ≈ 320.0` to within `1e-9` (`ratio_w = 0.2666…`, `ratio_h = 0.2962…`, FR5.3).
- [ ] Out-of-bound counting: a record whose 2nd fixation is at `(1920.0, 540.0)` increments
      `counters["oob_fixations_dropped"]` by exactly 1, and the resulting `fixs` list is one shorter
      than the record's length. A record whose **first** fixation is out of bound increments the
      counter by 0 — upstream exempts index 0 (FR5.5).
- [ ] `counters["len1_scanpaths"]` counts records with `length == 1`, and such a record still produces
      one valid label with a single fixation (FR5.6).
- [ ] A record with `length == 21` (> `max_traj_length = 20`) raises `EveSenetError` (FR11.8) rather
      than being silently truncated.
- [ ] `is_coco_dataset=False` is honoured: the first fixation of the returned label is the record's own
      rescaled `(X[0], Y[0])`, **not** `(im_w // 2, im_h // 2)`. A centre-forced first fixation means
      the COCO-Search18 convention leaked in and every embedding starts from the same point.
- [ ] `EveSupportDataset.__getitem__(0)` returns a dict whose only key is `"anchor"`, and the call
      **does not raise** on a dataset containing exactly one subject. Calling
      `Siamese_Triplet_Gaze.__getitem__` on the same dataset raises `IndexError` — assert both, since
      the second is the whole reason the subclass exists (FR5.2).

### Group 4 — Model loading and forward pass (GPU, `senet` env)

- [ ] `check_env.py` exits `0` with every required item `ok`, and its JSON names resolved versions for
      `torch`, `detectron2`, MSDeformAttn, `timm`, `numpy`, `scipy` (FR1.4). Deliberately breaking one
      import makes it exit non-zero.
- [ ] `load_model()` returns `missing_keys` containing **no key matching `subject_predictor.*`**. Such a
      key means `num_subjects` disagrees with the checkpoint and the head is randomly initialised under
      `strict=False` (FR6.2). Raise, do not allowlist.
- [ ] `unexpected_keys` is reported in full in `senet_report.json`, empty or not.
- [ ] A single forward pass over one subject's 10 support scanpaths returns
      `logits["user_emb"]` of shape `(B, 384)`, dtype `float32`, all finite.
- [ ] `n_seen == num_fewshot == 10` for every one of the 38 subjects (FR7.2/FR11.5). At `batch_size=8`
      with `drop_last=True` this would be `8` — the check that catches the silent 20 % evidence loss.
- [ ] The per-subject row is the **mean**, not the sum: for a subject, `row` equals
      `torch.stack(all_user_embs).mean(0)` to within `1e-6`. `evaluate_user_siamese`'s
      `num_fewshot == 1` branch saves an un-normalised sum; F3 must not reproduce that.
- [ ] No optimiser object is constructed anywhere in the call graph, and
      `all(not p.requires_grad or p.grad is None for p in model.parameters())` after the run (D8).

### Group 5 — The duration-deadness pin (GPU, `senet` env) ◀ the finding that must not rot

- [ ] Running `build_embeddings` twice at the same seed — once with FR3 decile bins, once with raw
      milliseconds substituted — produces tensors that are **bitwise identical**
      (`torch.equal(a, b) is True`, not `allclose`). This pins the observation that
      `SE-Net/src/models.py` L677–678 adds `duration_encoding` into `ventral_pos` and then calls
      `ventral_pos.fill_(0)`, so the duration never reaches the network (FR3.4).
- [ ] The same comparison with an **absurd** duration (all fixations set to `1e6`) is also bitwise
      identical. A tolerance-based check could hide a small live contribution; an absurd input cannot.
- [ ] **On failure, this is not a test bug.** It means the duration channel is live in this
      env/checkpoint combination, in which case: every embedding generated under the assumption is
      invalid, FR3's binning becomes load-bearing rather than merely faithful, and the result must be
      escalated to a roadmap note before F5 consumes anything.
- [ ] `senet_report.json` records `duration_channel_pinned: true` with the tensor hashes of both runs,
      so the check is an artefact and not only a green test.

### Group 6 — Frozen-tree and convention compliance (CPU)

- [ ] `git status --porcelain SE-Net/ ISP/` is empty after a full run — no tracked file under either
      tree was modified (FR10.2, D1). Run before and after; compare.
- [ ] `git diff --stat` over `ISP/*/GazeformerISP/src/utils/` is empty, checked separately and
      explicitly, because D1 is the constitution's hardest constraint and a blanket check can pass while
      hiding it.
- [ ] `grep -r "eve_bridge" tools/eve_senet/` returns nothing — F3 reads the bridge's artefacts, never
      its code (FR10.3, D2).
- [ ] `grep -rE "evaluation|evaltools" tools/eve_senet/` returns nothing (FR10.1).
- [ ] No new untracked `.pyc` remains under `SE-Net/` after the run script completes (FR10.5, working
      convention 6). Verified with `git status --ignored SE-Net/`.
- [ ] No absolute path appears in any `.py` under `tools/eve_senet/` or in
      `SE-Net/configs/eve_useremb.json` (working convention 4). Cluster paths live in
      `bash/embed_eve_subjects.sh` and CLI args only.
- [ ] `data/eve_senet/` is covered by `.gitignore` and `git status --ignored` shows the `.pt` and
      `senet_report.json` as ignored — subject-level derived data never reaches the repo (working
      convention 5). Note the `!spec/**/*.json` negation does **not** apply here.
- [ ] MSDeformAttn build artefacts (`SE-Net/src/pixel_decoder/ops/build/`, `*.so`, `*.egg-info/`) are
      ignored and do not appear as untracked. This is the one step that writes into `SE-Net/`, and it
      must be visibly a compiler output rather than an edit.

---

## Data Validity

Run once on the cluster against the real artefacts. Each states its expected outcome; these are
notebook-or-shell checks, not pytest.

- [ ] **Tensor shape and dtype.** `torch.load(...)` gives `(38, 384)`, `torch.float32`. Expected: exactly
      that. `(10, 384)` would mean the released OSIE tensor was loaded by mistake.
- [ ] **No zero rows.** `(emb == 0).all(dim=1).sum() == 0`. Expected: 0. The released
      `fewshot_user_embedding_10.pt` has 5 zero rows; ours must have none (FR7.5).
- [ ] **No duplicate rows.** No two rows identical to `1e-9`. Two identical rows would mean two subjects
      were fed the same support set — a selection bug that the per-subject seed should make impossible.
- [ ] **Row norms are comparable.** `||row||₂` across the 38 rows has a max/min ratio below ~3. A single
      wildly-scaled row suggests one subject's forward passes differed structurally (e.g. a
      length-1-dominated support set).
- [ ] **Off-diagonal cosine similarity.** Report min / mean / max over the `38 × 38` matrix
      (FR9.3). Expected: a mean well below 1.0 with visible spread. If the mean is ≈ 1.0 the encoder is
      not discriminating our subjects at all, and F7 cannot claim personalization regardless of what F5's
      metrics do. **No threshold is enforced — this is reported and interpreted, not gated.**
- [ ] **Support selection covers the pool.** Across seeds 0 and 1, the union of selected names per
      subject exceeds 10 for every subject whose pool has more than 10 entries. Expected: true for 37 of
      38 (subject 34 has exactly 10 and must select the same 10 at both seeds).
- [ ] **Counters.** `oob_fixations_dropped` and `len1_scanpaths` are reported with their values. From
      the bridge data, `Y` reaches exactly 1080.0, so a small non-zero `oob_fixations_dropped` is
      **expected**, not alarming; a value in the hundreds would mean the rescale ratio is wrong.
      `truncated_scanpaths` must be exactly `0` (support max length is 9 against `max_traj_length` 20).
- [ ] **Duration bin occupancy.** Each of the 10 bins holds roughly `4240 / 10 = 424` fixations. Large
      imbalance is acceptable only where ties explain it; report the counts.
- [ ] **Support/query disjointness holds in the realised selection.** The union of all selected support
      names intersected with the set of `test`-split names is **empty**. Expected: empty. This re-checks
      F2's guarantee on the artefact F3 actually consumed, rather than trusting the bridge report.
- [ ] **Cross-check against `bridge_report.json`.** `num_subjects == 38`,
      `min_support_per_subject == 10`, `num_trials_train == 742`, and the train-split record count read
      from `fixations.json` equals 742. Any disagreement means the report and the JSON have drifted.

---

## Data Architecture Integrity

The keying invariants. These are the checks that stop a plausible, wrong number.

- [ ] **`exp_key` / dense-id roundtrip.** For every dense id `i ∈ 0..37`,
      `subject_id_map["to_dense"][subject_id_map["to_eve"][str(i)]] == i`. Expected: true for all 38.
      This is D4's core claim and the only thing that lets anyone answer "which of my real subjects is
      row 3?".
- [ ] **No phantom keys.** `set(subject_id_map["to_eve"]) == {"0".."37"}` **and**
      `set(r["subject"] for r in fixations) == set(range(38))`. A dense id present in one and not the
      other means the map and the data have diverged (FR11.7).
- [ ] **Row order is positional and asserted, not assumed.** `verify_embedding.py` prints
      `i → to_eve[str(i)] → ||row_i||₂` for all 38 rows, and the printed EVE ids are in the same order
      as `sorted(bridge_report["args"]["unseen_subjects"])`. The dense ids are the index into the
      *sorted* participant list (TechStack §3.6), so this is checkable, not a matter of trust.
- [ ] **The embedding row order survives ISP's remap.** `ISP`'s own `select_fewshot_subject()` remaps
      `--fewshot_subject` to a dense `0..N-1` range and indexes `self.subject_embed[subjects]`. Assert
      that passing `--fewshot_subject 0 1 2 … 37` produces the **identity** remap, so row `i` of our
      tensor is still subject `i`. **This is F5's live trap**: any other argument order silently
      permutes the entire cohort's embeddings, and every metric would still look plausible.
- [ ] **`fixations_sha256` gate is not bypassable.** Corrupting one byte of `fixations.json` — and,
      separately, merely **reordering** its records — both make `embed.py` raise `EveSenetError`
      (FR11.1). A reorder must fail: F2's artefacts are addressed positionally (working convention 10),
      so an order-insensitive check would let a mis-indexed run through.
- [ ] **`embedding_sha256` is recorded and matches.** Re-hashing the saved `.pt` reproduces
      `senet_report["embedding_sha256"]`. F5 asserts this before consuming the tensor, so the two
      features cannot silently be discussing different artefacts (FR8.2).
- [ ] **`checkpoint_sha256` is recorded.** Establishes which released SE-Net checkpoint the embeddings
      came from, so OPEN-7's "the released checkpoint may differ from the one behind the table" question
      has a concrete handle for the SE-Net side too.
- [ ] **Seed appears in the filename and inside the report, and they agree.**
      `eve_fewshot_user_embedding_10_seed0.pt` ↔ `senet_report["args"]["seed"] == 0`. This is the guard
      `aggregate_seeds.py` needed for F1 (a mis-targeted copy step pooling non-replicates); F3 must not
      reintroduce the same hazard for F5.
