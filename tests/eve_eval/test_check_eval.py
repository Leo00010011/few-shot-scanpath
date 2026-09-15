"""Group 6 -- the preflight. Each case builds a deliberately broken artefact set.

Every one of these failures, unchecked, produces a run that completes and reports a
plausible number. The fixture here is small and rebuilt per case (tiny feature tensors:
the preflight hashes bytes, it never loads a tensor), so a case can corrupt exactly one
thing and assert exactly one message.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from conftest import (EVE_IDS, N_POOL, OSIE_EMBEDDINGS, REPO_ROOT, TOOLS,
                      build_fixations, exp_key_for, sha256_file)

COHORT = dict(subject_num=3, num_fewshot=2, expected_cells=18, expected_images=6,
              expected_subjects=N_POOL)


def build_case(root, fixations=None, bridge_overrides=None, senet_overrides=None,
               feature_overrides=None, subject_map=None, drop_features=(),
               corrupt_features=(), embeddings_src=OSIE_EMBEDDINGS,
               subject_map_file=None):
    """A complete, minimal artefact set under ``root``; returns the path dict."""
    from eve_bridge.store import GtHeatmapStore

    fixations = build_fixations() if fixations is None else fixations
    bridge = os.path.join(root, "eve_bridge")
    senet = os.path.join(root, "eve_senet", "seed0")
    features = os.path.join(root, "eve_features")
    feat_dir = os.path.join(features, "image_features")
    weights = os.path.join(root, "weights")
    for d in (bridge, senet, feat_dir, os.path.join(weights, "checkpoints")):
        os.makedirs(d, exist_ok=True)

    fix_path = os.path.join(bridge, "fixations.json")
    with open(fix_path, "w") as fh:
        json.dump(fixations, fh)
    fix_sha = sha256_file(fix_path)

    subject_map = subject_map or {"to_dense": {v: k for k, v in EVE_IDS.items()},
                                  "to_eve": {str(k): v for k, v in EVE_IDS.items()}}
    # subject_map_file is written AFTER the store is built: GtHeatmapStore.build reads
    # to_eve itself, so a hole in the map would break the fixture rather than the
    # artefact under test.
    with open(os.path.join(bridge, "subject_id_map.json"), "w") as fh:
        json.dump(subject_map_file or subject_map, fh)

    exp_keys = [exp_key_for(r["name"], r["subject"]) for r in fixations]
    store = GtHeatmapStore.build(
        fixations, exp_keys, subject_map,
        {"fixations_sha256": fix_sha, "bundle_dir": "<synthetic>"},
        origin_size=(1080, 1920), action_map=(24, 32), max_length=16, blur_sigma=1)
    if bridge_overrides and "h5_attrs" in bridge_overrides:
        store.attrs.update(bridge_overrides["h5_attrs"])
    hm_path = os.path.join(bridge, "gt_heatmaps.h5")
    store.save(hm_path)

    bridge_report = {"fixations_sha256": fix_sha, "origin_size": [1080, 1920],
                     "action_map": [24, 32], "max_length": 16,
                     "min_support_per_subject": 10, "support_pool_size": 20,
                     "counters": {}}
    bridge_report.update({k: v for k, v in (bridge_overrides or {}).items()
                          if k != "h5_attrs"})
    with open(os.path.join(bridge, "bridge_report.json"), "w") as fh:
        json.dump(bridge_report, fh)

    # Tiny tensors: the preflight hashes bytes and never loads one (FR3.4).
    feature_sha = {}
    for i, key in enumerate(exp_keys):
        path = os.path.join(feat_dir, key + ".pth")
        torch.save(torch.full((4,), float(i), dtype=torch.float32), path)
        feature_sha[key] = sha256_file(path)
    for key in drop_features:
        os.remove(os.path.join(feat_dir, key + ".pth"))
    for key in corrupt_features:
        path = os.path.join(feat_dir, key + ".pth")
        with open(path, "ab") as fh:
            fh.write(b"\x00")               # bytes change, the recorded hash does not

    shutil.copyfile(embeddings_src, os.path.join(features, "embeddings.npy"))
    feature_report = {"fixations_sha256": fix_sha, "keying": "exp_key",
                      "feature_sha256": feature_sha}
    feature_report.update(feature_overrides or {})
    with open(os.path.join(features, "feature_report.json"), "w") as fh:
        json.dump(feature_report, fh)

    emb_path = os.path.join(senet, "eve_fewshot_user_embedding_10_seed0.pt")
    torch.save(torch.zeros((N_POOL, 384), dtype=torch.float32), emb_path)
    senet_report = {"fixations_sha256": fix_sha,
                    "embedding_sha256": sha256_file(emb_path),
                    "args": {"seed": 0}}
    senet_report.update(senet_overrides or {})
    with open(os.path.join(senet, "senet_report.json"), "w") as fh:
        json.dump(senet_report, fh)

    with open(os.path.join(weights, "checkpoints", "checkpoint_best.pth"), "wb") as fh:
        fh.write(b"synthetic checkpoint")

    return {"bridge_dir": bridge, "senet_dir": senet, "feature_dir": features,
            "feat_dir": feat_dir, "weights_dir": weights, "fixations": fix_path,
            "heatmaps": hm_path, "user_embedding": emb_path,
            "exp_keys": exp_keys}


def run_check(paths, **kwargs):
    from eve_eval.check_eval import check_eval

    opts = dict(COHORT)
    opts.update(kwargs)
    return check_eval(paths["bridge_dir"], paths["senet_dir"], paths["feature_dir"],
                      paths["weights_dir"], **opts)


def failures_matching(report, *needles):
    return [f for f in report["failures"]
            if all(n.lower() in f.lower() for n in needles)]


@pytest.fixture
def case(tmp_path):
    def _case(**kwargs):
        root = tmp_path / "case{}".format(len(os.listdir(str(tmp_path))))
        root.mkdir()
        return build_case(str(root), **kwargs)
    return _case


def test_clean_fixture_passes(case):
    report = run_check(case())
    assert report["ok"], report["failures"]
    assert report["counts"]["n_cells"] == 18
    assert report["counts"]["n_images"] == 6
    assert report["heatmap_parity"]["bitwise_equal"] is True
    assert report["feature_sha256_checked"] is True


def test_missing_artefact_names_the_path_and_the_git_ignore(case):
    paths = case()
    os.remove(paths["user_embedding"])
    report = run_check(paths)
    assert not report["ok"]
    hits = failures_matching(report, "FR3.1", "user_embedding")
    assert hits, report["failures"]
    assert "git-ignored" in hits[0]


def test_changed_fixations_fails_all_three_handshakes(case):
    paths = case()
    with open(paths["fixations"]) as fh:
        records = json.load(fh)
    records[0]["X"][0] += 1.0
    with open(paths["fixations"], "w") as fh:
        json.dump(records, fh)

    report = run_check(paths)
    assert not report["ok"]
    for label in ("bridge_report.json", "senet_report.json", "feature_report.json"):
        assert failures_matching(report, "FR3.2", label), (label, report["failures"])


def test_reordered_fixations_still_fails(case):
    """Working convention 10: bridge artefacts are addressed POSITIONALLY."""
    paths = case()
    with open(paths["fixations"]) as fh:
        records = json.load(fh)
    with open(paths["fixations"], "w") as fh:
        json.dump(list(reversed(records)), fh)
    report = run_check(paths)
    assert failures_matching(report, "FR3.2"), report["failures"]


def test_changed_embedding_names_both_hashes(case):
    paths = case()
    with open(paths["user_embedding"], "ab") as fh:
        fh.write(b"\x00")
    report = run_check(paths)
    hits = failures_matching(report, "FR3.3", "hash mismatch")
    assert hits, report["failures"]
    assert "senet_report.json" in hits[0]


def test_seed_disagreement_fails(case):
    report = run_check(case(senet_overrides={"args": {"seed": 1}}))
    assert failures_matching(report, "FR3.3", "seed"), report["failures"]


def test_min_support_bound(case):
    report = run_check(case(bridge_overrides={"min_support_per_subject": 9}),
                       num_fewshot=10)
    hits = failures_matching(report, "FR3.5")
    assert hits, report["failures"]
    assert "MINIMUM" in hits[0] and "support_pool_size" in hits[0]


def test_wrong_origin_size_fails(case):
    report = run_check(case(bridge_overrides={"origin_size": [600, 800]}))
    assert failures_matching(report, "FR3.6", "origin_size"), report["failures"]


@pytest.mark.parametrize("attr,value", [("action_map", [30, 40]),
                                        ("max_length", 20)])
def test_wrong_h5_attrs_fail(case, attr, value):
    report = run_check(case(bridge_overrides={"h5_attrs": {attr: value}}))
    assert failures_matching(report, "FR3.6", attr), report["failures"]


def test_task_embedding_must_be_the_osie_one(case, tmp_path):
    import numpy as np

    substitute = tmp_path / "other_embeddings.npy"
    np.save(str(substitute), {"free-viewing": np.zeros(768, dtype=np.float32)},
            allow_pickle=True)
    report = run_check(case(embeddings_src=str(substitute)))
    assert failures_matching(report, "FR3.7", "byte-identical"), report["failures"]


def test_task_embedding_missing_the_key_fails(case, tmp_path):
    import numpy as np

    substitute = tmp_path / "no_key.npy"
    np.save(str(substitute), {"search": np.zeros(768, dtype=np.float32)},
            allow_pickle=True)
    report = run_check(case(embeddings_src=str(substitute)))
    assert failures_matching(report, "FR3.7", "free-viewing"), report["failures"]


@pytest.mark.parametrize("delta", [-1, 1])
def test_ragged_image_fails_naming_it(case, delta):
    """The invariant that stops -1 folding into every metric (Mission P4)."""
    records = build_fixations()
    if delta == -1:
        records = [r for r in records
                   if not (r["name"] == "img0.jpg" and r["subject"] == 2)]
        expected_count = "2"
    else:
        extra = dict(records[0])
        extra["subject"] = 4
        records.append(extra)
        expected_count = "4"
    report = run_check(case(fixations=records),
                       expected_cells=len(
                           [r for r in records if r["split"] == "test"]))
    hits = failures_matching(report, "FR3.8", "ragged")
    assert hits, report["failures"]
    assert "img0.jpg" in hits[0] and expected_count in hits[0]


def test_gap_in_dense_subject_ids_fails(case):
    records = [r for r in build_fixations() if r["subject"] != 3]
    records = [dict(r, subject=4 if r["subject"] == 4 else r["subject"])
               for r in records]
    report = run_check(case(fixations=records),
                       expected_cells=len([r for r in records
                                           if r["split"] == "test"]),
                       expected_images=len({r["name"] for r in records
                                            if r["split"] == "test"}))
    assert failures_matching(report, "FR3.8", "dense subject ids"), report["failures"]


def test_decile_bin_file_is_caught_and_nothing_else_is(case):
    """Every OTHER invariant passes on this fixture -- which is why the check exists."""
    records = [dict(r, T=[i % 10 for i in range(len(r["T"]))])
               for r in build_fixations()]
    report = run_check(case(fixations=records))
    hits = failures_matching(report, "FR3.8", "max(T)")
    assert hits, report["failures"]
    assert "DECILE-BIN" in hits[0]
    assert len(report["failures"]) == 1, report["failures"]


def test_wrong_condition_fails_naming_the_record(case):
    records = build_fixations()
    records[0] = dict(records[0], condition="search")
    report = run_check(case(fixations=records))
    hits = failures_matching(report, "FR3.8", "condition")
    assert hits, report["failures"]
    assert records[0]["name"] in hits[0]


def test_missing_tensor_is_listed_with_the_total(case):
    paths = case(drop_features=["img0_s0"])
    report = run_check(paths)
    hits = failures_matching(report, "FR3.9", "missing")
    assert hits, report["failures"]
    assert "img0_s0.pth" in hits[0] and "1 total" in hits[0]


def test_fifteen_missing_tensors_list_ten_and_say_how_many_more(case):
    keys = [exp_key_for(r["name"], r["subject"]) for r in build_fixations()][:15]
    report = run_check(case(drop_features=keys))
    hits = failures_matching(report, "FR3.9", "missing")
    assert hits, report["failures"]
    assert "15 total" in hits[0] and "(+5 more)" in hits[0]


def test_corrupt_tensor_fails_the_hash_sweep(case):
    report = run_check(case(corrupt_features=["img1_s2"]))
    hits = failures_matching(report, "FR3.4", "img1_s2")
    assert hits, report["failures"]


def test_fast_skips_only_the_hash_sweep(case):
    """FR11.4 -- a skipped check is stated, never implied."""
    report = run_check(case(corrupt_features=["img1_s2"]), fast=True)
    assert report["ok"], report["failures"]
    assert report["feature_sha256_checked"] is False

    report = run_check(case(drop_features=["img1_s2"]), fast=True)
    assert failures_matching(report, "FR3.9"), report["failures"]


def test_dense_id_absent_from_to_eve_fails(case):
    holed = {"to_dense": {v: k for k, v in EVE_IDS.items()},
             "to_eve": {str(k): v for k, v in EVE_IDS.items() if k != 4}}
    report = run_check(case(subject_map_file=holed))
    assert failures_matching(report, "FR7.5", "to_eve"), report["failures"]


def test_cli_writes_preflight_json_on_failure_and_exits_one(case, tmp_path):
    paths = case(drop_features=["img0_s0"])
    out = tmp_path / "preflight.json"
    # The real cohort constants apply on the CLI path, so this fixture fails on the
    # counts as well -- which is exactly the point: a failing run still writes the
    # report, with every failure in it (Step 3, FR11.2).
    proc = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "eve_eval", "check_eval.py"),
         "--bridge-dir", paths["bridge_dir"], "--senet-dir", paths["senet_dir"],
         "--feature-dir", paths["feature_dir"], "--weights-dir", paths["weights_dir"],
         "--out", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT)
    assert proc.returncode == 1, proc.stdout.decode(errors="replace")
    assert out.is_file()
    with open(str(out)) as fh:
        report = json.load(fh)
    assert report["ok"] is False and report["failures"]
    assert b"FATAL" in proc.stderr


def test_check_eval_imports_no_torch_and_no_evedataset():
    """FR1.3 / FR11.1 -- it must import on a login node with nothing staged."""
    code = (
        "import sys; sys.path.insert(0, {tools!r});"
        "import eve_eval.check_eval as m;"
        "print('torch' in sys.modules, 'evedataset' in sys.modules)"
    ).format(tools=TOOLS)
    proc = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout.split() == [b"False", b"False"], proc.stdout


def test_cli_accepts_an_explicit_checkpoint(case, tmp_path):
    """The preflight must be runnable BEFORE the run script has ever executed.

    `<weights-dir>/checkpoints/checkpoint_best.pth` is a symlink bash/test_eve.sh
    creates to the released OSIE checkpoint, so on a fresh cluster checkout it does not
    exist yet and the standalone preflight would fail on FR3.1 for a file that is not
    actually missing. --checkpoint points at the released one directly.
    """
    paths = case()
    released = tmp_path / "released_checkpoint_best.pth"
    released.write_bytes(b"the released OSIE free-viewing checkpoint")
    out = tmp_path / "preflight_explicit.json"

    proc = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "eve_eval", "check_eval.py"),
         "--bridge-dir", paths["bridge_dir"], "--senet-dir", paths["senet_dir"],
         "--feature-dir", paths["feature_dir"],
         "--weights-dir", os.path.join(paths["weights_dir"], "no-symlink-here"),
         "--checkpoint", str(released), "--out", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT)

    with open(str(out)) as fh:
        report = json.load(fh)
    # The cohort constants make the CLI path fail on counts over this 6-image fixture;
    # what matters is that the checkpoint is NOT among the failures and was hashed.
    assert not [f for f in report["failures"] if "checkpoint" in f], report["failures"]
    assert report["sha256"]["checkpoint"]
    assert report["paths"]["checkpoint"] == str(released)
