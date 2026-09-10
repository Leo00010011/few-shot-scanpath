"""Data Validity checks against a real EVE bundle. Run with ``--bundle-dir DIR``.

Each check states its expected outcome; a deviation is a finding to record in
``bridge_report.json`` and carry into F7, not something to silence.
"""

import json
import os

import numpy as np
import pytest
from PIL import Image

from eve_bridge.build import parse_args, run
from eve_bridge.store import GtHeatmapStore

pytestmark = pytest.mark.bundle


@pytest.fixture(scope="module")
def artefacts(request, tmp_path_factory):
    bundle_dir = request.config.getoption("--bundle-dir")
    if not bundle_dir:
        pytest.skip("needs --bundle-dir")
    from evedataset import EveBundle
    bundle = EveBundle.load(bundle_dir)

    out = tmp_path_factory.mktemp("eve_bridge_out")
    argv = ["--bundle-dir", bundle_dir, "--out-dir", str(out),
            "--support-pool-size",
            str(request.config.getoption("--bridge-support-pool-size"))]
    subjects = request.config.getoption("--bridge-subjects")
    if subjects:
        argv += ["--unseen-subjects"] + [s.strip() for s in subjects.split(",")]
    args = parse_args(argv)
    report = run(args, bundle=bundle)

    with open(os.path.join(str(out), "fixations.json")) as fh:
        fixations = json.load(fh)
    with open(os.path.join(str(out), "subject_id_map.json")) as fh:
        subject_id_map = json.load(fh)
    store = GtHeatmapStore.load(os.path.join(str(out), "gt_heatmaps.h5"),
                                fixations_path=os.path.join(str(out), "fixations.json"))
    return bundle, str(out), report, fixations, subject_id_map, store


def test_equal_subject_feasibility_frontier(real_bundle):
    """Report the largest (n_subjects, n_shared_stimuli) the bundle can support.

    FR2.3 keeps only stimuli present for *all* selected subjects, because the
    ISP loader requires every image to yield the same number of subjects. EVE
    assigns a largely per-participant image set, so this frontier -- not the
    participant count -- is what bounds F5's ``--subject_num``. Reporting only:
    the number is a finding for F7, not a pass/fail.
    """
    import itertools

    df = real_bundle.samples_df
    valid = df[df["valid"].astype(bool)]
    sets = {s: set(g["stimulus_name"]) for s, g in valid.groupby("subject")}
    print("participants with any valid trial: {}".format(len(sets)))
    for pref in ("train", "val", "test"):
        n = sum(1 for s in sets if s.startswith(pref))
        print("  {}*: {}".format(pref, n))

    subs = sorted(sets)
    for k in (2, 3, 4):
        best, arg = 0, None
        for combo in itertools.combinations(subs, k):
            n = len(set.intersection(*[sets[c] for c in combo]))
            if n > best:
                best, arg = n, combo
        print("{} subjects -> at most {} shared stimuli {}".format(k, best, arg))
    assert subs, "no participant has a single valid trial"


def test_subject_coverage(artefacts):
    _, _, report, _, subject_id_map, _ = artefacts
    n = report["num_subjects"]
    print("num_subjects = {} (F5's --subject_num)".format(n))
    assert set(subject_id_map["to_dense"].values()) == set(range(n))
    assert subject_id_map["to_eve"] == {str(v): k
                                        for k, v in subject_id_map["to_dense"].items()}
    if n != 10:
        pytest.xfail("EVE test partition on this bundle has {} participants, not 10; "
                     "record it as F5's --subject_num".format(n))


def test_stimulus_coverage(artefacts):
    """The scored pool is the shared stimuli; the support pool is subject-private.

    Since 2026-09-10 the equal-subject filter binds the *query* split only, so a
    stimulus that is not shared by every subject is no longer discarded -- it is
    a support candidate. What matters now is that the query pool is non-empty and
    that every subject's support pool can supply F5's ``--num_fewshot``.
    """
    _, _, report, _, _, _ = artefacts
    print("scored stimuli   : {}".format(report["num_stimuli_test"]))
    print("support stimuli  : {} (per subject: {})".format(
        report["num_stimuli_train"], report["support_per_subject"]))
    print("scored cells     : {}".format(
        report["num_stimuli_test"] * report["num_subjects"]))
    assert report["num_stimuli_test"] > 0
    assert report["min_support_per_subject"] >= 1


def test_support_pool_can_supply_num_fewshot(artefacts):
    """FR3.4 -- F5 must assert ``num_fewshot <= min_support_per_subject``.

    The pools are per-subject and do not share image names, so the binding
    constraint is the MINIMUM across subjects, not ``support_pool_size``.
    """
    _, _, report, _, _, _ = artefacts
    n = report["min_support_per_subject"]
    print("min support/subject = {} (caps F5's --num_fewshot); "
          "support_pool_size = {}".format(n, report["support_pool_size"]))
    # The pools are deliberately NOT trimmed to a common size -- only the minimum
    # binds, since each subject's embedding is built from its own scanpaths.
    assert n >= 1
    if n < 10:
        pytest.xfail("min_support_per_subject = {} caps --num_fewshot below the "
                     "paper's n = 10; exclude the thinnest participant(s) or "
                     "lower num_fewshot".format(n))


def test_trial_count_identity(artefacts):
    """The query split is square; the support split is deliberately ragged."""
    _, _, report, fixations, _, _ = artefacts
    # K per SCORED image -- not num_subjects, which is the whole cohort.
    K = report["subjects_per_image"]
    assert report["num_trials_test"] == report["num_stimuli_test"] * K
    assert report["num_trials_train"] + report["num_trials_test"] == report["num_trials"]
    assert report["num_trials"] == len(fixations)


def test_scanpath_length_distribution(artefacts):
    _, _, report, fixations, _, _ = artefacts
    lengths = np.array([r["length"] for r in fixations])
    print("length min/median/max = {}/{}/{}".format(
        lengths.min(), np.median(lengths), lengths.max()))
    print("short_scanpath (<3) = {}".format(report["counters"]["short_scanpath"]))
    assert lengths.min() >= 1


def test_duration_sanity(artefacts):
    _, _, _, fixations, _, _ = artefacts
    all_T = np.concatenate([np.array(r["T"]) for r in fixations])
    lo, hi = int(all_T.min()), int(all_T.max())
    print("T range = [{}, {}] ms".format(lo, hi))
    assert lo >= 100 and hi <= 1200, (
        "durations outside the EVE fixation filter's [100, 1200] ms window suggest "
        "the wrong scanpath row was read (FR4.1 reads row 1, not row 0)")


def test_coordinate_distribution(artefacts):
    _, _, report, fixations, _, _ = artefacts
    n_fix = sum(r["length"] for r in fixations)
    clamped = report["counters"]["clamped_coords"]
    print("clamped_coords = {} / {} fixation coordinates".format(clamped, n_fix))
    X = np.concatenate([np.array(r["X"]) for r in fixations])
    Y = np.concatenate([np.array(r["Y"]) for r in fixations])
    print("X mean/std = {:.1f}/{:.1f}; Y mean/std = {:.1f}/{:.1f}".format(
        X.mean(), X.std(), Y.mean(), Y.std()))
    assert clamped <= 0.05 * (4 * n_fix), (
        "a large clamp count means off-screen gaze reached the bundle and needs a "
        "decision, not a clip")


def test_cross_check_against_the_source(artefacts):
    bundle, _, _, fixations, _, store = artefacts
    rng = np.random.RandomState(0)
    idx = rng.choice(len(fixations), size=min(20, len(fixations)), replace=False)
    for i in idx:
        rec = fixations[int(i)]
        sp = bundle.get_scanpath(store.exp_keys[int(i)])
        for t in range(rec["length"]):
            assert rec["X"][t] == pytest.approx(
                float(np.clip(float(sp[2][t]) + 1.0, 1.0, 1920.0)), abs=1e-4)
            assert rec["Y"][t] == pytest.approx(
                float(np.clip(float(sp[3][t]) + 1.0, 1.0, 1080.0)), abs=1e-4)
            assert rec["T"][t] == max(int(round(float(sp[1][t]))), 1)


def test_stimulus_files(artefacts):
    _, out, report, fixations, _, _ = artefacts
    names = sorted({r["name"] for r in fixations})
    for name in names:
        with Image.open(os.path.join(out, "stimuli", name)) as img:
            assert img.size == (1920, 1080), "{} is {}".format(name, img.size)
            assert img.mode == "RGB"
    # OPEN-6, recorded 2026-09-08 and still open: the same stimulus_name renders
    # differently per participant. It blocks F4 (one .pth per stimulus_name cannot
    # represent two renderings), not F2 -- the bridge's job is to count it, which
    # it does. Note the conflicts can only land on SHARED stimuli; the support
    # pool is subject-private and therefore single-rendered by construction.
    n_conflict = report["counters"]["stimulus_image_conflict"]
    if n_conflict:
        pytest.xfail(
            "OPEN-6: {} of the {} scored stimuli render differently per "
            "participant; F4's feature extraction is ambiguous until it is "
            "resolved".format(n_conflict, report["num_stimuli_test"]))


def test_heatmap_mass(artefacts):
    _, _, _, fixations, _, store = artefacts
    sums = store.heatmaps.sum(axis=(2, 3))
    violations = 0
    for i, rec in enumerate(fixations):
        n = min(rec["length"], store.attrs["max_length"])
        violations += int(np.abs(sums[i, :n] - 1.0).max(initial=0.0) > 1e-5)
        violations += int(np.abs(sums[i, n:]).max(initial=0.0) != 0.0)
    print("heatmap mass violations = {}".format(violations))
    assert violations == 0


def test_heatmap_spatial_coverage(artefacts):
    _, _, _, _, _, store = artefacts
    agg = store.heatmaps.sum(axis=(0, 1))
    assert agg.shape == (24, 32)
    total = float(agg.sum())
    corners = float(agg[0, 0] + agg[0, -1] + agg[-1, 0] + agg[-1, -1])
    centre = float(agg[8:16, 10:22].sum())
    print("centre mass fraction = {:.3f}, corner fraction = {:.5f}".format(
        centre / total, corners / total))
    assert centre / total > (8 * 12) / (24 * 32), (
        "no centre bias in the aggregate map; check for a row/col swap or a missing -1")


def test_schema_matches_the_shipped_osie_json(artefacts, repo_root):
    _, _, _, fixations, _, _ = artefacts
    osie_path = os.path.join(repo_root, "ISP", "OSIE", "GazeformerISP", "src", "data",
                             "fixations.json")
    with open(osie_path) as fh:
        osie = json.load(fh)
    ours, theirs = fixations[0], osie[0]
    assert set(ours) == set(theirs)
    for key in theirs:
        if isinstance(theirs[key], list):
            assert isinstance(ours[key], list)
            assert type(ours[key][0]) is type(theirs[key][0])
        else:
            assert type(ours[key]) is type(theirs[key]), key


def test_loader_smoke_test(artefacts, repo_root, tmp_path):
    """The contract F5 depends on: OSIE_evaluation must consume our fixations.json."""
    import argparse
    import importlib.util

    torch = pytest.importorskip("torch")
    _, out, report, fixations, _, _ = artefacts

    ds_path = os.path.join(repo_root, "ISP", "OSIE", "GazeformerISP", "src",
                           "dataset", "dataset.py")
    spec = importlib.util.spec_from_file_location("isp_osie_dataset", ds_path)
    dataset_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_mod)

    n_subjects = report["num_subjects"]
    K = report["subjects_per_image"]
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for name in sorted({r["name"] for r in fixations}):
        torch.save(torch.zeros(768, 2048), str(feature_dir / name.replace("jpg", "pth")))

    emb_path = tmp_path / "embeddings.npy"
    np.save(str(emb_path), {"free-viewing": np.zeros(768, dtype=np.float32)})

    args = argparse.Namespace(ex_subject=[-1], fewshot_subject=list(range(n_subjects)),
                              num_fewshot=10, random_support=0,
                              subject_num=K, log_root=None)
    ds = dataset_mod.OSIE_evaluation(
        args,
        stimuli_dir=os.path.join(out, "stimuli"),
        feature_dir=str(feature_dir),
        fixations_dir=os.path.join(out, "fixations.json"),
        task_emb_dir=str(emb_path),
        action_map=(24, 32), origin_size=(1080, 1920), resize=(384, 512),
        type="test")

    assert len(ds) == report["num_stimuli_test"]
    # Every scored image must yield exactly K records -- the invariant the frozen
    # evaluator's -1-initialised collectors depend on. The subject IDENTITIES may
    # differ from image to image, so check the count on every image, not just one.
    for idx in range(len(ds)):
        assert len(ds.imgid_to_sub[ds.imgid[idx]]) == K, (
            "scored image {} yields {} records, expected {}".format(
                ds.imgid[idx], len(ds.imgid_to_sub[ds.imgid[idx]]), K))
    item = ds[0]
    assert len(item["fix_vectors"]) == K
    assert len({r["subject"] for r in fixations if r["split"] == "test"}) > K, (
        "the cohort should span more subjects than any single image carries")
    for vec in item["fix_vectors"]:
        assert vec.dtype == np.dtype({"names": ("start_x", "start_y", "duration"),
                                      "formats": ("f8", "f8", "f8")})
        assert 0.0 <= vec["start_x"].min() and vec["start_x"].max() <= 512.0
        assert 0.0 <= vec["start_y"].min() and vec["start_y"].max() <= 384.0
        assert vec["duration"].max() < 2.0


def test_key_invariants_on_the_artefacts(artefacts):
    _, out, _, fixations, subject_id_map, store = artefacts
    assert len(set(store.exp_keys)) == len(store.exp_keys)
    assert set(store.trial_keys) == {
        "{}|{}".format(r["name"], r["subject"]) for r in fixations}
    for i, key in enumerate(store.trial_keys):
        assert store.trial_key_of(store.exp_keys[i]) == key
        name, subject = key.rsplit("|", 1)
        assert store.exp_key_of(name, int(subject)) == store.exp_keys[i]
        eve_id = subject_id_map["to_eve"][str(int(store.subjects[i]))]
        assert store.exp_keys[i].startswith(eve_id + "_")

    train = {r["name"] for r in fixations if r["split"] == "train"}
    test = {r["name"] for r in fixations if r["split"] == "test"}
    assert train & test == set()
