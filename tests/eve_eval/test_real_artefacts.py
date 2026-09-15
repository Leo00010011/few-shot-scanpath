"""The parts of the Data Validity / Data Architecture Integrity blocks that can run
before the cluster run -- against the REAL ``data/eve_bridge/`` artefacts.

Marked ``bridge`` and skipped when that directory is absent (it is git-ignored, so a
fresh checkout has none). The feature cache and F3's embedding are not on the dev
machine, so everything here is bridge-side; the rest of those blocks needs the run.
"""

import json
import os

import pytest

from conftest import BRIDGE_DIR, OSIE_EMBEDDINGS, TEST_PY

pytestmark = pytest.mark.bridge

EXPECTED_FIXATIONS_SHA = (
    "46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b")


@pytest.fixture(scope="module")
def real_fixations():
    with open(os.path.join(BRIDGE_DIR, "fixations.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def real_report():
    with open(os.path.join(BRIDGE_DIR, "bridge_report.json")) as fh:
        return json.load(fh)


def test_fixations_is_the_artefact_the_spec_names(real_report):
    """FR3.2 -- the sha the whole feature chain is gated on."""
    from eve_eval.check_eval import sha256_file

    actual = sha256_file(os.path.join(BRIDGE_DIR, "fixations.json"))
    assert actual == EXPECTED_FIXATIONS_SHA
    assert real_report["fixations_sha256"] == actual


def test_the_realised_cohort(real_fixations, real_report):
    """FR3.8 over the real split: 1062 cells, 354 images, 38 participants."""
    scored = [r for r in real_fixations if r["split"] == "test"]
    names = {r["name"] for r in scored}
    subjects = sorted({int(r["subject"]) for r in scored})

    assert len(scored) == 1062
    assert len(names) == 354
    assert subjects == list(range(38))

    counts = {}
    for r in scored:
        counts[r["name"]] = counts.get(r["name"], 0) + 1
    assert set(counts.values()) == {3}, sorted(set(counts.values()))

    assert real_report["origin_size"] == [1080, 1920]
    assert real_report["min_support_per_subject"] == 10
    assert max(max(r["T"]) for r in scored) > 20      # not the decile-bin file


def test_subject_trios_really_do_vary_by_image(real_fixations):
    """The Data Validity check that OPEN-5's resolution depends on.

    If there were one trio, OPEN-5's original premise would have been right after all
    and the cohort would need re-deriving.
    """
    scored = [r for r in real_fixations if r["split"] == "test"]
    trios = {}
    for r in scored:
        trios.setdefault(r["name"], set()).add(int(r["subject"]))
    distinct = {frozenset(v) for v in trios.values()}
    assert len(distinct) > 100, len(distinct)


def test_ground_truth_coordinates_land_inside_the_metric_screen(real_fixations):
    """The squash, checked on the real data: max(X)/3.75 <= 512, max(Y)/2.8125 <= 384.

    Validation predicted near-saturation on BOTH axes ("EVE is full-screen"). The
    realised data says otherwise and the numbers are recorded rather than forced:
    **y saturates exactly (1080 -> 384.0) but x does not** -- the scored split's
    maximum is 1720.8 px, 458.9 in metric space, 89.6% of the width. Participants'
    gaze reaches the top and bottom edges but stops ~200 px short of the right-hand
    one. It is an observation for F7, not a failure: nothing here is mis-scaled, the
    horizontal extent of the recorded gaze is simply smaller than the display.
    """
    scored = [r for r in real_fixations if r["split"] == "test"]
    max_x = max(max(r["X"]) for r in scored)
    max_y = max(max(r["Y"]) for r in scored)
    assert max_x / 3.75 <= 512.0
    assert max_y / 2.8125 <= 384.0
    assert max_x == pytest.approx(1720.822, abs=1e-2)
    assert max_y == 1080.0
    assert min(min(r["X"]) for r in scored) >= 1.0      # 1-indexed, never 0
    assert min(min(r["Y"]) for r in scored) >= 1.0


def test_exp_key_roundtrip_is_total_over_the_cohort(real_fixations):
    """Data Architecture Integrity -- neither direction has a fallback."""
    from eve_bridge.store import GtHeatmapStore
    from eve_prep.trial_keys import exp_key_filename, load_trial_exp_keys

    fix_path = os.path.join(BRIDGE_DIR, "fixations.json")
    hm_path = os.path.join(BRIDGE_DIR, "gt_heatmaps.h5")
    mapping = load_trial_exp_keys(hm_path, fixations_path=fix_path)
    store = GtHeatmapStore.load(hm_path, fixations_path=fix_path)

    scored = [r for r in real_fixations if r["split"] == "test"]
    for r in scored:
        key = (r["name"], int(r["subject"]))
        assert key in mapping
        exp_key = store.exp_key_of(*key)
        assert exp_key == mapping[key]
        assert store.trial_key_of(exp_key) == "{}|{}".format(*key)
        exp_key_filename(exp_key)        # charset + no-jpg rule, on every real key

    # Every scored trial has its OWN capture: a repeat would mean two participants
    # were handed one screen (OPEN-6 reappearing).
    scored_keys = [mapping[(r["name"], int(r["subject"]))] for r in scored]
    assert len(set(scored_keys)) == len(scored_keys) == 1062


def test_store_rows_stay_aligned_with_the_records(real_fixations):
    """The store is POSITIONAL: row i is record i. A drift of one would still
    produce finite NSS/CC/KLD."""
    from eve_bridge.store import GtHeatmapStore

    fix_path = os.path.join(BRIDGE_DIR, "fixations.json")
    store = GtHeatmapStore.load(os.path.join(BRIDGE_DIR, "gt_heatmaps.h5"),
                                fixations_path=fix_path)
    assert len(store.trial_keys) == len(real_fixations) == 1804
    for i in range(0, len(real_fixations), 97):
        rec = real_fixations[i]
        assert store.trial_keys[i] == "{}|{}".format(rec["name"], rec["subject"])
        assert int(store.lengths[i]) == min(int(rec["length"]), 16)


def test_the_dataset_constructs_over_the_real_cohort():
    """354 images, the ascending 0..37 identity remap, and the squash -- on the real
    fixations.json. ``__getitem__`` is not exercised: the 6.7 GB feature cache lives
    on the cluster."""
    from dataset.dataset import EVE_evaluation

    from conftest import Args

    ds = EVE_evaluation(
        Args(fewshot_subject=list(range(38)), subject_num=3, num_fewshot=10),
        os.path.join(BRIDGE_DIR, "stimuli"),
        "<no feature dir on this machine>",
        os.path.join(BRIDGE_DIR, "fixations.json"),
        OSIE_EMBEDDINGS,
        heatmaps_dir=os.path.join(BRIDGE_DIR, "gt_heatmaps.h5"),
        exp_key_map={}, action_map=(24, 32), origin_size=(1080, 1920),
        resize=(384, 512), max_length=16, type="test", transform=None)

    assert len(ds) == 354
    assert len(ds.fixations) == 1062
    assert ds.resizescale_x == 3.75 and ds.resizescale_y == 2.8125
    assert sorted({int(r["subject"]) for r in ds.fixations}) == list(range(38))


def test_a_permuted_cohort_is_rejected_on_the_real_data():
    """The D4 trap, on the artefact that will actually be scored."""
    from dataset.dataset import EVE_evaluation

    from conftest import Args

    with pytest.raises(ValueError) as exc:
        EVE_evaluation(
            Args(fewshot_subject=list(reversed(range(38))), subject_num=3,
                 num_fewshot=10),
            os.path.join(BRIDGE_DIR, "stimuli"), "<none>",
            os.path.join(BRIDGE_DIR, "fixations.json"), OSIE_EMBEDDINGS,
            heatmaps_dir=os.path.join(BRIDGE_DIR, "gt_heatmaps.h5"),
            exp_key_map={}, action_map=(24, 32), origin_size=(1080, 1920),
            resize=(384, 512), max_length=16, type="test", transform=None)
    assert "ascending" in str(exc.value)
