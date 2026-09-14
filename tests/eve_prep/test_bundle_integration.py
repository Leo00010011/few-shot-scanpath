"""Validation Group 6 -- bundle integration. ``bundle``-marked; needs --bundle-dir.

These run against the real EVE bundle and the real ``data/eve_bridge/`` artefacts.
The cross-check here is the **D4 gate**: a single disagreement stops F4.
"""

import json
import os

import numpy as np
import pytest

from eve_prep.trial_keys import (crosscheck_exp_keys, derive_trial_exp_keys,
                                 exp_key_filename, load_trial_exp_keys)

pytestmark = [pytest.mark.bundle, pytest.mark.bridge]

SAMPLES_COLUMNS = {"exp_key", "subject", "stimulus_name", "split", "valid",
                   "stimulus_path"}


@pytest.fixture(scope="module")
def _artefacts(request):
    bundle_dir = request.config.getoption("--bundle-dir")
    if bundle_dir is None:
        pytest.skip("needs --bundle-dir")
    from evedataset import EveBundle

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    bridge = os.path.join(repo, "data", "eve_bridge")
    with open(os.path.join(bridge, "fixations.json")) as fh:
        fixations = json.load(fh)
    with open(os.path.join(bridge, "subject_id_map.json")) as fh:
        subject_id_map = json.load(fh)
    with open(os.path.join(bridge, "bridge_report.json")) as fh:
        report = json.load(fh)
    mapping = load_trial_exp_keys(os.path.join(bridge, "gt_heatmaps.h5"),
                                 os.path.join(bridge, "fixations.json"))
    return {"bundle": EveBundle.load(bundle_dir), "fixations": fixations,
            "subject_id_map": subject_id_map, "report": report,
            "mapping": mapping}


def test_samples_df_shape(_artefacts):
    df = _artefacts["bundle"].samples_df
    assert SAMPLES_COLUMNS.issubset(set(df.columns))
    assert len(df) >= 3096


def test_mapping_counts_match_the_bridge_report(_artefacts):
    mapping, report = _artefacts["mapping"], _artefacts["report"]
    assert len(mapping) == report["num_trials"] == 1804
    assert len(set(mapping.values())) == 1804

    splits = {(r["name"], int(r["subject"])): r["split"]
              for r in _artefacts["fixations"]}
    n_test = sum(1 for k in mapping if splits[k] == "test")
    n_train = sum(1 for k in mapping if splits[k] == "train")
    assert (n_test, n_train) == (report["num_trials_test"],
                                 report["num_trials_train"]) == (1062, 742)


def test_crosscheck_agrees_on_every_trial(_artefacts):
    """FR2.4 -- the D4 gate, on all 1804 trials."""
    derived = derive_trial_exp_keys(_artefacts["bundle"].samples_df,
                                    _artefacts["fixations"],
                                    _artefacts["subject_id_map"])
    got = crosscheck_exp_keys(_artefacts["mapping"], derived)
    assert got == {"source": "gt_heatmaps.h5", "derived_from": "samples_df",
                   "agree": True, "n": 1804}


def test_crosscheck_is_not_vacuous(_artefacts):
    """Data Architecture Integrity -- a check that cannot fail is not a check."""
    from eve_prep import EvePrepError

    df = _artefacts["bundle"].samples_df.copy()
    victim = sorted(_artefacts["mapping"])[0]
    row = df["exp_key"] == _artefacts["mapping"][victim]
    df.loc[row, "stimulus_name"] = "corrupted_stimulus_name"

    with pytest.raises(EvePrepError) as exc:
        derived = derive_trial_exp_keys(df, _artefacts["fixations"],
                                        _artefacts["subject_id_map"])
        crosscheck_exp_keys(_artefacts["mapping"], derived)
    assert victim[0][:-4] in str(exc.value)


def test_get_stimulus_shape_on_a_sample(_artefacts):
    bundle = _artefacts["bundle"]
    keys = sorted(_artefacts["mapping"].values())
    sampled = [keys[i] for i in np.linspace(0, len(keys) - 1, 20).astype(int)]
    for key in sampled:
        arr = np.asarray(bundle.get_stimulus(key))
        assert tuple(arr.shape) == (1080, 1920, 3)
        assert arr.dtype == np.uint8


def test_every_exp_key_is_filename_safe(_artefacts):
    """FR7.4 -- all 1804, not a sample."""
    for key in _artefacts["mapping"].values():
        assert exp_key_filename(key) == key + ".pth"


def test_origin_size_agrees_with_the_observed_stimulus(_artefacts):
    report = _artefacts["report"]
    assert list(report["origin_size"]) == [1080, 1920]
    key = sorted(_artefacts["mapping"].values())[0]
    arr = np.asarray(_artefacts["bundle"].get_stimulus(key))
    assert arr.shape[:2] == tuple(report["origin_size"])


def test_subject_identity_survives_end_to_end(_artefacts):
    """D4 -- dense id -> EVE participant -> the capture they saw."""
    df = _artefacts["bundle"].samples_df
    by_key = {str(r.exp_key): str(r.subject) for r in df.itertuples(index=False)}
    to_eve = _artefacts["subject_id_map"]["to_eve"]

    items = sorted(_artefacts["mapping"].items())
    for i in np.linspace(0, len(items) - 1, 10).astype(int):
        (name, dense), exp_key = items[i]
        assert by_key[exp_key] == to_eve[str(dense)]


def test_per_trial_keying_actually_did_something(_artefacts):
    """Data Validity's falsifiable prediction, on a CPU sample (the OPEN-6 claim).

    Two participants' tensors for the same photograph must be **different** -- if they
    were bit-identical, per-trial keying silently collapsed back to per-name and
    OPEN-6 is not resolved -- and yet materially more similar to each other than to
    other stimuli, or the features are not discriminating images and the comparison
    means nothing.

    **Validation's stated form of the second half does not hold and is deliberately
    not asserted here.** It predicted every within-name pair at cosine >= 0.7;
    measured over 20 multi-viewer stimuli the within-name cosine runs
    0.457 - 0.968 (median 0.721), with 9 of 20 below 0.7, and its value tracks the
    per-trial display-scale ratio almost monotonically -- scale 1.004 gives 0.968,
    scale 1.117 gives 0.457. That is the augmentation FR11.1 measured showing up in
    the features exactly as it should, not a defect. The separation is therefore
    asserted on the medians, where it is decisive (0.721 vs 0.382), rather than on a
    flat floor the data does not support. See notes.md.
    """
    import torch

    from eve_prep.extract_features import build_backbone, extract_one

    mapping, bundle = _artefacts["mapping"], _artefacts["bundle"]
    by_name = {}
    for name, subject in mapping:
        by_name.setdefault(name, []).append(subject)
    multi = sorted(n for n, subs in by_name.items() if len(subs) >= 2)
    assert multi, "no stimulus is viewed by two participants"

    sample = [multi[i] for i in np.linspace(0, len(multi) - 1, 6).astype(int)]
    backbone = build_backbone(torch.device("cpu"))
    cache = {}

    def feature(key):
        if key not in cache:
            cache[key] = extract_one(backbone, np.asarray(bundle.get_stimulus(key)))
        return cache[key]

    def cosine(a, b):
        a, b = a.flatten(), b.flatten()
        return float(torch.dot(a, b) / (a.norm() * b.norm()))

    within = []
    for name in sample:
        first, second = sorted(by_name[name])[:2]
        a = feature(mapping[(name, first)])
        b = feature(mapping[(name, second)])
        assert not torch.equal(a, b), (
            "{}: the two participants' tensors are bit-identical -- per-trial keying "
            "collapsed back to per-name and OPEN-6 is NOT resolved".format(name))
        within.append(cosine(a, b))

    cross = [cosine(feature(mapping[(sample[i], sorted(by_name[sample[i]])[0])]),
                    feature(mapping[(sample[j], sorted(by_name[sample[j]])[0])]))
             for i in range(len(sample)) for j in range(i + 1, len(sample))]

    # The load-bearing claim: same photograph, two scales, clearly more alike than
    # two different photographs. A generous margin, so this fails on a real collapse
    # rather than on sampling scatter.
    assert float(np.median(within)) > float(np.median(cross)) + 0.2, (within, cross)


def test_dense_ids_are_exactly_0_to_37(_artefacts):
    dense = sorted({subject for _, subject in _artefacts["mapping"]})
    assert dense == list(range(38))
    assert _artefacts["report"]["num_subjects"] == 38
