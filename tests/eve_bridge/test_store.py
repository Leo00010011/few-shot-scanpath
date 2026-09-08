"""Validation Group 4 -- HDF5 store roundtrip (store.py)."""

import datetime
import json

import h5py
import numpy as np
import pytest

from conftest import make_fake_bundle
from eve_bridge import store as store_mod
from eve_bridge.convert import build_fixations
from eve_bridge.heatmaps import build_step_heatmaps
from eve_bridge.store import GtHeatmapStore, sha256_file

HM_KWARGS = dict(origin_size=(1080, 1920), action_map=(24, 32),
                 max_length=16, blur_sigma=1)


@pytest.fixture
def built(tmp_path):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=11)
    fixations, exp_keys, subject_id_map, _ = build_fixations(
        bundle, support_pool_size=2)

    fx_path = tmp_path / "fixations.json"
    fx_path.write_text(json.dumps(fixations, indent=4))

    st = GtHeatmapStore.build(
        fixations, exp_keys, subject_id_map,
        attrs={"bundle_dir": "fake", "fixations_sha256": sha256_file(str(fx_path))},
        **HM_KWARGS)
    h5_path = tmp_path / "gt_heatmaps.h5"
    st.save(str(h5_path))
    return fixations, exp_keys, subject_id_map, str(fx_path), str(h5_path)


def test_row_order_matches_fixations(built):
    fixations, _, _, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)
    assert len(st.trial_keys) == len(fixations)
    for i, rec in enumerate(fixations):
        assert st.trial_keys[i] == "{}|{}".format(rec["name"], rec["subject"])


def test_dataset_shape_dtype_chunks_and_compression(built):
    fixations, _, _, _, h5_path = built
    with h5py.File(h5_path, "r") as f:
        dset = f["trials"]["heatmaps"]
        assert dset.shape == (len(fixations), 16, 24, 32)
        assert dset.dtype == np.float32
        assert dset.chunks == (1, 16, 24, 32)
        assert dset.compression == "gzip"


def test_root_attrs(built):
    _, _, _, _, h5_path = built
    with h5py.File(h5_path, "r") as f:
        assert list(f.attrs["origin_size"]) == [1080, 1920]
        assert list(f.attrs["action_map"]) == [24, 32]
        assert int(f.attrs["max_length"]) == 16
        assert float(f.attrs["blur_sigma"]) == 1.0
        datetime.datetime.strptime(str(f.attrs["created_utc"]), "%Y-%m-%dT%H:%M:%SZ")
        sha = str(f.attrs["fixations_sha256"])
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


def test_roundtrip_is_lossless(built):
    fixations, _, _, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)
    for rec in fixations:
        expected, _ = build_step_heatmaps(rec["X"], rec["Y"], rec["length"], **HM_KWARGS)
        assert np.array_equal(st.get(rec["name"], rec["subject"]), expected)


def test_missing_key_raises(built):
    _, _, _, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)
    with pytest.raises(KeyError) as exc:
        st.get("missing.jpg", 0)
    assert "missing.jpg|0" in str(exc.value)

    with pytest.raises(KeyError):
        st.get_batch([st.trial_keys[0], "missing.jpg|0"])


def test_get_batch_preserves_requested_order(built):
    _, _, _, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)
    k0, k1, k2 = st.trial_keys[0], st.trial_keys[1], st.trial_keys[2]
    out = st.get_batch([k2, k0, k1])
    assert np.array_equal(out[0], st.heatmaps[2])
    assert np.array_equal(out[1], st.heatmaps[0])
    assert np.array_equal(out[2], st.heatmaps[1])


def test_hash_mismatch_is_rejected(built, tmp_path):
    fixations, _, _, fx_path, h5_path = built
    GtHeatmapStore.load(h5_path, fixations_path=fx_path)      # the original: fine

    other = tmp_path / "other.json"
    reordered = list(reversed(fixations))
    other.write_text(json.dumps(reordered, indent=4))
    with pytest.raises(ValueError) as exc:
        GtHeatmapStore.load(h5_path, fixations_path=str(other))
    msg = str(exc.value)
    assert sha256_file(str(other)) in msg
    with h5py.File(h5_path, "r") as f:
        assert str(f.attrs["fixations_sha256"]) in msg


def test_build_is_deterministic():
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=12)
    fixations, exp_keys, subject_id_map, _ = build_fixations(
        bundle, support_pool_size=2)
    a = GtHeatmapStore.build(fixations, exp_keys, subject_id_map, attrs={}, **HM_KWARGS)
    b = GtHeatmapStore.build(fixations, exp_keys, subject_id_map, attrs={}, **HM_KWARGS)
    assert a.heatmaps.tobytes() == b.heatmaps.tobytes()


def test_size_guard(monkeypatch):
    bundle = make_fake_bundle(n_subjects=3, n_stimuli=5, seed=13)
    fixations, exp_keys, subject_id_map, _ = build_fixations(
        bundle, support_pool_size=2)
    monkeypatch.setattr(store_mod, "MAX_HEATMAP_BYTES", 10)
    with pytest.raises(ValueError) as exc:
        GtHeatmapStore.build(fixations, exp_keys, subject_id_map, attrs={}, **HM_KWARGS)
    assert "FR8.6" in str(exc.value)


def test_exp_key_roundtrip_and_uniqueness(built):
    fixations, exp_keys, subject_id_map, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)

    assert len(set(st.exp_keys)) == len(st.exp_keys)
    for i, key in enumerate(st.trial_keys):
        assert st.trial_key_of(st.exp_keys[i]) == key
        name, subject = key.rsplit("|", 1)
        assert st.exp_key_of(name, int(subject)) == st.exp_keys[i]

    with pytest.raises(KeyError):
        st.trial_key_of("not-a-key")

    # dense-id roundtrip: exp_keys are f"{subject}_{step}"
    for i in range(len(st.trial_keys)):
        eve_id = subject_id_map["to_eve"][str(int(st.subjects[i]))]
        assert st.exp_keys[i].startswith(eve_id + "_")


def test_no_phantom_keys(built):
    fixations, _, _, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)
    assert set(st.trial_keys) == {
        "{}|{}".format(r["name"], r["subject"]) for r in fixations}


def test_keys_carry_no_split_information(built):
    fixations, _, _, fx_path, h5_path = built
    st = GtHeatmapStore.load(h5_path, fixations_path=fx_path)
    train = next(r for r in fixations if r["split"] == "train")
    test = next(r for r in fixations if r["split"] == "test")
    for rec in (train, test):
        assert st.get(rec["name"], rec["subject"]).shape == (16, 24, 32)
        assert "train" not in "{}|{}".format(rec["name"], rec["subject"])
        assert "test" not in "{}|{}".format(rec["name"], rec["subject"])
