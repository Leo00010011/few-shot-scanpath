"""Validation Groups 2 and 3 -- the trial -> exp_key mapping and the filename rule."""

import json
import os

import pytest

from conftest import TRIALS, write_fixations, write_store
from eve_prep import EvePrepError
from eve_prep.trial_keys import (crosscheck_exp_keys, derive_trial_exp_keys,
                                 exp_key_filename, load_trial_exp_keys,
                                 parse_trial_key, split_of)


# --------------------------------------------------------------------------
# Group 2 -- the mapping
# --------------------------------------------------------------------------

def test_load_returns_typed_mapping(bridge, exp_keys):
    got = load_trial_exp_keys(bridge["heatmaps"], bridge["fixations"])
    assert got == exp_keys
    assert len(got) == len(TRIALS)
    for (name, subject), key in got.items():
        assert isinstance(name, str)
        assert isinstance(subject, int)
        assert isinstance(key, str)


def test_parse_trial_key_splits_on_the_last_separator():
    """FR2.2 -- a stimulus name containing the separator must not corrupt the id."""
    assert parse_trial_key("a|b|c.jpg|7") == ("a|b|c.jpg", 7)
    assert parse_trial_key("plain.jpg|0") == ("plain.jpg", 0)


@pytest.mark.parametrize("bad", ["noseparator", "x|notanint", "|7", "x|"])
def test_parse_trial_key_rejects_malformed(bad):
    with pytest.raises(EvePrepError):
        parse_trial_key(bad)


def test_duplicate_exp_key_raises(tmp_path, fixations, exp_keys):
    """FR2.3 -- two trials sharing one screen capture."""
    fix = write_fixations(tmp_path / "fixations.json", fixations)
    keys = [exp_keys[(r["name"], int(r["subject"]))] for r in fixations]
    keys[1] = keys[0]
    store = write_store(tmp_path / "gt.h5", fixations, exp_keys,
                        fixations_path=fix, overrides=keys)
    with pytest.raises(EvePrepError) as exc:
        load_trial_exp_keys(store, fix)
    assert "duplicate exp_key" in str(exc.value)
    assert keys[0] in str(exc.value)


def test_sha_mismatch_raises_and_shows_both_hashes(tmp_path, fixations, exp_keys):
    """FR2.5 -- the store and the records would describe different builds."""
    fix = write_fixations(tmp_path / "fixations.json", fixations)
    store = write_store(tmp_path / "gt.h5", fixations, exp_keys, fixations_path=fix)
    with open(fix, "w") as fh:                      # same trials, different bytes
        json.dump(fixations, fh, indent=2)
    with pytest.raises(EvePrepError) as exc:
        load_trial_exp_keys(store, fix)
    msg = str(exc.value)
    assert "hash mismatch" in msg
    assert msg.count("46") >= 0                     # both hashes are quoted
    assert len([tok for tok in msg.split() if len(tok.strip(".,")) == 64]) == 2


def test_loader_never_reads_the_heatmaps_dataset(bridge, exp_keys, monkeypatch):
    """FR2.1 -- the ~88 MB array is never materialised.

    The source-grep version validation offers as the cheap alternative cannot work
    here: the file is *called* ``gt_heatmaps.h5``, so the substring is unavoidable in
    the docstring and in every path. This does the real thing instead -- any
    subscript of a dataset whose name contains ``heatmaps`` raises, and the loader
    must still complete. It is what F5's reuse actually depends on.
    """
    import h5py

    original = h5py.Dataset.__getitem__

    def guarded(self, item):
        if "heatmaps" in self.name:
            raise AssertionError(
                "load_trial_exp_keys read {} -- ~88 MB it has no use for".format(
                    self.name))
        return original(self, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    assert load_trial_exp_keys(bridge["heatmaps"], bridge["fixations"]) == exp_keys


def test_derivation_reproduces_the_store(bridge, samples_df, fixations,
                                         subject_id_map, exp_keys):
    derived = derive_trial_exp_keys(samples_df, fixations, subject_id_map)
    assert derived == exp_keys


def test_derivation_raises_on_zero_matches(samples_df, fixations, subject_id_map):
    df = samples_df[samples_df["stimulus_name"] != "shared_a"]
    with pytest.raises(EvePrepError) as exc:
        derive_trial_exp_keys(df, fixations, subject_id_map)
    assert "0 rows in samples_df" in str(exc.value)


def test_derivation_raises_on_two_matches(samples_df, fixations, subject_id_map):
    import pandas as pd
    dup = samples_df.iloc[[0]].copy()
    dup["exp_key"] = "train00_step999"
    df = pd.concat([samples_df, dup], ignore_index=True)
    with pytest.raises(EvePrepError) as exc:
        derive_trial_exp_keys(df, fixations, subject_id_map)
    assert "2 rows in samples_df" in str(exc.value)


def test_crosscheck_agrees(exp_keys):
    got = crosscheck_exp_keys(dict(exp_keys), dict(exp_keys))
    assert got["agree"] is True
    assert got["n"] == len(exp_keys)


def test_crosscheck_three_distinguishable_failures(exp_keys):
    """FR2.4 -- the three mean three different upstream problems."""
    key = ("shared_a.jpg", 0)

    only_store = dict(exp_keys)
    derived = dict(exp_keys)
    derived.pop(key)
    with pytest.raises(EvePrepError) as exc_store:
        crosscheck_exp_keys(only_store, derived)

    authoritative = dict(exp_keys)
    authoritative.pop(key)
    with pytest.raises(EvePrepError) as exc_derived:
        crosscheck_exp_keys(authoritative, dict(exp_keys))

    disagreeing = dict(exp_keys)
    disagreeing[key] = "train00_step999"
    with pytest.raises(EvePrepError) as exc_value:
        crosscheck_exp_keys(dict(exp_keys), disagreeing)

    messages = {str(exc_store.value), str(exc_derived.value), str(exc_value.value)}
    assert len(messages) == 3
    assert "but not derivable from samples_df" in str(exc_store.value)
    assert "but absent from gt_heatmaps.h5" in str(exc_derived.value)
    assert "disagree" in str(exc_value.value)


def test_split_of(fixations):
    got = split_of(fixations)
    assert got[("shared_a.jpg", 0)] == "test"
    assert got[("support_x.jpg", 0)] == "train"
    assert len(got) == len(TRIALS)


def test_split_of_raises_on_duplicate_trial(fixations):
    with pytest.raises(EvePrepError) as exc:
        split_of(fixations + [fixations[0]])
    assert "duplicate trial" in str(exc.value)


# --------------------------------------------------------------------------
# Group 3 -- the filename rule
# --------------------------------------------------------------------------

def test_exp_key_filename_appends_pth():
    assert exp_key_filename("train24_step059") == "train24_step059.pth"


def test_exp_key_filename_rejects_jpg_substring():
    """FR5.2 -- the message must cite the unanchored replace it protects against."""
    with pytest.raises(EvePrepError) as exc:
        exp_key_filename("has_jpg_inside")
    assert "str.replace('jpg', 'pth')" in str(exc.value)


@pytest.mark.parametrize("bad", ["bad-key!", "with space", "", "a/b", "dot.key"])
def test_exp_key_filename_rejects_charset(bad):
    with pytest.raises(EvePrepError):
        exp_key_filename(bad)


def test_exp_key_filename_round_trips(exp_keys):
    for key in exp_keys.values():
        assert exp_key_filename(key)[:-4] == key


@pytest.mark.bridge
def test_real_mapping_round_trips(bridge_dir):
    """Group 3's round-trip and Group 6's charset check, on the real artefacts."""
    mapping = load_trial_exp_keys(
        os.path.join(bridge_dir, "gt_heatmaps.h5"),
        os.path.join(bridge_dir, "fixations.json"))
    assert len(mapping) == 1804
    assert len(set(mapping.values())) == 1804
    for key in mapping.values():
        assert exp_key_filename(key)[:-4] == key


@pytest.mark.bridge
def test_real_mapping_split_counts(bridge_dir):
    with open(os.path.join(bridge_dir, "fixations.json")) as fh:
        fixations = json.load(fh)
    with open(os.path.join(bridge_dir, "bridge_report.json")) as fh:
        report = json.load(fh)
    splits = split_of(fixations)
    n_test = sum(1 for s in splits.values() if s == "test")
    n_train = sum(1 for s in splits.values() if s == "train")
    assert (n_test, n_train) == (report["num_trials_test"],
                                 report["num_trials_train"])
    assert n_test + n_train == report["num_trials"] == 1804
