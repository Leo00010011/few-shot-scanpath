"""Validation Group 2 -- support selection (FR4)."""

import json
import os
import random
import subprocess
import sys

import pytest

from eve_senet import EveSenetError
from eve_senet.embed import assert_stimuli_exist, select_support

from conftest import BRIDGE_DIR, make_record

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tools")


def test_shape_of_the_selection(records):
    sel = select_support(records, 10, 0)
    assert sorted(sel) == list(range(38))
    for s, names in sel.items():
        assert len(names) == 10
        assert len(set(names)) == 10  # without replacement


def test_repeatable_within_a_process(records):
    assert select_support(records, 10, 0) == select_support(records, 10, 0)


def test_repeatable_across_processes(records, tmp_path):
    """The seed is a *string* fed to random.Random, not the interpreter's hash seed --
    a PYTHONHASHSEED-dependent draw would be irreproducible (FR4.2)."""
    path = tmp_path / "recs.json"
    path.write_text(json.dumps(records))
    code = (
        "import sys, json; sys.path.insert(0, %r);"
        "from eve_senet.embed import select_support;"
        "recs = json.load(open(%r));"
        "print(json.dumps({str(k): v for k, v in select_support(recs, 10, 0).items()}))"
    ) % (TOOLS, str(path))
    env = dict(os.environ, PYTHONHASHSEED="99991")
    out = json.loads(subprocess.check_output([sys.executable, "-c", code], env=env))
    assert out == {str(k): v for k, v in select_support(records, 10, 0).items()}


def test_a_different_seed_is_a_different_draw(records):
    """If a seed change moves nothing, support_pool_size = 20 > num_fewshot = 10 is
    buying nothing and OPEN-3's averaging plan is void."""
    a, b = select_support(records, 10, 0), select_support(records, 10, 1)
    assert sum(1 for s in a if a[s] != b[s]) >= 30


def test_input_order_cannot_leak_in(records):
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    assert select_support(shuffled, 10, 0) == select_support(records, 10, 0)


def test_a_thin_subject_raises_and_is_named(records):
    thin = ([r for r in records if r["subject"] != 5]
            + [r for r in records if r["subject"] == 5][:9])
    with pytest.raises(EveSenetError) as exc:
        select_support(thin, 10, 0)
    assert "5" in str(exc.value) and "num_fewshot" in str(exc.value)


def test_a_planted_test_record_raises_rather_than_being_filtered(records):
    leaked = list(records)
    leaked[0] = dict(leaked[0], split="test")
    with pytest.raises(EveSenetError) as exc:
        select_support(leaked, 10, 0)
    assert "split" in str(exc.value)  # FR11.4 -- a leak, never a filter


def test_every_selected_name_resolves_to_a_train_record(records):
    sel = select_support(records, 10, 0)
    by_key = {(r["name"], r["subject"]): r for r in records}
    for s, names in sel.items():
        for n in names:
            assert by_key[(n, s)]["split"] == "train"


def test_missing_stimulus_raises_with_the_name(records, stub_images):
    sel = select_support(records, 10, 0)
    os.remove(os.path.join(stub_images, sel[0][0]))
    with pytest.raises(EveSenetError) as exc:
        assert_stimuli_exist(sel, stub_images)
    assert sel[0][0] in str(exc.value)  # FR11.3


@pytest.mark.bridge
def test_real_pools_yield_37_of_38_different_draws_across_seeds():
    """Subject 34 has exactly 10 support stimuli and must draw the same 10 at both
    seeds; every other subject's pool is larger and must move."""
    with open(os.path.join(BRIDGE_DIR, "fixations.json")) as fh:
        train = [r for r in json.load(fh) if r["split"] == "train"]
    a, b = select_support(train, 10, 0), select_support(train, 10, 1)
    moved = [s for s in a if a[s] != b[s]]
    assert len(moved) == 37
    assert 34 not in moved


@pytest.mark.bridge
def test_real_selection_is_disjoint_from_the_scored_split():
    """Re-checked on the artefact F3 consumes, rather than trusted from the report."""
    with open(os.path.join(BRIDGE_DIR, "fixations.json")) as fh:
        recs = json.load(fh)
    train = [r for r in recs if r["split"] == "train"]
    test_names = {r["name"] for r in recs if r["split"] == "test"}
    sel = select_support(train, 10, 0)
    assert not ({n for names in sel.values() for n in names} & test_names)
