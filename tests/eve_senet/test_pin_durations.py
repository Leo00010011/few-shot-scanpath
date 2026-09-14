"""Validation Group 5's harness -- the duration-deadness pin (FR3.4).

Group 5 itself is GPU work: it compares real embedding tensors produced by three runs
of ``embed.py --duration-arm``. What is testable on the dev machine is the comparison
itself, and it is worth testing for one reason -- a pin that cannot fail proves
nothing. So the failing direction is asserted here as carefully as the passing one.
"""

import json
import os

import pytest
import torch

from eve_senet import EveSenetError
from eve_senet.durations import ABSURD_DURATION, absurd_scanpath_durations
from eve_senet.pin_durations import pin, sha256_file

from conftest import make_record


# ----------------------------------------------------------- the absurd arm itself

def test_absurd_arm_replaces_every_duration_without_mutating_the_input():
    recs = [make_record("a.jpg", 0, [1.0, 2.0], [3.0, 4.0], [150, 260])]
    out = absurd_scanpath_durations(recs)
    assert out[0]["T"] == [ABSURD_DURATION, ABSURD_DURATION]
    # copies, like bin_scanpath_durations -- the caller still reports milliseconds
    assert recs[0]["T"] == [150, 260]
    assert out[0] is not recs[0]


def test_absurd_arm_keeps_length_and_every_other_field():
    recs = [make_record("a.jpg", 7, [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [150, 260, 300])]
    out = absurd_scanpath_durations(recs)
    assert len(out[0]["T"]) == 3
    assert {k: v for k, v in out[0].items() if k != "T"} == \
           {k: v for k, v in recs[0].items() if k != "T"}


# --------------------------------------------------------------------- the pin

def _write(tmp_path, name, tensor):
    p = os.path.join(str(tmp_path), name)
    torch.save(tensor, p)
    return p


def test_identical_arms_pin_the_channel(tmp_path):
    t = torch.randn(38, 384)
    ref = _write(tmp_path, "ref.pt", t)
    arms = {"raw": _write(tmp_path, "raw.pt", t.clone()),
            "absurd": _write(tmp_path, "absurd.pt", t.clone())}
    s = pin(ref, arms)
    assert s["duration_channel_pinned"] is True
    assert all(a["bitwise_identical"] for a in s["arms"].values())
    assert s["reference"]["sha256"] == sha256_file(ref)


def test_a_one_ulp_difference_fails_the_pin(tmp_path):
    """The whole point: torch.equal, not allclose. A tolerance would pass this."""
    t = torch.randn(38, 384)
    live = t.clone()
    live[3, 17] = torch.nextafter(live[3, 17], torch.tensor(float("inf")))
    ref = _write(tmp_path, "ref.pt", t)
    arms = {"absurd": _write(tmp_path, "absurd.pt", live)}
    assert torch.allclose(t, live)          # a tolerance check would be green here
    with pytest.raises(EveSenetError) as exc:
        pin(ref, arms)
    assert "LIVE" in str(exc.value) and "absurd" in str(exc.value)


def test_failure_is_still_reported_when_not_raising(tmp_path):
    t = torch.randn(4, 384)
    ref = _write(tmp_path, "ref.pt", t)
    arms = {"raw": _write(tmp_path, "raw.pt", t + 1.0)}
    s = pin(ref, arms, raise_on_diff=False)
    assert s["duration_channel_pinned"] is False
    assert s["arms"]["raw"]["bitwise_identical"] is False
    assert s["arms"]["raw"]["max_abs_diff"] == pytest.approx(1.0)


def test_no_arm_raises_rather_than_vacuously_passing(tmp_path):
    ref = _write(tmp_path, "ref.pt", torch.randn(4, 384))
    with pytest.raises(EveSenetError):
        pin(ref, {})


def test_shape_mismatch_raises_instead_of_comparing(tmp_path):
    ref = _write(tmp_path, "ref.pt", torch.randn(38, 384))
    arms = {"raw": _write(tmp_path, "raw.pt", torch.randn(10, 384))}
    with pytest.raises(EveSenetError) as exc:
        pin(ref, arms)
    assert "cohort" in str(exc.value)


def test_missing_tensor_raises_our_error(tmp_path):
    ref = _write(tmp_path, "ref.pt", torch.randn(4, 384))
    with pytest.raises(EveSenetError):
        pin(ref, {"raw": os.path.join(str(tmp_path), "nope.pt")})


# ------------------------------------------------------- the senet_report stamp

def test_report_is_stamped_on_success(tmp_path):
    t = torch.randn(38, 384)
    ref = _write(tmp_path, "ref.pt", t)
    arms = {"absurd": _write(tmp_path, "absurd.pt", t.clone())}
    rp = os.path.join(str(tmp_path), "senet_report.json")
    with open(rp, "w") as fh:
        json.dump({"num_subjects": 38}, fh)

    pin(ref, arms, report_path=rp)
    got = json.load(open(rp))
    assert got["num_subjects"] == 38            # nothing else was clobbered
    assert got["duration_channel_pinned"] is True
    assert got["duration_pin"]["arms"]["absurd"]["bitwise_identical"] is True
    assert got["duration_pin"]["reference_sha256"] == sha256_file(ref)


def test_report_is_stamped_on_failure_too(tmp_path):
    """A failed pin is the more important artefact, not the less."""
    t = torch.randn(4, 384)
    ref = _write(tmp_path, "ref.pt", t)
    arms = {"raw": _write(tmp_path, "raw.pt", t + 1.0)}
    rp = os.path.join(str(tmp_path), "senet_report.json")
    with open(rp, "w") as fh:
        json.dump({"num_subjects": 4}, fh)

    with pytest.raises(EveSenetError):
        pin(ref, arms, report_path=rp)
    got = json.load(open(rp))
    assert got["duration_channel_pinned"] is False
