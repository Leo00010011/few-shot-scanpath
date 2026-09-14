"""The detectron2 ResNet stage-naming shim (cluster finding, 2026-09-14).

The released SE-Net checkpoint stores its backbone under `stages.res{2..5}.*`;
detectron2 0.6's `build_resnet_backbone` produces bare `res{2..5}.*`. Identical
tensors, identical shapes, different names. `align_stage_prefix` translates the names
in whichever direction the *installed* detectron2 needs -- so these tests pin both
directions and, most importantly, the no-op case: under the authors' own detectron2
this code must not touch anything.
"""

import pytest

from eve_senet import EveSenetError
from eve_senet.embed import align_stage_prefix

BARE = ["stem.conv1.weight", "res2.0.conv1.weight", "res5.2.conv3.norm.bias"]
STAGED = ["stem.conv1.weight", "stages.res2.0.conv1.weight",
          "stages.res5.2.conv3.norm.bias"]


def _sd(keys):
    return {k: i for i, k in enumerate(keys)}


def test_strips_the_segment_when_the_module_wants_bare_names():
    out, n, direction = align_stage_prefix(_sd(STAGED), BARE)
    assert direction == "strip" and n == 2
    assert sorted(out) == sorted(BARE)


def test_adds_the_segment_when_the_module_wants_staged_names():
    out, n, direction = align_stage_prefix(_sd(BARE), STAGED)
    assert direction == "add" and n == 2
    assert sorted(out) == sorted(STAGED)


@pytest.mark.parametrize("keys", [BARE, STAGED])
def test_is_a_no_op_when_the_two_already_agree(keys):
    """Under the authors' own detectron2 this must change nothing at all."""
    out, n, direction = align_stage_prefix(_sd(keys), keys)
    assert direction == "none" and n == 0
    assert out == _sd(keys)


def test_values_are_never_touched_only_names():
    state = {"stages.res2.0.conv1.weight": "tensor-A", "stem.conv1.weight": "tensor-B"}
    out, _, _ = align_stage_prefix(state, BARE)
    assert out["res2.0.conv1.weight"] == "tensor-A"
    assert out["stem.conv1.weight"] == "tensor-B"


def test_non_stage_keys_pass_through_untouched():
    state = _sd(STAGED + ["stem.conv1.norm.running_var", "res_something_else.w"])
    out, _, direction = align_stage_prefix(state, BARE)
    assert direction == "strip"
    assert "stem.conv1.norm.running_var" in out
    # a bare key that merely starts with "res" is already in the target convention
    assert "res_something_else.w" in out


def test_scope_confines_the_rename_to_the_backbone():
    """Inside a whole-model checkpoint, only encoder.backbone.* may be rewritten."""
    state = _sd(["encoder.backbone.stages.res2.0.conv1.weight",
                 "encoder.backbone.stem.conv1.weight",
                 "subject_predictor.weight",
                 "encoder.pixel_decoder.stages.res2.x"])
    reference = ["encoder.backbone.res2.0.conv1.weight",
                 "encoder.backbone.stem.conv1.weight",
                 "subject_predictor.weight"]
    out, n, direction = align_stage_prefix(state, reference, scope="encoder.backbone.")
    assert direction == "strip" and n == 1
    assert "encoder.backbone.res2.0.conv1.weight" in out
    assert "subject_predictor.weight" in out
    # a same-named segment elsewhere in the model is out of scope and stays put
    assert "encoder.pixel_decoder.stages.res2.x" in out


def test_a_colliding_rename_raises_rather_than_dropping_a_tensor():
    state = _sd(["stages.res2.0.conv1.weight", "res2.0.conv1.weight"])
    with pytest.raises(EveSenetError) as exc:
        align_stage_prefix(state, BARE)
    assert "drop a backbone tensor" in str(exc.value)


def test_key_count_is_preserved_in_both_directions():
    for state_keys, ref in ((STAGED, BARE), (BARE, STAGED)):
        out, _, _ = align_stage_prefix(_sd(state_keys), ref)
        assert len(out) == len(state_keys)
