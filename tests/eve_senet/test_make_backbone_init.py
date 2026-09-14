"""F3 preflight -- deriving data/M2F_R50.pkl from the released checkpoint.

The real check that the derived file is *complete* is the strict load inside
``ImageFeatureEncoder`` on the cluster. What is testable here is that the extraction
takes exactly the backbone, strips exactly the prefix, and refuses the two cases that
would produce a plausible-but-wrong file: a checkpoint with no backbone at all, and
silently clobbering a pickle that is already there.
"""

import os

import pytest
import torch

from eve_senet import EveSenetError
from eve_senet.make_backbone_init import (BACKBONE_PREFIX, extract_backbone,
                                          write_backbone_init)


def _ckpt(tmp_path, state, wrap=True):
    p = os.path.join(str(tmp_path), "ckp.pt")
    torch.save({"model": state, "optimizer": {}, "step": 11999} if wrap else state, p)
    return p


def _senet_like():
    return {
        "encoder.backbone.stem.conv1.weight": torch.randn(2, 2),
        "encoder.backbone.stages.res2.0.shortcut.weight": torch.randn(2, 2),
        "encoder.pixel_decoder.thing": torch.randn(2),
        "subject_predictor.weight": torch.randn(10, 4),
    }


def test_extracts_only_the_backbone_and_strips_the_prefix(tmp_path):
    got = extract_backbone(_ckpt(tmp_path, _senet_like()))
    assert set(got) == {"stem.conv1.weight", "stages.res2.0.shortcut.weight"}
    assert all(not k.startswith(BACKBONE_PREFIX) for k in got)


def test_accepts_a_bare_state_dict_as_well_as_a_wrapped_one(tmp_path):
    got = extract_backbone(_ckpt(tmp_path, _senet_like(), wrap=False))
    assert set(got) == {"stem.conv1.weight", "stages.res2.0.shortcut.weight"}


def test_a_checkpoint_without_a_backbone_raises_rather_than_writing_an_empty_file(tmp_path):
    p = _ckpt(tmp_path, {"subject_predictor.weight": torch.randn(10, 4)})
    with pytest.raises(EveSenetError) as exc:
        extract_backbone(p)
    assert "Mask2Former" in str(exc.value)


def test_a_bare_res_key_raises_because_models_py_would_rename_it(tmp_path):
    """models.py renames leading-'res' keys; ours must already be 'stages.res*'."""
    p = _ckpt(tmp_path, {"encoder.backbone.res2.0.conv1.weight": torch.randn(2, 2)})
    with pytest.raises(EveSenetError) as exc:
        extract_backbone(p)
    assert "rename" in str(exc.value)


def test_written_file_round_trips_to_the_same_tensors(tmp_path):
    state = _senet_like()
    src = _ckpt(tmp_path, state)
    out = os.path.join(str(tmp_path), "M2F_R50.pkl")
    summary = write_backbone_init(src, out)
    assert summary["n_tensors"] == 2
    got = torch.load(out, map_location="cpu")
    assert torch.equal(got["stem.conv1.weight"],
                       state["encoder.backbone.stem.conv1.weight"])
    assert summary["top_level_groups"] == ["stages", "stem"]


def test_an_existing_pickle_is_never_overwritten_without_force(tmp_path):
    src = _ckpt(tmp_path, _senet_like())
    out = os.path.join(str(tmp_path), "M2F_R50.pkl")
    with open(out, "wb") as fh:
        fh.write(b"the authors' own file")
    with pytest.raises(EveSenetError) as exc:
        write_backbone_init(src, out)
    assert "--force" in str(exc.value)
    assert open(out, "rb").read() == b"the authors' own file"

    write_backbone_init(src, out, force=True)
    assert torch.is_tensor(torch.load(out, map_location="cpu")["stem.conv1.weight"])


def test_missing_checkpoint_raises_our_error(tmp_path):
    with pytest.raises(EveSenetError):
        extract_backbone(os.path.join(str(tmp_path), "nope.pt"))


# ------------------------------------------------------------ the pixel decoder

from eve_senet.make_backbone_init import (component_path, extract_component,
                                          write_all)


def _full_senet_like():
    d = _senet_like()
    d.update({
        "encoder.pixel_decoder.lateral_convs.adapter_1.weight": torch.randn(2, 2),
        "encoder.pixel_decoder.output_convs.layer_1.weight": torch.randn(2, 2),
        "encoder.pixel_decoder.transformer.layers.0.weight": torch.randn(2),
    })
    d.pop("encoder.pixel_decoder.thing")
    return d


def test_pixel_decoder_filename_is_derived_exactly_as_models_py_derives_it():
    # models.py: cfg.MODEL.WEIGHTS[:-4] + '_MSDeformAttnPixelDecoder.pkl'
    assert component_path("data/M2F_R50.pkl", "pixel_decoder") == \
        "data/M2F_R50_MSDeformAttnPixelDecoder.pkl"
    assert component_path("data/M2F_R50.pkl", "backbone") == "data/M2F_R50.pkl"


def test_extracts_the_pixel_decoder_subtree(tmp_path):
    got = extract_component(_ckpt(tmp_path, _full_senet_like()), "pixel_decoder")
    assert set(got) == {"lateral_convs.adapter_1.weight",
                        "output_convs.layer_1.weight",
                        "transformer.layers.0.weight"}


def test_pre_rename_decoder_keys_raise(tmp_path):
    """A bare 'adapter*' key would be rewritten by models.py and then not match."""
    p = _ckpt(tmp_path, {"encoder.pixel_decoder.adapter_1.weight": torch.randn(2, 2)})
    with pytest.raises(EveSenetError) as exc:
        extract_component(p, "pixel_decoder")
    assert "pre-rename" in str(exc.value)


def test_write_all_produces_both_files(tmp_path):
    src = _ckpt(tmp_path, _full_senet_like())
    out = os.path.join(str(tmp_path), "M2F_R50.pkl")
    summary = write_all(src, out)
    names = {c["component"]: c["out"] for c in summary["components"]}
    assert set(names) == {"backbone", "pixel_decoder"}
    assert all(os.path.isfile(p) for p in names.values())
    assert names["pixel_decoder"].endswith("_MSDeformAttnPixelDecoder.pkl")
    assert summary["checkpoint_sha256"]


def test_unknown_component_raises(tmp_path):
    with pytest.raises(EveSenetError):
        extract_component(_ckpt(tmp_path, _full_senet_like()), "transformer_decoder")
