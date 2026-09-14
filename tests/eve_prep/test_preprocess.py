"""Validation Group 1 -- preprocessing parity, the load-bearing group.

The first test is FR4.4 and is **unmarked** (FR14.3): it needs only a synthetic image
and a CPU backbone. If it fails, the FR4.3 transcription is wrong and nothing else in
F4 is trustworthy.
"""

import numpy as np
import PIL.Image
import pytest
import torch

from eve_prep import EvePrepError, FEATURE_SHAPE, STIMULUS_SHAPE
from eve_prep.extract_features import (build_backbone, extract_one,
                                       import_backbone_class, preprocess)


def _probe_array(seed=0):
    return (np.random.RandomState(seed).rand(*STIMULUS_SHAPE) * 255).astype(np.uint8)


def test_preprocess_matches_upstream_image_data(tmp_path):
    """FR4.4 -- bit-identity with ``image_data()``. Exact equality, not allclose."""
    import_backbone_class()          # registers the sentence_transformers stub first
    from preprocess.feature_extractor import image_data

    src = tmp_path / "train"
    src.mkdir()
    out = tmp_path / "out"
    (out / "image_features").mkdir(parents=True)

    PIL.Image.fromarray(_probe_array()).save(src / "probe.jpg", quality=95,
                                             subsampling=0)

    device = torch.device("cpu")
    image_data(dataset_path=str(tmp_path), output_path=str(out), device=device,
               overwrite=True)
    upstream = torch.load(str(out / "image_features" / "probe.pth"),
                          map_location="cpu")

    # From the array the JPEG DECODES to, never the pre-encode one: JPEG is lossy and
    # that failure would look like a transform defect without being one.
    decoded = np.array(PIL.Image.open(src / "probe.jpg").convert("RGB"),
                       dtype=np.uint8)
    ours = extract_one(build_backbone(device), decoded)

    assert torch.equal(upstream, ours)


def test_preprocess_shape_and_dtype():
    out = preprocess(_probe_array())
    assert tuple(out.shape) == (1, 3, 768, 1024)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("arr", [
    np.zeros((1080, 1920), dtype=np.uint8),
    np.zeros((1080, 1920, 4), dtype=np.uint8),
    np.zeros((600, 800, 3), dtype=np.uint8),
    np.zeros(STIMULUS_SHAPE, dtype=np.float32),
])
def test_preprocess_rejects_bad_input(arr):
    """FR3.2 -- four separate cases; the message names the shape."""
    with pytest.raises(EvePrepError) as exc:
        preprocess(arr)
    assert str(tuple(arr.shape)) in str(exc.value)


def test_preprocess_does_not_mutate_its_input():
    arr = _probe_array(1)
    before = arr.copy()
    preprocess(arr)
    preprocess(arr)
    assert np.array_equal(arr, before)


def test_extract_one_returns_cpu_float32():
    out = extract_one(build_backbone(torch.device("cpu")), _probe_array(2))
    assert tuple(out.shape) == FEATURE_SHAPE
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert out.is_cuda is False


def test_extract_one_rejects_wrong_backbone_output():
    class _Wrong(object):
        def __call__(self, x):
            return torch.zeros((1, 700, 2048), dtype=torch.float32)

    with pytest.raises(EvePrepError) as exc:
        extract_one(_Wrong(), _probe_array(3))
    assert "(768, 2048)" in str(exc.value)
