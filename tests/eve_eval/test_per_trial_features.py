"""Group 3 -- per-trial feature loading: OPEN-6 arriving at Stage D.

EVE presents each photograph at a per-trial display scale, so one tensor per stimulus
name cannot represent what every participant saw. F4 measured the difference on real
tensors (within-name cosine median 0.775 against a cross-stimulus 0.450). These tests
pin that the loader actually uses it.
"""

import os

import pytest
import torch

from conftest import exp_key_for


def test_two_subjects_on_one_image_get_different_tensors(make_dataset):
    """FR5.1 -- and the second half states what upstream's behaviour would be.

    Under the per-image load the three slots hold three references to ONE tensor, and
    every subject sees the first participant's screen. The test documents the
    difference rather than only the fix.
    """
    ds = make_dataset()
    item = ds[0]
    assert not torch.equal(item["image"][0], item["image"][1])
    assert not torch.equal(item["image"][1], item["image"][2])

    # The upstream shape, simulated: one key for the whole image.
    name = item["img_name"]
    per_image = {(name, int(s)): exp_key_for(name, int(item["subject"][0]))
                 for s in item["subject"]}
    upstream_like = make_dataset(exp_keys={**dict(ds.exp_key_map), **per_image})
    same = upstream_like[0]
    assert torch.equal(same["image"][0], same["image"][1])


def test_path_is_built_by_concatenation(make_dataset, artefacts, monkeypatch):
    """FR5.1 / FR2.5 -- join(feature_dir, exp_key + '.pth'), for every trial."""
    ds = make_dataset()
    seen = []
    real_load = torch.load

    def spy(path, *a, **kw):
        seen.append(path)
        return real_load(path, *a, **kw)

    monkeypatch.setattr(torch, "load", spy)
    for idx in range(len(ds)):
        item = ds[idx]
        name = item["img_name"]
        for subject in item["subject"]:
            expected = os.path.join(artefacts["feat_dir"],
                                    exp_key_for(name, int(subject)) + ".pth")
            assert expected in seen, (expected, seen[-3:])
    assert len(seen) == 18


def test_exp_key_containing_jpg_raises(make_dataset):
    """An unsafe key is rejected BEFORE any load (FR5.1, TechStack section 3.2)."""
    from eve_prep import EvePrepError

    ds = make_dataset()
    name = ds.imgid[0]
    subject = int(ds.fixations[ds.imgid_to_sub[name][0]]["subject"])
    poisoned = dict(ds.exp_key_map)
    poisoned[(name, subject)] = "train00_jpg_step1"
    bad = make_dataset(exp_keys=poisoned)
    with pytest.raises(EvePrepError) as exc:
        bad[bad.imgid.index(name)]
    assert "jpg" in str(exc.value)


def test_missing_mapping_raises_naming_both(make_dataset):
    """FR5.3 -- never a fallback key, never a sibling subject's tensor."""
    ds = make_dataset()
    name = ds.imgid[0]
    subject = int(ds.fixations[ds.imgid_to_sub[name][0]]["subject"])
    holed = {k: v for k, v in ds.exp_key_map.items() if k != (name, subject)}
    broken = make_dataset(exp_keys=holed)
    with pytest.raises(KeyError) as exc:
        broken[broken.imgid.index(name)]
    message = str(exc.value)
    assert name in message and str(subject) in message


@pytest.mark.parametrize("shape,dtype,expect", [
    ((768, 1024), torch.float32, "(768, 1024)"),
    ((768, 2048), torch.float64, "float64"),
])
def test_wrong_shape_or_dtype_raises(make_dataset, artefacts, tmp_path, shape, dtype,
                                     expect):
    """FR5.2 -- naming the exp_key, the expected and the actual (D7)."""
    ds = make_dataset()
    name = ds.imgid[0]
    subject = int(ds.fixations[ds.imgid_to_sub[name][0]]["subject"])
    key = exp_key_for(name, subject)

    # Only the three tensors this image needs are materialised: the cache is 145 MB
    # and the loader touches one image's worth per __getitem__.
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for ids in ds.imgid_to_sub[name]:
        rec = ds.fixations[ids]
        rel = exp_key_for(rec["name"], int(rec["subject"])) + ".pth"
        if rel == key + ".pth":
            torch.save(torch.zeros(shape, dtype=dtype), str(feat_dir / rel))
        else:
            torch.save(torch.zeros((768, 2048), dtype=torch.float32),
                       str(feat_dir / rel))

    broken = make_dataset(feature_dir=str(feat_dir))
    with pytest.raises(ValueError) as exc:
        broken[broken.imgid.index(name)]
    message = str(exc.value)
    assert key in message and "(768, 2048)" in message and expect in message
