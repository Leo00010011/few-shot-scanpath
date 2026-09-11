"""FR9 -- the independent structural verifier. CPU-only, Windows-runnable."""

import json
import os

import pytest
import torch

from eve_senet import EveSenetError
from eve_senet.verify_embedding import verify

N, DIM = 38, 384


@pytest.fixture
def subject_map(tmp_path):
    to_eve = {str(i): "p{:02d}".format(i) for i in range(N)}
    path = tmp_path / "subject_id_map.json"
    path.write_text(json.dumps({
        "to_eve": to_eve,
        "to_dense": {v: int(k) for k, v in to_eve.items()},
    }))
    return str(path)


def _save(tmp_path, tensor, name="eve_fewshot_user_embedding_10_seed0.pt"):
    path = tmp_path / name
    torch.save(tensor, path)
    return str(path)


@pytest.fixture
def good(tmp_path):
    torch.manual_seed(0)
    return _save(tmp_path, torch.randn(N, DIM, dtype=torch.float32))


def test_a_good_tensor_verifies_and_answers_D4(good, subject_map):
    s = verify(good, subject_map)
    assert s["shape"] == [N, DIM] and s["dtype"] == "torch.float32"
    assert len(s["rows"]) == N
    assert s["rows"][3] == {"dense_id": 3, "eve_id": "p03",
                            "norm": pytest.approx(s["rows"][3]["norm"])}
    # FR9.3 -- reported, never gated
    assert -1.0 <= s["offdiag_cosine"]["min"] <= s["offdiag_cosine"]["max"] <= 1.0


def test_the_released_osie_shape_is_rejected(tmp_path, subject_map):
    """A (10, 384) here means the released OSIE tensor was loaded by mistake."""
    path = _save(tmp_path, torch.randn(10, DIM))
    with pytest.raises(EveSenetError) as exc:
        verify(path, subject_map)
    assert "(10, 384)" in str(exc.value) or "10" in str(exc.value)


def test_a_zero_row_is_rejected(tmp_path, subject_map):
    """The released fewshot_user_embedding_10.pt has 5 zero rows; ours must have none."""
    t = torch.randn(N, DIM)
    t[7] = 0
    with pytest.raises(EveSenetError) as exc:
        verify(_save(tmp_path, t), subject_map)
    assert "p07" in str(exc.value)


def test_duplicate_rows_are_rejected(tmp_path, subject_map):
    t = torch.randn(N, DIM)
    t[11] = t[4]
    with pytest.raises(EveSenetError) as exc:
        verify(_save(tmp_path, t), subject_map)
    assert "identical" in str(exc.value)


def test_non_finite_is_rejected(tmp_path, subject_map):
    t = torch.randn(N, DIM)
    t[2, 2] = float("nan")
    with pytest.raises(EveSenetError):
        verify(_save(tmp_path, t), subject_map)


def test_a_wrong_dtype_is_rejected(tmp_path, subject_map):
    with pytest.raises(EveSenetError):
        verify(_save(tmp_path, torch.randn(N, DIM, dtype=torch.float64)), subject_map)


def test_a_broken_subject_map_roundtrip_is_rejected(tmp_path, good):
    to_eve = {str(i): "p{:02d}".format(i) for i in range(N)}
    bad = tmp_path / "bad_map.json"
    to_dense = {v: int(k) for k, v in to_eve.items()}
    to_dense["p03"] = 4  # D4's core claim, broken
    bad.write_text(json.dumps({"to_eve": to_eve, "to_dense": to_dense}))
    with pytest.raises(EveSenetError) as exc:
        verify(good, str(bad))
    assert "roundtrip" in str(exc.value)


def test_seed_in_the_filename_must_agree_with_the_report(tmp_path, good, subject_map):
    """The guard aggregate_seeds.py needed for F1: a mis-targeted copy step must not
    let F5 pool non-replicates."""
    report = tmp_path / "senet_report.json"
    report.write_text(json.dumps({
        "embedding_file": os.path.basename(good), "args": {"seed": 1}}))
    with pytest.raises(EveSenetError) as exc:
        verify(good, subject_map, str(report))
    assert "seed" in str(exc.value)

    report.write_text(json.dumps({
        "embedding_file": os.path.basename(good), "args": {"seed": 0}}))
    assert verify(good, subject_map, str(report))["report_cross_check"] == {
        "embedding_file_matches": True, "seed_in_filename_matches": True}
