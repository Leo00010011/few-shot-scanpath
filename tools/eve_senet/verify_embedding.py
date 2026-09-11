"""Independently verify an F3 embedding tensor (FR9).

Structural checks only -- shape, dtype, finiteness, no all-zero row, no duplicate
rows, and that the row order really is ``subject_id_map.json``'s dense ids. It shares
no code with :mod:`embed`'s own assertions on purpose: a checker that reuses the
writer's helpers can only confirm the writer was self-consistent.

D4's question "which of my real subjects is row 3?" gets a literal answer here: the
``dense_id -> eve_id -> ||row||`` table is printed for every row (FR9.2).

The pairwise cosine block (FR9.3) is a **sanity signal, not a metric**. No threshold
is enforced -- a cohort whose rows are all near-identical would mean the encoder is
not discriminating our subjects, which F7 needs to know before interpreting any
personalization claim, but it is reported and interpreted rather than gated.

CPU-only and CUDA-free, so it runs on a login node or the Windows dev machine. Its
exit code gates ``bash/embed_eve_subjects.sh`` (FR9.4).

CLI: ``py tools/eve_senet/verify_embedding.py --embedding PATH --subject-map PATH``
"""

import argparse
import json
import os
import sys

import torch

try:
    from . import EveSenetError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_senet import EveSenetError

EMBEDDING_DIM = 384
DUPLICATE_TOL = 1e-9


def verify(embedding_path, subject_map_path, report_path=None, expect_dim=EMBEDDING_DIM):
    """Returns a JSON-able summary. Raises :class:`EveSenetError` on any FR9.1 failure."""
    emb = torch.load(embedding_path, map_location="cpu")
    if not isinstance(emb, torch.Tensor):
        raise EveSenetError("verify: {} holds {}, not a tensor".format(
            embedding_path, type(emb).__name__))
    with open(subject_map_path) as fh:
        subject_map = json.load(fh)

    to_eve = subject_map["to_eve"]
    n = len(to_eve)
    if set(to_eve) != {str(i) for i in range(n)}:  # FR9.1
        raise EveSenetError(
            "verify: subject_id_map to_eve keys are not {{0..{}}} (FR9.1)".format(n - 1))
    for i in range(n):  # D4 roundtrip -- the claim the whole feature rests on
        if subject_map["to_dense"].get(to_eve[str(i)]) != i:
            raise EveSenetError(
                "verify: subject_id_map roundtrip fails at dense id {} ({!r}) "
                "(D4)".format(i, to_eve[str(i)]))

    if tuple(emb.shape) != (n, expect_dim):  # FR9.1
        raise EveSenetError(
            "verify: tensor is {}, expected ({}, {}). A (10, 384) here means the "
            "released OSIE tensor was loaded by mistake (FR9.1)".format(
                tuple(emb.shape), n, expect_dim))
    if emb.dtype != torch.float32:
        raise EveSenetError("verify: dtype is {}, expected float32 (FR9.1)".format(emb.dtype))
    if not bool(torch.isfinite(emb).all()):
        raise EveSenetError("verify: tensor holds non-finite values (FR9.1)")

    zero_rows = [i for i in range(n) if bool((emb[i] == 0).all())]
    if zero_rows:  # FR7.5
        raise EveSenetError(
            "verify: all-zero row(s) {} ({}). The released fewshot_user_embedding_10.pt "
            "has 5 such rows; ours must have none (FR7.5/FR9.1)".format(
                zero_rows, [to_eve[str(i)] for i in zero_rows]))

    dup = [(i, j) for i in range(n) for j in range(i + 1, n)
           if bool(torch.allclose(emb[i], emb[j], atol=DUPLICATE_TOL, rtol=0))]
    if dup:  # two subjects fed the same support set
        raise EveSenetError(
            "verify: identical embedding rows {} -- two subjects were fed the same "
            "support set, which the per-subject seed should make impossible "
            "(FR9.1)".format(dup[:10]))

    norms = torch.linalg.norm(emb, dim=1)
    normed = emb / norms.unsqueeze(1)
    cos = normed @ normed.T
    off = cos[~torch.eye(n, dtype=torch.bool)]

    rows = [{"dense_id": i, "eve_id": to_eve[str(i)], "norm": float(norms[i])}
            for i in range(n)]
    summary = {
        "embedding": os.path.basename(embedding_path),
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "n_subjects": n,
        "zero_rows": [], "duplicate_rows": [],
        "row_norm": {"min": float(norms.min()), "max": float(norms.max()),
                     "ratio": float(norms.max() / norms.min())},
        "offdiag_cosine": {"min": float(off.min()), "mean": float(off.mean()),
                           "max": float(off.max())},          # FR9.3 -- reported only
        "rows": rows,
    }

    if report_path and os.path.isfile(report_path):
        with open(report_path) as fh:
            rep = json.load(fh)
        summary["report_cross_check"] = {
            "embedding_file_matches": rep.get("embedding_file") == os.path.basename(
                embedding_path),
            # the seed in the filename must agree with the seed in the report -- the
            # guard aggregate_seeds.py needed for F1, so F5 cannot pool non-replicates
            "seed_in_filename_matches": "seed{}".format(
                rep.get("args", {}).get("seed")) in os.path.basename(embedding_path),
        }
        bad = [k for k, v in summary["report_cross_check"].items() if not v]
        if bad:
            raise EveSenetError(
                "verify: senet_report.json disagrees with the tensor filename: {} "
                "(D5)".format(bad))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Verify an F3 embedding tensor (FR9)")
    p.add_argument("--embedding", required=True)
    p.add_argument("--subject-map", dest="subject_map", required=True)
    p.add_argument("--report", default=None,
                   help="senet_report.json, to cross-check the seed and filename")
    args = p.parse_args(argv)

    try:
        summary = verify(args.embedding, args.subject_map, args.report)
    except EveSenetError as exc:
        sys.stderr.write("FATAL verification failure: {}\n".format(exc))
        return 1
    print(json.dumps(summary, indent=2))
    sys.stderr.write("{:>8}  {:<12} {}\n".format("dense_id", "eve_id", "||row||"))
    for row in summary["rows"]:  # FR9.2 -- D4's question, answered in an artefact
        sys.stderr.write("{:>8}  {:<12} {:.4f}\n".format(
            row["dense_id"], row["eve_id"], row["norm"]))
    cos = summary["offdiag_cosine"]
    sys.stderr.write(
        "off-diagonal cosine: min {:.4f} mean {:.4f} max {:.4f} "
        "(reported, not gated -- FR9.3)\n".format(cos["min"], cos["mean"], cos["max"]))
    sys.stderr.write("verify_embedding: OK -- {} rows\n".format(summary["n_subjects"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
