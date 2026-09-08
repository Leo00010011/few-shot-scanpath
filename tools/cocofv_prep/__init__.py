"""COCO-FreeView baseline preflight tooling (Roadmap F1).

Small CPU-only checks that run *before* a GPU allocation is spent, so a coverage
or shape mismatch fails loudly (D7) instead of producing a plausible, wrong
number. Spec: ``spec/2026-09-08-cocofv-baseline-on-cluster/``.

Scope rules for this package, mirroring ``tools/eve_bridge/``:

* Preflight only. It validates the authors' artefacts; it never rewrites them and
  it never computes a metric.
* This package never imports from ``ISP/*/GazeformerISP/src/utils/`` -- those files
  are frozen (D1).
* ``normalize_fixations`` and ``check_fixations`` are stdlib-only so they run on the
  Windows dev machine and on a cluster login node; ``check_features`` is the sole
  torch importer.
"""


class CocoFvPreflightError(RuntimeError):
    """Raised when a precondition for the COCO-FreeView evaluation run is violated."""
