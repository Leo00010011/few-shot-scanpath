"""OSIE baseline preflight tooling (Roadmap F1).

Small CPU-only checks that run *before* a GPU allocation is spent, so a coverage
or shape mismatch fails loudly (D7) instead of producing a plausible, wrong
number. Spec: ``spec/2026-09-08-osie-eval-baseline/``.

Retargeted 2026-09-09 from ``tools/cocofv_prep/`` when F1 reverted to OSIE (the
COCO-FreeView test split is a held-out challenge benchmark -- Roadmap section 4).
Three things changed and nothing else did:

* ``OSIE_evaluation`` keys on ``fixation['name']`` **alone**, not ``"{task}/{name}"``.
  The equal-subject check and the feature-path derivation both lose the task
  component, and the stimulus directory is flat rather than ``<category>/``.
* Coordinate bounds are the native OSIE stimulus frame, **800x600**, not 512x320.
  ``test.py`` does not pass ``origin_size``, so ``OSIE_evaluation`` uses its
  ``(600, 800)`` default and rescales to the 512x384 metric screen (D3).
* ``normalize_fixations.py`` is gone. OSIE's shipped
  ``data/osie_fixations_update_duration.json`` is already canonical -- the TechStack
  section 3.1 schema, splits ``train``/``validation``/``test``, ``condition="freeview"``,
  ``task="none"`` -- so there is no transform to apply.

Scope rules for this package, mirroring ``tools/eve_bridge/``:

* Preflight only. It validates the authors' artefacts; it never rewrites them and
  it never computes a metric.
* This package never imports from ``ISP/*/GazeformerISP/src/utils/`` -- those files
  are frozen (D1).
* ``check_fixations`` is stdlib-only so it runs on the Windows dev machine and on a
  cluster login node; ``check_features`` is the sole torch importer.
"""


class OsiePreflightError(RuntimeError):
    """Raised when a precondition for the OSIE evaluation run is violated."""
