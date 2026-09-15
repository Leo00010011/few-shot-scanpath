"""EVE eval branch preflight and parity tooling (Stage D + E) -- F5.

This package holds the **CPU-only** half of F5: the artefact handshakes that must pass
before the ISP checkpoint is built, and the re-run of F2's heatmap bitwise-parity check
under whatever numpy the run's env resolved to.

Scope rules for this package:

* Stage D + E only, and only the preconditions. It runs no model and computes no
  metric; ``check_eval`` is a gate, not a score.
* **No torch, no h5py heatmaps, no ``evedataset``** on the import path. It must import
  on the Windows dev machine and on a login node with nothing staged (FR11.1).
* Nothing under ``ISP/*/GazeformerISP/src/utils/`` is read, imported or reached (D1).
* F2, F3 and F4's artefacts are consumed **read-only** and are sha-gated against their
  own reports before anything downstream trusts them (FR3.2 - FR3.4).

The constants below are the F5 contract, single-sourced so the preflight, the dataset
and the tests assert against one derivation rather than three (the mistake F4's
``FEATURE_SHAPE`` note records).
"""


class EveEvalError(RuntimeError):
    """Any F5 precondition or contract violation (D7, FR14)."""


#: The realised cohort (Roadmap OPEN-5, F2's bridge_report.json).
SCORED_CELLS = 1062
SCORED_IMAGES = 354
N_SUBJECTS = 38
SUBJECTS_PER_IMAGE = 3

#: EVE's native stimulus space, (H, W). The OSIE default (600, 800) would mis-scale
#: every coordinate by 2.4x / 1.8x and every metric would still return a number (D3).
ORIGIN_SIZE = (1080, 1920)

#: The model's action map and the metric screen, (rows, cols) / (H, W).
ACTION_MAP = (24, 32)
RESIZE = (384, 512)

#: One (768, 2048) float32 tensor per TRIAL (F4, OPEN-6) and F3's (38, 384) table.
FEATURE_SHAPE = (768, 2048)
EMBEDDING_SHAPE = (38, 384)

MAX_LENGTH = 16

#: The duration-bin guard (TechStack section 3.7): the *other* OSIE label file stores
#: the decile BIN INDEX (0-9) in T. It passes every other invariant and silently
#: invalidates ScanMatch-with-duration and MultiMatch's duration dimension.
MIN_PLAUSIBLE_MAX_T = 20
