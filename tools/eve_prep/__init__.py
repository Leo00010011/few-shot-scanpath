"""EVE per-trial image features (Stage B only) -- F4.

Produces one Mask R-CNN ResNet-50-FPN feature tensor per **trial** -- per
``(stimulus, participant)`` pair -- from the exact 1920x1080 screen capture that
participant actually saw, plus the ``embeddings.npy`` carrying the ``"free-viewing"``
task key. Keying by ``exp_key`` rather than by ``stimulus_name`` is what resolves
Roadmap **OPEN-6**: EVE presents each photograph at a per-trial display scale, so a
single image per name cannot represent what every participant saw.

Scope rules for this package:

* Stage B only. No training, no model changes, no metric changes (D8).
* Nothing under ``ISP/`` is modified (working convention 2). ``ResNetCOCO`` is
  *imported*; the preprocessing chain is transcribed and its bit-identity with
  ``image_data()`` is *proven* by test (FR4.4), never assumed.
* Nothing under ``ISP/*/GazeformerISP/src/utils/`` is read, imported or reached
  (D1). The frozen metric code is not on this path at all.
* F2's artefacts are consumed **read-only**; ``fixations.json``, ``gt_heatmaps.h5``,
  ``subject_id_map.json`` and ``bridge_report.json`` are never rewritten.
* This package reads ``bundle.h5`` directly through ``evedataset`` -- a declared
  deviation from **D2** (FR12), contained here and re-anchored to a bridge artefact
  by the FR2.4 cross-check.
* ``trial_keys.py`` and ``check_features.py`` run on the Windows dev machine;
  ``extract_features.py`` needs a GPU and is the only torch importer that matters.
"""


class EvePrepError(RuntimeError):
    """Raised on any coverage, shape, identity or mapping violation (D7, FR13)."""


#: The Stage B contract (TechStack section 3.2), single-sourced here because both the
#: extractor (which needs torch) and the preflight guard (which must import with no
#: bundle and no ``evedataset``) assert against it. Two derivations of a shape are how
#: a checker ends up agreeing with a writer that is wrong.
FEATURE_SHAPE = (768, 2048)          # 24*32 spatial positions x 2048 channels
RESIZE_INPUT = (384 * 2, 512 * 2)    # (768, 1024) -- upstream's own expression
STIMULUS_SHAPE = (1080, 1920, 3)     # EVE origin_size (H, W) + RGB
