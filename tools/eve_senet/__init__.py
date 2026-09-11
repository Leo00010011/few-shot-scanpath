"""EVE subject embeddings (Stage C only) -- F3.

Drives the *released* SE-Net checkpoint over each EVE participant's own 10-shot
support set from the F2 bridge and writes a ``(38, 384)`` float32 tensor that
``ISP/EVE/.../src/test.py --user_emb_path`` consumes (F5).

Scope rules for this package:

* Stage C only. No training, no optimiser, no fine-tuning (D8).
* Nothing under ``SE-Net/`` is modified (working convention 2). This package
  *imports* and *subclasses* SE-Net; it never edits it.
* Nothing under ``ISP/*/GazeformerISP/src/utils/`` is read, imported or modified
  (D1). F3 does not touch the ISP tree at all.
* The F2 bridge is consumed as *artefacts*, never as code (D2) -- this package
  shares no module with ``tools/eve_bridge/``.
* ``durations.py``, ``check_env.py`` and ``verify_embedding.py`` run on the Windows
  dev machine; ``dataset.py`` and ``embed.py`` import SE-Net and are cluster-only.
"""


class EveSenetError(RuntimeError):
    """Raised on any F3 invariant violation (FR11)."""
