"""EVE -> ISP-SENet bridge (Stage A only).

Turns an EVE ``bundle.h5`` (read through the installed ``evedataset`` package) into
the canonical inputs ISP-SENet consumes: ``fixations.json`` (TechStack §3.1),
``subject_id_map.json`` (D4), a flat ``stimuli/`` directory, and the ground-truth
per-timestep 24x32 heatmap cache ``gt_heatmaps.h5``.

Scope rules for this package:

* Stage A only. No feature extraction (F4), no SE-Net (F3), no eval branch (F5).
* CPU / Windows runnable. Everything except :mod:`heatmap_metrics` is
  numpy / scipy / h5py / PIL only; :mod:`heatmap_metrics` is the sole torch importer.
* This package never imports from ``ISP/*/GazeformerISP/src/utils/`` — those files
  are frozen (D1). The only ISP code it touches is ``models/loss.py``, loaded by
  path via ``importlib`` and never modified.
"""
