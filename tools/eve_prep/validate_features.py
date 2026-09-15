"""Post-run Data Validity + Data Architecture Integrity checks (F4 validation.md).

CPU-only and GPU-free, so it runs on a login node or on Windows. Every check the
spec lists as a "notebook cell" is here instead, because a notebook session leaves
no artefact and D5 wants one: this writes ``validation_report.json`` beside the
features and exits non-zero if anything failed.

Each check is wrapped, so the script runs to completion on a *broken* cache -- the
only cache where it is interesting. Checks needing the EVE bundle (which lives on
node-local scratch and is gone once the allocation ends) **skip** cleanly unless
``--bundle-dir`` is given; a skip is reported as a skip and never as a pass.

CLI: ``python tools/eve_prep/validate_features.py --features data/eve_features
--bridge-dir data/eve_bridge [--bundle-dir DIR] [--senet-report PATH] [--no-sha]``
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

try:
    from . import EvePrepError, FEATURE_SHAPE
    from .check_features import select_trials
    from .trial_keys import (exp_key_filename, load_json, load_trial_exp_keys,
                             sha256_file, split_of)
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_prep import EvePrepError, FEATURE_SHAPE
    from eve_prep.check_features import select_trials
    from eve_prep.trial_keys import (exp_key_filename, load_json,
                                     load_trial_exp_keys, sha256_file, split_of)

#: A (768, 2048) float32 tensor is 6,291,456 bytes before pickle overhead. Materially
#: smaller means a truncated write.
MIN_FEATURE_BYTES = 6_291_456

#: Sample size for the per-tensor statistics. Loading all 1062 would be 6.7 GB.
N_SAMPLED = 30

#: Multi-viewer stimuli compared for the OPEN-6 falsifiable prediction.
N_COSINE_STIMULI = 10


class Checks(object):
    """Collects results so one failure does not hide the rest."""

    def __init__(self):
        self.results = []

    def run(self, name, fn):
        try:
            detail = fn()
        except _Skip as skip:
            self.results.append({"check": name, "status": "skip",
                                 "detail": str(skip)})
            return None
        except Exception as exc:
            self.results.append({"check": name, "status": "FAIL",
                                 "detail": "{}: {}".format(type(exc).__name__, exc)})
            return None
        self.results.append({"check": name, "status": "ok", "detail": detail})
        return detail

    @property
    def failed(self):
        return [r for r in self.results if r["status"] == "FAIL"]

    @property
    def skipped(self):
        return [r for r in self.results if r["status"] == "skip"]


class _Skip(Exception):
    """Raised by a check whose inputs are absent -- reported as skip, never as pass."""


def _load(path):
    return torch.load(path, map_location="cpu")


def _cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def _sample(seq, n):
    seq = list(seq)
    if len(seq) <= n:
        return seq
    return [seq[i] for i in np.linspace(0, len(seq) - 1, n).astype(int)]


def validate(features_dir, bridge_dir, bundle_dir=None, senet_report=None,
             verify_sha=True):
    """Run every check. Returns ``(report_dict, ok)`` and never raises."""
    checks = Checks()
    feat_dir = os.path.join(features_dir, "image_features")
    report_path = os.path.join(features_dir, "feature_report.json")
    fixations_path = os.path.join(bridge_dir, "fixations.json")
    heatmaps_path = os.path.join(bridge_dir, "gt_heatmaps.h5")

    # The prelude is itself a check. Loading the mapping applies FR2.5's hash gate, so
    # an edited or regenerated fixations.json fails *here* -- and it must be reported
    # as a failed check rather than escaping as an exception, or the one artefact this
    # script exists to produce never gets written.
    try:
        report = load_json(report_path)
        split = report["args"]["split"]
        bridge_report = load_json(os.path.join(bridge_dir, "bridge_report.json"))
        subject_id_map = load_json(os.path.join(bridge_dir, "subject_id_map.json"))
        fixations = load_json(fixations_path)
        exp_of = load_trial_exp_keys(heatmaps_path, fixations_path)
        selected, splits = select_trials(fixations, split)
        wanted_keys = sorted(exp_of[k] for k in selected)
        stems = sorted(f[:-4] for f in os.listdir(feat_dir) if f.endswith(".pth"))
    except Exception as exc:
        checks.results.append({
            "check": "f2_artefacts_untouched",
            "status": "FAIL",
            "detail": "cannot even load the inputs -- {}: {}".format(
                type(exc).__name__, exc)})
        return ({"features_dir": os.path.abspath(features_dir), "split": None,
                 "n_tensors": None, "cosine": {}, "checks": checks.results,
                 "n_ok": 0, "n_failed": 1, "n_skipped": 0}, False)

    # ------------------------------------------------------------------ Data Validity

    def coverage():
        expected = {"both": bridge_report["num_trials"],
                    "test": bridge_report["num_trials_test"],
                    "train": bridge_report["num_trials_train"]}[split]
        if len(stems) != expected:
            raise EvePrepError(
                "{} .pth files for split={!r}, expected exactly {} -- any other "
                "number means trials were skipped (D7)".format(
                    len(stems), split, expected))
        return "{} tensors, exactly as bridge_report.json says".format(len(stems))

    checks.run("coverage_is_exact", coverage)

    def file_sizes():
        sizes = [os.path.getsize(os.path.join(feat_dir, s + ".pth")) for s in stems]
        if min(sizes) < MIN_FEATURE_BYTES:
            small = [s for s, z in zip(stems, sizes) if z < MIN_FEATURE_BYTES]
            raise EvePrepError(
                "{} files below {} bytes -- a truncated write: {}".format(
                    len(small), MIN_FEATURE_BYTES, small[:10]))
        return "min {} B, max {} B".format(min(sizes), max(sizes))

    checks.run("every_file_is_non_trivial", file_sizes)

    sampled = _sample(stems, N_SAMPLED)
    tensors = {}

    def load_sampled():
        for stem in sampled:
            tensors[stem] = _load(os.path.join(feat_dir, stem + ".pth"))
        bad = [s for s, t in tensors.items()
               if tuple(t.shape) != FEATURE_SHAPE or t.dtype != torch.float32]
        if bad:
            raise EvePrepError("wrong shape or dtype: {}".format(bad[:10]))
        return "{} tensors loaded, all {} float32".format(len(tensors), FEATURE_SHAPE)

    checks.run("sampled_tensors_load", load_sampled)

    def no_dead_tensors():
        if not tensors:
            raise _Skip("no tensors loaded")
        dead = [s for s, t in tensors.items()
                if float(t.abs().sum()) == 0.0 or float(t.std()) == 0.0
                or not bool(torch.isfinite(t).all())]
        if dead:
            raise EvePrepError(
                "zero, constant or non-finite tensors -- the backbone ran on a blank "
                "or corrupt image: {}".format(dead[:10]))
        stds = [float(t.std()) for t in tensors.values()]
        return "std min {:.4f} max {:.4f}, all finite".format(min(stds), max(stds))

    checks.run("no_dead_tensors", no_dead_tensors)

    def post_relu():
        if not tensors:
            raise _Skip("no tensors loaded")
        negative = {s: float(t.min()) for s, t in tensors.items()
                    if float(t.min()) < 0.0}
        if negative:
            raise EvePrepError(
                "negative values after layer4's ReLU -- the wrong module was "
                "tapped: {}".format(dict(list(negative.items())[:5])))
        return "min == 0.0 on all {} sampled".format(len(tensors))

    checks.run("post_relu_non_negative", post_relu)

    def sparsity():
        if not tensors:
            raise _Skip("no tensors loaded")
        fracs = sorted(float((t == 0).float().mean()) for t in tensors.values())
        median = float(np.median(fracs))
        note = ""
        # validation.md predicted roughly 0.3-0.8 and flagged >0.95 / <0.05 as worth
        # investigating. F4's CPU sample measured 0.752-0.858, so a median above the
        # stated band is EXPECTED here; only the investigate thresholds fail.
        if median > 0.95 or median < 0.05:
            raise EvePrepError(
                "median zero-fraction {:.3f} is outside the investigate "
                "thresholds (0.05, 0.95)".format(median))
        if median > 0.8:
            note = (" -- above validation's stated 0.3-0.8 band, consistent with the "
                    "0.752-0.858 measured pre-run on the cream-page stimuli")
        return "min {:.3f} median {:.3f} max {:.3f}{}".format(
            fracs[0], median, fracs[-1], note)

    checks.run("sparsity_in_band", sparsity)

    by_name = {}
    for (name, subject) in selected:
        by_name.setdefault(name, []).append(subject)
    multi = sorted(n for n, subs in by_name.items() if len(subs) >= 2)

    cosine_stats = {}

    def per_trial_keying():
        """The OPEN-6 falsifiable prediction, on the REAL extracted tensors."""
        if not multi:
            raise _Skip("no stimulus in this split has two participants")
        chosen = _sample(multi, N_COSINE_STIMULI)
        cache = {}

        def feat(key):
            if key not in cache:
                cache[key] = _load(os.path.join(feat_dir, key + ".pth"))
            return cache[key]

        within, identical = [], []
        for name in chosen:
            first, second = sorted(by_name[name])[:2]
            a = feat(exp_of[(name, first)])
            b = feat(exp_of[(name, second)])
            if torch.equal(a, b):
                identical.append(name)
            within.append(_cosine(a, b))
        if identical:
            raise EvePrepError(
                "bit-identical tensors for two participants -- per-trial keying "
                "collapsed back to per-name and OPEN-6 is NOT resolved: {}".format(
                    identical[:10]))

        cross = [_cosine(feat(exp_of[(chosen[i], sorted(by_name[chosen[i]])[0])]),
                         feat(exp_of[(chosen[j], sorted(by_name[chosen[j]])[0])]))
                 for i in range(len(chosen)) for j in range(i + 1, len(chosen))]

        cosine_stats.update({
            "within_name": {"min": min(within), "median": float(np.median(within)),
                            "max": max(within), "n": len(within)},
            "cross_name": {"min": min(cross), "median": float(np.median(cross)),
                           "max": max(cross), "n": len(cross)},
        })
        if float(np.median(within)) <= float(np.median(cross)):
            raise EvePrepError(
                "within-name cosine median {:.4f} does not exceed cross-name "
                "{:.4f} -- the features are not discriminating images".format(
                    float(np.median(within)), float(np.median(cross))))
        return ("within min {:.4f} median {:.4f} max {:.4f} | cross median {:.4f} "
                "| none identical".format(min(within), float(np.median(within)),
                                          max(within), float(np.median(cross))))

    checks.run("per_trial_keying_did_something", per_trial_keying)

    def squash_recorded():
        sq = report["squash"]
        if (abs(sq["x"] - 1024 / 1920.0) > 1e-12
                or abs(sq["y"] - 768 / 1080.0) > 1e-12 or sq["uniform"]):
            raise EvePrepError("squash is {}".format(sq))
        if report["resize_input"] != [768, 1024]:
            raise EvePrepError("resize_input is {}".format(report["resize_input"]))
        return "x {:.6f} y {:.6f}, non-uniform (FR4.6)".format(sq["x"], sq["y"])

    checks.run("squash_is_recorded", squash_recorded)

    def embeddings_identical():
        info = report["embeddings"]
        dst = os.path.join(features_dir, "embeddings.npy")
        actual = sha256_file(dst)
        if actual != info["sha256"]:
            raise EvePrepError(
                "embeddings.npy hashes to {} but the report says {}".format(
                    actual, info["sha256"]))
        if os.path.isfile(info["src"]):
            src_sha = sha256_file(info["src"])
            if src_sha != actual:
                raise EvePrepError(
                    "embeddings.npy differs from the OSIE source {}".format(src_sha))
            return "byte-identical to the OSIE source, sha {}".format(actual[:16])
        return ("matches the report ({}); OSIE source not present to "
                "re-compare".format(actual[:16]))

    checks.run("embeddings_match_osie", embeddings_identical)

    def counts_agree():
        got = (report["n_trials"], report["n_trials_test"], report["n_trials_train"])
        expect = {"both": (bridge_report["num_trials"],
                           bridge_report["num_trials_test"],
                           bridge_report["num_trials_train"]),
                  "test": (bridge_report["num_trials_test"],
                           bridge_report["num_trials_test"], 0),
                  "train": (bridge_report["num_trials_train"], 0,
                            bridge_report["num_trials_train"])}[split]
        if got != expect:
            raise EvePrepError("{} vs bridge_report's {}".format(got, expect))
        return "n_trials {} / test {} / train {}".format(*got)

    checks.run("counts_agree_with_bridge_report", counts_agree)

    # ------------------------------------------- Data Architecture Integrity

    def sha_matches_disk():
        if not verify_sha:
            raise _Skip("--no-sha")
        recorded = report["feature_sha256"]
        missing = [k for k in wanted_keys if k not in recorded]
        if missing:
            raise EvePrepError(
                "{} extracted keys absent from feature_sha256: {}".format(
                    len(missing), missing[:10]))
        bad = []
        for key in wanted_keys:
            if sha256_file(os.path.join(feat_dir, key + ".pth")) != recorded[key]:
                bad.append(key)
        if bad:
            raise EvePrepError(
                "{} tensors differ from the sha recorded in feature_report.json -- "
                "the cache is not what was checked (FR8.2): {}".format(
                    len(bad), bad[:10]))
        return "all {} tensors match feature_report.json's feature_sha256".format(
            len(wanted_keys))

    checks.run("feature_sha256_matches_disk", sha_matches_disk)

    def no_phantom_keys():
        want, have = set(wanted_keys), set(stems)
        extra, absent = sorted(have - want), sorted(want - have)
        if extra or absent:
            raise EvePrepError(
                "stems with no trial (stale files from an earlier run): {}; trials "
                "with no stem (coverage hole): {}".format(extra[:10], absent[:10]))
        return "{} stems == {} trials, both directions".format(len(have), len(want))

    checks.run("no_phantom_keys", no_phantom_keys)

    def round_trip_via_store():
        """Uses the store's REVERSE index, so the forward path cannot self-validate."""
        import h5py

        with h5py.File(heatmaps_path, "r") as f:
            grp = f["trials"]
            trial_keys = [v.decode() if isinstance(v, bytes) else str(v)
                          for v in grp["trial_key"][:]]
            store_exp = [v.decode() if isinstance(v, bytes) else str(v)
                         for v in grp["exp_key"][:]]
        reverse = dict(zip(store_exp, trial_keys))
        bad = []
        for (name, subject) in selected:
            key = exp_of[(name, subject)]
            if reverse.get(key) != "{}|{}".format(name, subject):
                bad.append((name, subject, key, reverse.get(key)))
        if bad:
            raise EvePrepError("round-trip failed: {}".format(bad[:5]))
        return "all {} trials round-trip through the store's reverse index".format(
            len(selected))

    checks.run("exp_key_round_trip", round_trip_via_store)

    def dense_ids():
        dense = sorted({s for _, s in exp_of})
        expected = list(range(bridge_report["num_subjects"]))
        if dense != expected:
            raise EvePrepError("dense ids are {}, expected {}".format(
                dense[:5], expected[:5]))
        return "exactly 0..{}".format(len(expected) - 1)

    checks.run("dense_ids_are_contiguous", dense_ids)

    def filename_rule():
        for key in wanted_keys:
            if exp_key_filename(key) != key + ".pth":
                raise EvePrepError("bad key {!r}".format(key))
        return "all {} exp_keys filename-safe, none contains 'jpg'".format(
            len(wanted_keys))

    checks.run("exp_keys_are_filename_safe", filename_rule)

    def f2_untouched():
        actual = sha256_file(fixations_path)
        if actual != bridge_report["fixations_sha256"]:
            raise EvePrepError(
                "fixations.json now hashes to {}, bridge_report says {}".format(
                    actual, bridge_report["fixations_sha256"]))
        if report["fixations_sha256"] != actual:
            raise EvePrepError(
                "feature_report.json was built against {}".format(
                    report["fixations_sha256"]))
        return "fixations.json unchanged since F2, and F4 used it"

    checks.run("f2_artefacts_untouched", f2_untouched)

    def f3_untouched():
        if not senet_report:
            raise _Skip("--senet-report not given")
        senet = load_json(senet_report)
        emb = os.path.join(os.path.dirname(senet_report),
                           os.path.basename(senet.get("embedding_path", "")))
        if not os.path.isfile(emb):
            raise _Skip("embedding tensor not beside {}".format(senet_report))
        actual = sha256_file(emb)
        if actual != senet["embedding_sha256"]:
            raise EvePrepError("F3's embedding now hashes to {}".format(actual))
        return "F3's embedding unchanged, sha {}".format(actual[:16])

    checks.run("f3_embedding_untouched", f3_untouched)

    # --------------------------------------------------- bundle-dependent checks

    def fixations_on_the_stimulus():
        """5 trials: the fixations must land on the photo, not the cream page."""
        if not bundle_dir:
            raise _Skip("--bundle-dir not given (scratch is gone after the job)")
        from evedataset import EveBundle

        bundle = EveBundle.load(bundle_dir)
        cream = np.array([245, 240, 210])
        by_trial = {(r["name"], int(r["subject"])): r for r in fixations}
        out = []
        for (name, subject) in _sample(selected, 5):
            img = np.asarray(bundle.get_stimulus(exp_of[(name, subject)]))
            rec = by_trial[(name, subject)]
            on_photo = 0
            for x, y in zip(rec["X"], rec["Y"]):
                px = img[min(int(y) - 1, img.shape[0] - 1),
                         min(int(x) - 1, img.shape[1] - 1)]
                if np.abs(px.astype(int) - cream).max() > 12:
                    on_photo += 1
            out.append("{}/{}".format(on_photo, len(rec["X"])))
            if on_photo == 0:
                raise EvePrepError(
                    "every fixation of {}|{} is on the cream background -- the trial "
                    "is mapped to the wrong capture".format(name, subject))
        return "fixations on the photo: " + ", ".join(out)

    checks.run("fixations_land_on_the_stimulus", fixations_on_the_stimulus)

    def subject_identity():
        if not bundle_dir:
            raise _Skip("--bundle-dir not given")
        from evedataset import EveBundle

        bundle = EveBundle.load(bundle_dir)
        by_key = {str(r.exp_key): str(r.subject)
                  for r in bundle.samples_df.itertuples(index=False)}
        to_eve = subject_id_map["to_eve"]
        bad = []
        for (name, subject) in _sample(selected, 10):
            key = exp_of[(name, subject)]
            if by_key.get(key) != to_eve[str(subject)]:
                bad.append((name, subject, key, by_key.get(key)))
        if bad:
            raise EvePrepError("dense id -> EVE participant broken: {}".format(bad))
        return "10 trials: dense id -> EVE participant -> capture, all consistent"

    checks.run("subject_identity_survives", subject_identity)

    summary = {
        "features_dir": os.path.abspath(features_dir),
        "split": split,
        "n_tensors": len(stems),
        "cosine": cosine_stats,
        "checks": checks.results,
        "n_ok": len([r for r in checks.results if r["status"] == "ok"]),
        "n_failed": len(checks.failed),
        "n_skipped": len(checks.skipped),
    }
    return summary, not checks.failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-run Data Validity checks for F4's feature cache")
    parser.add_argument("--features", default=os.path.join("data", "eve_features"))
    parser.add_argument("--bridge-dir", default=os.path.join("data", "eve_bridge"))
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--senet-report", default=None)
    parser.add_argument("--no-sha", action="store_true",
                        help="skip re-hashing every tensor against feature_report.json")
    parser.add_argument("--out", default=None,
                        help="default: <features>/validation_report.json")
    args = parser.parse_args(argv)

    summary, ok = validate(args.features, args.bridge_dir, args.bundle_dir,
                           args.senet_report, verify_sha=not args.no_sha)

    out = args.out or os.path.join(args.features, "validation_report.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    for r in summary["checks"]:
        sys.stderr.write("{:6s} {:38s} {}\n".format(
            r["status"], r["check"], r["detail"]))
    sys.stderr.write("\n{} ok, {} FAILED, {} skipped -> {}\n".format(
        summary["n_ok"], summary["n_failed"], summary["n_skipped"], out))

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
