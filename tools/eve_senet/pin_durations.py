"""Pin the dead duration channel with a bitwise comparison (validation Group 5).

``SE-Net/src/models.py`` adds ``duration_encoding`` into ``ventral_pos`` and then
calls ``ventral_pos.fill_(0)`` on the next line -- *after* ``ventral_embs +=
ventral_pos`` has already happened -- so the duration never reaches the network
(FR3.4). F3 feeds decile bins anyway, because that is the input contract the released
checkpoint was trained under; what it must not do is *assume* the deadness.

This tool is the check. It takes the real ``bins`` tensor and one or more control
tensors produced by ``embed.py --duration-arm {raw,absurd}`` at the same seed, and
asserts ``torch.equal`` -- bitwise, not ``allclose``. A tolerance would let a small
live contribution through, which is exactly the failure the pin exists to catch.

**A failure here is not a tool bug.** It means the duration channel is live in this
env/checkpoint combination: every embedding generated under the assumption is invalid,
FR3's binning becomes load-bearing rather than merely faithful, and the result has to
be escalated to a roadmap note before F5 consumes anything.

It writes ``duration_pin.json`` next to the reference tensor and stamps
``duration_channel_pinned`` plus the per-arm sha256 into that seed's
``senet_report.json``, so the check survives as an artefact and not only as a green
test run (D5).

CPU-only -- it loads saved tensors and never touches SE-Net, so it runs on a login
node or the Windows dev machine.

CLI: ``py tools/eve_senet/pin_durations.py --reference PATH --arm raw=PATH [...]``
"""

import argparse
import hashlib
import json
import os
import sys

import torch

try:
    from . import EveSenetError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_senet import EveSenetError


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path):
    if not os.path.isfile(path):
        raise EveSenetError("pin_durations: no such tensor: {}".format(path))
    t = torch.load(path, map_location="cpu")
    if not torch.is_tensor(t):
        raise EveSenetError(
            "pin_durations: {} holds {}, not a tensor".format(path, type(t).__name__))
    return t


def pin(reference_path, arm_paths, report_path=None, raise_on_diff=True):
    """``{summary}`` for one seed. Raises :class:`EveSenetError` if any arm differs.

    ``raise_on_diff=False`` returns the summary instead, so a caller can persist the
    evidence of a *failed* pin before failing -- a failure is the more important thing
    to have on disk, not the less.

    ``arm_paths`` is ``{"raw": path, "absurd": path}`` -- at least one arm, and the
    ``absurd`` arm is the one that carries the argument, since ``raw`` alone feeds a
    merely *different plausible* encoding.
    """
    if not arm_paths:
        raise EveSenetError(
            "pin_durations: no control arm given; --arm raw=PATH and/or "
            "--arm absurd=PATH are what the comparison is against (Group 5)")

    ref = _load(reference_path)
    arms, differing = {}, []
    for name in sorted(arm_paths):
        t = _load(arm_paths[name])
        if t.shape != ref.shape or t.dtype != ref.dtype:
            raise EveSenetError(
                "pin_durations: arm {!r} is {} {} but the reference is {} {} -- the "
                "two runs did not use the same cohort or num_fewshot, so the "
                "comparison would be meaningless".format(
                    name, tuple(t.shape), t.dtype, tuple(ref.shape), ref.dtype))
        identical = bool(torch.equal(ref, t))
        if not identical:
            differing.append(name)
        arms[name] = {
            "path": arm_paths[name],
            "sha256": sha256_file(arm_paths[name]),
            "bitwise_identical": identical,
            "max_abs_diff": float((ref - t).abs().max()),
        }

    summary = {
        "reference": {"path": reference_path, "sha256": sha256_file(reference_path)},
        "arms": arms,
        "duration_channel_pinned": not differing,
        "note": (
            "SE-Net/src/models.py fills ventral_pos with zeros after adding it into "
            "ventral_embs, so duration_encoding never reaches the network (FR3.4). "
            "Bitwise equality across duration arms is the evidence."),
    }

    if report_path:
        stamp_report(report_path, summary)

    if differing and raise_on_diff:
        raise EveSenetError(
            "pin_durations: arm(s) {} are NOT bitwise identical to the reference -- "
            "max |diff| {}. The duration channel is LIVE in this env/checkpoint "
            "combination. This is not a tool bug: every embedding generated under the "
            "deadness assumption is invalid, FR3's binning becomes load-bearing, and "
            "this must be escalated to a roadmap note BEFORE F5 consumes anything "
            "(validation Group 5)".format(
                differing, {n: arms[n]["max_abs_diff"] for n in differing}))
    return summary


def stamp_report(report_path, summary):
    """Record the pin inside that seed's ``senet_report.json`` (Group 5, D5)."""
    if not os.path.isfile(report_path):
        raise EveSenetError(
            "pin_durations: no senet_report.json at {}".format(report_path))
    with open(report_path) as fh:
        report = json.load(fh)
    report["duration_channel_pinned"] = summary["duration_channel_pinned"]
    report["duration_pin"] = {
        "reference_sha256": summary["reference"]["sha256"],
        "arms": {n: {"sha256": a["sha256"],
                     "bitwise_identical": a["bitwise_identical"]}
                 for n, a in summary["arms"].items()},
    }
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=1)
        fh.write("\n")


def _arm(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected NAME=PATH (e.g. raw=data/eve_senet/seed0/pin/raw/x.pt)")
    name, path = value.split("=", 1)
    if name not in ("raw", "absurd"):
        raise argparse.ArgumentTypeError("arm name must be 'raw' or 'absurd'")
    return name, path


def main(argv=None):
    p = argparse.ArgumentParser(description="F3 -- duration-channel deadness pin")
    p.add_argument("--reference", required=True,
                   help="the real --duration-arm bins tensor for this seed")
    p.add_argument("--arm", action="append", type=_arm, default=[],
                   metavar="NAME=PATH", help="raw=PATH and/or absurd=PATH")
    p.add_argument("--senet-report", dest="senet_report", default=None,
                   help="that seed's senet_report.json, stamped with the result")
    p.add_argument("--out", default=None,
                   help="where to write duration_pin.json "
                        "(default: alongside --reference)")
    args = p.parse_args(argv)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.reference)), "duration_pin.json")
    try:
        summary = pin(args.reference, dict(args.arm), args.senet_report)
    except EveSenetError as exc:
        sys.stderr.write("FATAL F3 duration pin: {}\n".format(exc))
        return 1
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    print(json.dumps(summary, indent=2))
    sys.stderr.write("pin_durations: OK -- duration channel is dead; wrote {}\n".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
