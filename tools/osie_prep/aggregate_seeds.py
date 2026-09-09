"""Aggregate an OSIE seed sweep into one machine-readable metric block (FR13.3, D6).

Inference is stochastic -- ``Sampling.random_sample()`` draws the action and the
log-normal duration per step -- so a single seed is a point, not a band. The sweep
runs ``--seed 0 1 2`` and this tool turns the three runs into mean +/- spread.

    *** The spread this tool reports is an ACROSS-SEED spread. ***

That is not the same quantity as ``cur_metrics_std``, which
``comprehensive_evaluation_by_subject()`` returns and ``test.py`` discards: that one
is a spread over (image, subject) **cells** within a single run. Ours is a spread
over **runs**, n = 3. They answer different questions -- "how much does this number
move if I re-sample?" versus "how much does it vary across the test set?" -- and a
write-up must never present one as the other. The per-cell std stays deferred to F6,
which recomputes every metric from ``prediction.json`` on CPU and can emit both.
``test.py`` is left unmodified for F1 (Roadmap F1), so this tool parses its log.

What it reads, per ``seed<N>/`` directory produced by ``bash/test_osie.sh``:

* ``log_test_subject_{num_fewshot}_{random_support}.txt`` -- the resolved arg
  namespace and every mean in ``cur_metrics``. ``test.py`` truncates this file at
  the top of ``main()`` (``open(log_file, 'w').close()``), so despite the
  ``FileHandler(mode='a')`` each file holds exactly one run's block.
* ``stdout.txt`` -- the headline ``SM / MM / SED`` line, which is a bare ``print()``
  and therefore never reaches the log file (TechStack section 3.5a note 1). Optional;
  when present it is used as an integrity cross-check, not as the source of truth.

Two guards, because averaging the wrong runs together is the failure mode that
produces a plausible, wrong band (D7):

* the ``seed`` in each log's arg namespace must equal its directory's seed;
* every argument that changes what is being measured must be identical across
  seeds. A sweep that silently mixes two ``--subject_num`` values, or two
  ``--fix_dir`` files, is not a sweep.

stdlib only -- no torch, no numpy, no scipy -- so it runs on the Windows dev machine
and on a cluster login node. It never imports frozen code (D1) and never recomputes a
metric from scanpaths; it only re-derives the two published *composites* (``SM`` is a
harmonic mean of the two ScanMatch variants, ``MM`` an arithmetic mean of the five
MultiMatch dimensions -- TechStack section 4) from numbers the authors' evaluator
already produced.

CLI, from the repo root or from ``ISP/OSIE/GazeformerISP/``::

    py tools/osie_prep/aggregate_seeds.py --log-dir result/OSIE-ex-10to15/log
    py tools/osie_prep/aggregate_seeds.py --log-dir DIR --out metrics_sweep.json \
        --markdown metrics_sweep.md

Prints the aggregate as JSON on **stdout**; the markdown table and all commentary
go to stderr unless ``--markdown`` names a file.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

try:
    from . import OsiePreflightError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from osie_prep import OsiePreflightError

#: ``logger.py``'s format: ``[%(asctime)s - %(name)s - %(levelname)s] %(message)s``.
#: The timestamp itself contains colons, so the prefix must come off before any
#: ``key: value`` parsing.
_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s?(.*)$")

_ARGS_MARKER = "The args corresponding to testing process are:"
_METRICS_MARKER = "The metrics for best model performance are:"

#: ``test.py``: ``"{metrics_key:10}-{metric_name:15}: {metric_value:.4f}"``. Neither a
#: metrics_key (``MultiMatch``, ``ScanMatch``, ``VAME``,
#: ``retrieval scanmatch w/ duration``) nor a metric_name (``w/o duration``,
#: ``SED_best``, ``pmrr``, ...) contains a hyphen, so splitting on the first one is
#: unambiguous. Both fields are *padded*, never truncated, hence the surrounding
#: ``\s*``.
_METRIC_RE = re.compile(
    r"^(?P<key>\S(?:.*?\S)?)\s*-(?P<name>\S(?:.*?\S)?)\s*:\s*"
    r"(?P<val>[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))\s*$"
)

#: ``test.py``'s last line: ``print('SM: {}, MM: {}, SED: {}'.format(...))`` with each
#: value ``round(x, 3)``.
_HEADLINE_RE = re.compile(
    r"^SM:\s*(?P<sm>[-+0-9.eE]+),\s*MM:\s*(?P<mm>[-+0-9.eE]+),\s*"
    r"SED:\s*(?P<sed>[-+0-9.eE]+)\s*$"
)

#: Arguments that change *what is being measured*. If any of these differs between
#: two seed directories the runs are not replicates and must not be pooled. ``seed``
#: is deliberately absent -- it is the thing that varies.
INVARIANT_ARGS = (
    "subject_num", "fewshot_subject", "eval_repeat_num", "width", "height",
    "fix_dir", "feat_dir", "emb_dir", "user_emb_path", "evaluation_dir",
    "max_length", "im_h", "im_w", "num_fewshot", "random_support",
)

#: The D6 headline block, in report order. ``SED_best``/``STDE_best`` are computed by
#: the evaluator but are not named by D6, so they are carried as supplementary.
D6_ORDER = (
    ("MultiMatch", "vector"), ("MultiMatch", "direction"), ("MultiMatch", "length"),
    ("MultiMatch", "position"), ("MultiMatch", "duration"),
    ("ScanMatch", "w/o duration"), ("ScanMatch", "with duration"),
    ("VAME", "SED"), ("VAME", "STDE"),
    ("retrieval scanmatch w/ duration", "pmrr"),
    ("retrieval scanmatch w/ duration", "pr1"),
    ("retrieval scanmatch w/ duration", "pr3"),
    ("retrieval scanmatch w/ duration", "pr5"),
)

#: Recomputing ``SM``/``MM`` from log values that were themselves written at ``.4f``,
#: then comparing against a headline printed at ``round(x, 3)``, cannot be exact.
#: Below WARN the agreement is as good as the formatting allows; above FATAL the two
#: artefacts are not from the same run.
_HEADLINE_WARN = 2e-3
_HEADLINE_FATAL = 1e-2


def _strip_prefix(line):
    match = _PREFIX_RE.match(line.rstrip("\n"))
    return match.group(1) if match else line.rstrip("\n")


def parse_run_log(path):
    """Parse one ``log_test_subject_*.txt`` into ``(args, metrics)``.

    ``metrics`` is ``{metrics_key: {metric_name: float}}``. Both sections are taken
    from their **last** occurrence in the file: ``test.py`` truncates on start so
    there is normally exactly one of each, but a file that was appended to by an
    older ``test.py``, or hand-concatenated, must not yield a silent mixture.
    """
    if not os.path.isfile(path):
        raise OsiePreflightError("missing run log: {}".format(path))

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [_strip_prefix(_) for _ in handle]

    args_at = [i for i, _ in enumerate(lines) if _.startswith(_ARGS_MARKER)]
    metrics_at = [i for i, _ in enumerate(lines) if _.startswith(_METRICS_MARKER)]
    if not args_at:
        raise OsiePreflightError(
            "{}: no arg-namespace block ('{}'). Either the run died before "
            "logging, or this is not a test.py log.".format(path, _ARGS_MARKER))
    if not metrics_at:
        raise OsiePreflightError(
            "{}: no metric block ('{}'). The run did not reach "
            "comprehensive_evaluation_by_subject() -- do not pool it.".format(
                path, _METRICS_MARKER))

    run_args = {}
    for line in lines[args_at[-1] + 1:metrics_at[-1]]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        # The arg block is `{key:20}: {value:}`; a stray blank or a tqdm remnant is
        # not a key. Metric lines are excluded structurally by the slice above.
        if key and " " not in key:
            run_args[key] = value

    metrics = {}
    for line in lines[metrics_at[-1] + 1:]:
        match = _METRIC_RE.match(line)
        if not match:
            continue
        metrics.setdefault(match.group("key"), {})[match.group("name")] = float(
            match.group("val"))

    if not metrics:
        raise OsiePreflightError(
            "{}: metric block is present but empty -- parsed 0 values".format(path))
    return run_args, metrics


def parse_headline(path):
    """Return ``{'SM','MM','SED'}`` from a captured stdout, or ``None`` if absent.

    The headline is a bare ``print()`` and goes to stdout, never to the run log
    (TechStack section 3.5a). ``bash/test_osie.sh`` tees it into ``seed<N>/stdout.txt``
    precisely so it survives the next seed.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        found = None
        for line in handle:
            match = _HEADLINE_RE.match(line.strip())
            if match:
                found = {"SM": float(match.group("sm")),
                         "MM": float(match.group("mm")),
                         "SED": float(match.group("sed"))}
    return found


def composites(metrics):
    """Re-derive ``SM``/``MM``/``SED`` exactly as ``test.py``'s last lines do.

    ``SM = hmean(ScanMatch.values())``, ``MM = mean(MultiMatch.values())``,
    ``SED = VAME['SED']`` (TechStack section 4). ``statistics.harmonic_mean`` is
    order-independent and agrees with ``scipy.stats.hmean`` on positive finite input,
    which is all this ever sees.
    """
    try:
        scanmatch = list(metrics["ScanMatch"].values())
        multimatch = list(metrics["MultiMatch"].values())
        sed = metrics["VAME"]["SED"]
    except KeyError as exc:
        raise OsiePreflightError(
            "metric block is missing {} -- cannot form the headline "
            "composites".format(exc))
    return {
        "SM": statistics.harmonic_mean(scanmatch),
        "MM": statistics.fmean(multimatch),
        "SED": sed,
    }


def _seed_of(dirname):
    match = re.match(r"^seed(\d+)$", os.path.basename(dirname.rstrip(os.sep)))
    return int(match.group(1)) if match else None


def collect_seed_dirs(log_dir):
    found = []
    for path in sorted(glob.glob(os.path.join(log_dir, "seed*"))):
        seed = _seed_of(path)
        if seed is not None and os.path.isdir(path):
            found.append((seed, path))
    if not found:
        raise OsiePreflightError(
            "no seed<N>/ directories under {}. bash/test_osie.sh writes them; run "
            "the sweep before aggregating it.".format(log_dir))
    return sorted(found)


def read_seed(seed, seed_dir):
    logs = sorted(glob.glob(os.path.join(seed_dir, "log_test_subject_*.txt")))
    if not logs:
        raise OsiePreflightError(
            "{}: no log_test_subject_*.txt".format(seed_dir))
    if len(logs) > 1:
        raise OsiePreflightError(
            "{}: {} log files, expected 1 -- --num_fewshot/--random_support must "
            "not vary within a seed sweep, they name the file".format(
                seed_dir, len(logs)))

    run_args, metrics = parse_run_log(logs[0])

    logged_seed = run_args.get("seed")
    if logged_seed is None:
        raise OsiePreflightError(
            "{}: arg namespace has no 'seed' -- cannot verify provenance "
            "(D5)".format(logs[0]))
    if int(logged_seed) != seed:
        raise OsiePreflightError(
            "{}: directory says seed {} but the run logged seed {}. The copy step "
            "in bash/test_osie.sh picked up another seed's artefacts; pooling "
            "these would report a spread over the wrong runs.".format(
                seed_dir, seed, logged_seed))

    derived = composites(metrics)
    headline = parse_headline(os.path.join(seed_dir, "stdout.txt"))
    headline_delta = None
    if headline is not None:
        headline_delta = {k: abs(headline[k] - derived[k]) for k in derived}
        worst = max(headline_delta.values())
        if worst > _HEADLINE_FATAL:
            raise OsiePreflightError(
                "{}: stdout headline {} disagrees with the values re-derived from "
                "the run log {} (max delta {:.4g}). The two files are not from the "
                "same run.".format(seed_dir, headline,
                                   {k: round(v, 4) for k, v in derived.items()},
                                   worst))
        if worst > _HEADLINE_WARN:
            sys.stderr.write(
                "WARNING: seed {}: headline vs re-derived composites differ by "
                "{:.4g}, more than formatting alone explains ({} vs {})\n".format(
                    seed, worst, headline,
                    {k: round(v, 4) for k, v in derived.items()}))

    return {
        "seed": seed,
        "dir": seed_dir,
        "log_file": logs[0],
        "args": run_args,
        "metrics": metrics,
        "composites": derived,
        "headline_printed": headline,
        "headline_delta": headline_delta,
    }


def check_invariant_args(runs):
    """Every argument that changes what is measured must match across seeds."""
    reference = runs[0]
    problems = []
    for name in INVARIANT_ARGS:
        values = {}
        for run in runs:
            if name in run["args"]:
                values.setdefault(run["args"][name], []).append(run["seed"])
        if len(values) > 1:
            problems.append("{}: {}".format(name, dict(values)))
    if problems:
        raise OsiePreflightError(
            "seed runs are not replicates -- these arguments differ between "
            "them, so pooling them would average over a configuration change "
            "rather than over sampling noise:\n  " + "\n  ".join(problems))
    return {name: reference["args"][name]
            for name in INVARIANT_ARGS if name in reference["args"]}


def _spread(values):
    """Mean and spread of one metric across seeds.

    ``stdev`` is the **sample** standard deviation (ddof=1): the seeds are a sample
    of the sampler's behaviour, not the population of interest. At n = 3 it is a very
    noisy estimate, which is why ``min``/``max``/``n`` ride alongside it -- report
    the range too, never the std alone.
    """
    entry = {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "values": list(values),
    }
    entry["std"] = statistics.stdev(values) if len(values) > 1 else None
    return entry


def aggregate(log_dir):
    runs = [read_seed(seed, path) for seed, path in collect_seed_dirs(log_dir)]
    shared_args = check_invariant_args(runs)

    keys = set()
    for run in runs:
        for metrics_key, block in run["metrics"].items():
            for metric_name in block:
                keys.add((metrics_key, metric_name))

    metrics = {}
    incomplete = []
    for metrics_key, metric_name in sorted(keys):
        values = [run["metrics"][metrics_key][metric_name] for run in runs
                  if metric_name in run["metrics"].get(metrics_key, {})]
        if len(values) != len(runs):
            incomplete.append("{}-{}".format(metrics_key, metric_name))
        metrics.setdefault(metrics_key, {})[metric_name] = _spread(values)

    composite = {name: _spread([run["composites"][name] for run in runs])
                 for name in ("SM", "MM", "SED")}

    subject_num = int(shared_args.get("subject_num", 0) or 0)
    notes = [
        "std/min/max are ACROSS-SEED, over n={} runs -- a re-sampling band, NOT "
        "the per-(image, subject)-cell cur_metrics_std that test.py discards. Do "
        "not present one as the other (D6; deferred to F6).".format(len(runs)),
        "SM = harmonic mean of the two ScanMatch variants; MM = arithmetic mean of "
        "the five MultiMatch dimensions; both re-derived here from the logged "
        "means, matching test.py's final lines (TechStack section 4).",
        "OSIE's p2g() computes r3 as rank < 3, so pr3 really is R@3 here. The "
        "rank < 2 defect documented in TechStack section 4.2 is COCO_FV's, not "
        "this branch's.",
    ]
    if subject_num and subject_num <= 5:
        notes.append(
            "pr5 (R@5) is STRUCTURALLY SATURATED at 100.0: with subject_num={} "
            "every rank lies in {{0..{}}}, so rank < 5 is always true. It is "
            "reported because D6 names it, and must be annotated wherever it "
            "appears (Roadmap F1).".format(subject_num, subject_num - 1))
    if incomplete:
        notes.append("INCOMPLETE across seeds, pooled over fewer runs: {}".format(
            ", ".join(incomplete)))

    return {
        "log_dir": log_dir,
        "n_seeds": len(runs),
        "seeds": [run["seed"] for run in runs],
        "shared_args": shared_args,
        "composites": composite,
        "metrics": metrics,
        "d6_order": [list(_) for _ in D6_ORDER],
        "per_seed": [
            {"seed": run["seed"], "log_file": run["log_file"],
             "composites": run["composites"],
             "headline_printed": run["headline_printed"]}
            for run in runs
        ],
        "notes": notes,
    }


def _fmt(entry, places=4):
    if entry["std"] is None:
        return "{:.{p}f}".format(entry["mean"], p=places)
    return "{:.{p}f} +/- {:.{p}f}".format(entry["mean"], entry["std"], p=places)


def to_markdown(result):
    seeds = ", ".join(str(_) for _ in result["seeds"])
    out = [
        "### OSIE baseline -- {}-seed sweep (seeds {})".format(
            result["n_seeds"], seeds),
        "",
        "| metric | mean +/- std (across seeds) | min | max |",
        "|---|---|---|---|",
    ]
    for metrics_key, metric_name in D6_ORDER:
        entry = result["metrics"].get(metrics_key, {}).get(metric_name)
        if entry is None:
            continue
        out.append("| {} / {} | {} | {:.4f} | {:.4f} |".format(
            metrics_key, metric_name, _fmt(entry), entry["min"], entry["max"]))
    out.append("")
    out.append("| headline | mean +/- std (across seeds) | min | max |")
    out.append("|---|---|---|---|")
    for name in ("SM", "MM", "SED"):
        entry = result["composites"][name]
        out.append("| {} | {} | {:.4f} | {:.4f} |".format(
            name, _fmt(entry, 3), entry["min"], entry["max"]))
    out.append("")
    for note in result["notes"]:
        out.append("- {}".format(note))
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate an OSIE seed sweep into one metric block (FR13.3)")
    parser.add_argument("--log-dir", dest="log_dir", required=True,
                        help="result/<evaluation_dir basename>/log, the directory "
                             "holding the seed<N>/ subdirectories")
    parser.add_argument("--out", dest="out_path", default=None,
                        help="write the JSON here as well as to stdout")
    parser.add_argument("--markdown", dest="markdown_path", default=None,
                        help="write the markdown table here instead of to stderr")
    args = parser.parse_args(argv)

    try:
        result = aggregate(args.log_dir)
    except OsiePreflightError as exc:
        sys.stderr.write("FATAL aggregation failure: {}\n".format(exc))
        return 1

    payload = json.dumps(result, indent=2, sort_keys=False)
    print(payload)
    if args.out_path:
        with open(args.out_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    table = to_markdown(result)
    if args.markdown_path:
        with open(args.markdown_path, "w", encoding="utf-8") as handle:
            handle.write(table)
        sys.stderr.write("aggregate_seeds: wrote {}\n".format(args.markdown_path))
    else:
        sys.stderr.write("\n" + table)

    sys.stderr.write("aggregate_seeds: OK -- {} seeds {}\n".format(
        result["n_seeds"], result["seeds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
