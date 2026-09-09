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

**This tool is the run record.** There is deliberately no hand-written ``notes.md`` holding
copy-pasted numbers: a transcribed table starts drifting from the artefacts the moment anything is
re-run, and D5's whole claim is that every reported number be re-derivable from artefacts alone. So
the report is *generated* from the stored outputs, every time, by ``--report``.

Exactly two things cannot be derived and must be supplied by a human, and the report is explicit
about both rather than silently omitting them:

* **The paper's published row.** Nothing in this repository contains it — ``result-images/
  main-result.png`` is qualitative scanpath figures and the READMEs carry no numbers. It is
  transcribed once from the PDF into a ``--reference`` JSON (with its provenance recorded in the file)
  and then it, too, is data.
* **The verdict.** What the comparison licenses, given that the checkpoint was trained on OSIE
  subjects, is an argument rather than a measurement. It belongs in the spec and in F7, not in a
  generated file.

What it reads, per ``seed<N>/`` directory produced by ``bash/test_osie.sh``:

* ``log_test_subject_{num_fewshot}_{random_support}.txt`` -- the resolved arg
  namespace and every mean in ``cur_metrics``. ``test.py`` truncates this file at
  the top of ``main()`` (``open(log_file, 'w').close()``), so despite the
  ``FileHandler(mode='a')`` each file holds exactly one run's block.
* ``stdout.txt`` -- the headline ``SM / MM / SED`` line, which is a bare ``print()``
  and therefore never reaches the log file (TechStack section 3.5a note 1). Optional;
  when present it is used as an integrity cross-check, not as the source of truth.
* ``versions.txt`` -- the resolved stack (D5). TechStack section 1.1 is blunt that "consistent with
  the published numbers" is coarser than bitwise equality and that a future discrepancy can only be
  attributed if each run's stack was written down. Pooled runs must share one stack, so a difference
  raises for the same reason a differing ``--subject_num`` does.
* ``preflight_fixations.json`` -- ``check_fixations.py``'s counters: which label file, which split,
  how many images x subjects were actually scored, and the soft counters D7 requires be surfaced
  (out-of-range coordinates, short scanpaths). This is what makes the report state its own
  denominators instead of asserting them.

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

CLI, **from the repo root**. Note that ``result/<eval>/log`` is anchored to
``ISP/<DATASET>/GazeformerISP/`` by ``test.py``, *not* to the repo root, so it needs
the branch prefix here while ``spec/...`` does not. Keeping every path in one frame
is what avoids a half-and-half invocation::

    # the full run record -- this is the F1 deliverable
    py tools/osie_prep/aggregate_seeds.py \
        --log-dir   ISP/OSIE/GazeformerISP/result/OSIE-ex-10to15/log \
        --reference spec/2026-09-08-osie-eval-baseline/paper_reference.json \
        --out       ISP/OSIE/GazeformerISP/result/OSIE-ex-10to15/log/metrics_sweep.json \
        --report    spec/2026-09-08-osie-eval-baseline/run_record.md

Prints the aggregate as JSON on **stdout**; the markdown table and all commentary
go to stderr unless ``--markdown`` names a file. ``--report`` writes the generated
run record: environment, denominators, metrics, and the comparison. Regenerate it,
never hand-edit it.
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


def parse_versions(path):
    """Parse ``versions.txt`` (``"<module> <version...>"`` per line) into a dict.

    The python line carries the full ``sys.version`` with spaces in it, so only the
    first token is the key and the rest is the value verbatim. A module that failed
    to import was written as ``"<module> MISSING <error>"`` and is kept as such --
    that is a fact about the run, not a parse failure.
    """
    if not os.path.isfile(path):
        return None
    versions = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                versions[parts[0]] = parts[1]
    return versions or None


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        try:
            return json.load(handle)
        except ValueError as exc:
            raise OsiePreflightError("{}: not valid JSON ({})".format(path, exc))


def check_versions(runs):
    """Pooled runs must share one resolved stack (D5, TechStack section 1.1)."""
    seen = {}
    for run in runs:
        if run["versions"] is None:
            continue
        key = json.dumps(run["versions"], sort_keys=True)
        seen.setdefault(key, []).append(run["seed"])
    if len(seen) > 1:
        groups = [{"seeds": s, "versions": json.loads(k)} for k, s in seen.items()]
        detail = "\n  ".join(
            "seeds {}: {}".format(g["seeds"], g["versions"]) for g in groups)
        raise OsiePreflightError(
            "seed runs used DIFFERENT resolved stacks, so they are not replicates "
            "and their spread would mix sampling noise with a version change "
            "(TechStack section 1.1):\n  " + detail)
    return json.loads(list(seen)[0]) if seen else None


def check_preflight(runs):
    """The runs must have scored the same data.

    ``check_fixations.py`` already re-derives these per run; the point here is that
    three runs over *different* label files or splits must not be pooled. Compared
    on the fields that define what was scored -- not on ``fix_path``, which can
    legitimately differ by absolute path.
    """
    fields = ("split", "fewshot_subjects", "origin_size", "n_images", "n_subjects",
              "n_cells", "t_min_ms", "t_max_ms", "n_short", "oob")
    seen = {}
    for run in runs:
        pre = run["preflight"]
        if pre is None:
            continue
        key = json.dumps({f: pre.get(f) for f in fields}, sort_keys=True)
        seen.setdefault(key, []).append(run["seed"])
    if len(seen) > 1:
        detail = "\n  ".join(
            "seeds {}: {}".format(s, json.loads(k)) for k, s in seen.items())
        raise OsiePreflightError(
            "seed runs scored DIFFERENT data -- the preflight counters disagree, so "
            "the runs are not replicates:\n  " + detail)
    return json.loads(list(seen)[0]) if seen else None


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
    # Distinguish "you are in the wrong directory" from "the sweep has not run".
    # test.py anchors `result/<eval>/log` to ISP/<DATASET>/GazeformerISP/, so a
    # perfectly good sweep looks like a missing one when the tool is invoked from
    # the repo root -- and telling someone to re-run three GPU jobs they already
    # ran is the worst possible advice here.
    if not os.path.isdir(log_dir):
        hint = ""
        tail = log_dir.replace("\\", "/").lstrip("./")
        if tail.startswith("result/"):
            # Forward slashes deliberately: this hint is read on the cluster and
            # pasted into a shell, so os.path.join's backslashes would be wrong.
            branch_relative = "ISP/<DATASET>/GazeformerISP/" + tail
            hint = (" Note that `result/...` is relative to "
                    "ISP/<DATASET>/GazeformerISP/, not to the repo root -- from the "
                    "repo root this is probably `{}` (e.g. ISP/OSIE/GazeformerISP/"
                    "...).".format(branch_relative))
        raise OsiePreflightError(
            "--log-dir does not exist: {} (cwd: {}).{}".format(
                log_dir, os.getcwd(), hint))

    found = []
    for path in sorted(glob.glob(os.path.join(log_dir, "seed*"))):
        seed = _seed_of(path)
        if seed is not None and os.path.isdir(path):
            found.append((seed, path))
    if not found:
        present = sorted(os.listdir(log_dir))[:10]
        raise OsiePreflightError(
            "no seed<N>/ directories under {} (it contains: {}). bash/test_osie.sh "
            "writes them, one per run; run the sweep before aggregating it.".format(
                log_dir, ", ".join(present) if present else "nothing"))
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
        "versions": parse_versions(os.path.join(seed_dir, "versions.txt")),
        "preflight": read_json(
            os.path.join(seed_dir, "preflight_fixations.json")),
        "has_prediction": os.path.isfile(
            os.path.join(seed_dir, "prediction.json")),
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


def _best_aliases_base(runs):
    """True when every run has ``SED_best == SED`` and ``STDE_best == STDE``.

    Structurally it always is -- see the note this feeds -- but it is checked
    against the data rather than asserted, so the claim in the report is one this
    run's own numbers support.
    """
    seen = False
    for run in runs:
        vame = run["metrics"].get("VAME", {})
        for base in ("SED", "STDE"):
            best = "{}_best".format(base)
            if base not in vame or best not in vame:
                return False
            if vame[base] != vame[best]:
                return False
            seen = True
    return seen


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


def load_reference(path):
    """Load the transcribed published row, or ``None``.

    The file is data with provenance, not a constant baked into this tool: nothing
    in the repository contains the paper's numbers, so they are typed in once by a
    human and the file records who did it and from where. Entries left ``null`` are
    treated as not-yet-transcribed and reported as such -- an absent number must
    read as absent, never as a zero or as agreement.
    """
    if path is None:
        return None
    reference = read_json(path)
    if reference is None:
        raise OsiePreflightError("--reference file not found: {}".format(path))
    if not isinstance(reference, dict) or "metrics" not in reference:
        raise OsiePreflightError(
            "{}: expected an object with a 'metrics' key (and ideally 'source' / "
            "'transcribed_by' for provenance)".format(path))
    return reference


def compare_to_reference(result, reference):
    """Ours vs the published row, per metric, with the signed delta.

    No tolerance and no verdict: a threshold for "close enough" is a scientific
    judgement (F7's, and it depends on the seed spread), not something a parser
    should assert. This reports the numbers and the gap; a human reads them.
    """
    rows = []
    ref_metrics = reference.get("metrics") or {}
    for metrics_key, metric_name in D6_ORDER:
        ours = result["metrics"].get(metrics_key, {}).get(metric_name)
        if ours is None:
            continue
        theirs = (ref_metrics.get(metrics_key) or {}).get(metric_name)
        rows.append({
            "metrics_key": metrics_key,
            "metric_name": metric_name,
            "ours_mean": ours["mean"],
            "ours_std": ours["std"],
            "paper": theirs,
            "delta": None if theirs is None else ours["mean"] - theirs,
            "within_seed_spread": (
                None if theirs is None or ours["std"] is None
                else abs(ours["mean"] - theirs) <= ours["std"]),
        })
    ref_comp = reference.get("composites") or {}
    for name in ("SM", "MM", "SED"):
        ours = result["composites"][name]
        theirs = ref_comp.get(name)
        rows.append({
            "metrics_key": "headline",
            "metric_name": name,
            "ours_mean": ours["mean"],
            "ours_std": ours["std"],
            "paper": theirs,
            "delta": None if theirs is None else ours["mean"] - theirs,
            "within_seed_spread": (
                None if theirs is None or ours["std"] is None
                else abs(ours["mean"] - theirs) <= ours["std"]),
        })
    missing = [r["metric_name"] for r in rows if r["paper"] is None]
    return {
        "source": reference.get("source"),
        "transcribed_by": reference.get("transcribed_by"),
        "transcribed_utc": reference.get("transcribed_utc"),
        "reference_notes": reference.get("notes"),
        "rows": rows,
        "not_transcribed": missing,
    }


def aggregate(log_dir, reference_path=None):
    runs = [read_seed(seed, path) for seed, path in collect_seed_dirs(log_dir)]
    shared_args = check_invariant_args(runs)
    shared_versions = check_versions(runs)
    shared_preflight = check_preflight(runs)

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
    if _best_aliases_base(runs):
        notes.append(
            "SED_best == SED and STDE_best == STDE, and always will: "
            "evaluation.py does `SED_best_metrics = SED_metrics_rlts`, a bare alias "
            "with no best-of-N selection (OSIE evaluation.py ~line 155). The name "
            "promises a selection the code never performs, in every configuration "
            "and at any --eval_repeat_num. They are NOT a second, corroborating "
            "result -- never report them as one. Frozen under D1: documented, not "
            "fixed. They are excluded from the D6 table for this reason.")
    if subject_num and subject_num <= 5:
        notes.append(
            "pr5 (R@5) is STRUCTURALLY SATURATED at 100.0: with subject_num={} "
            "every rank lies in {{0..{}}}, so rank < 5 is always true. It is "
            "reported because D6 names it, and must be annotated wherever it "
            "appears (Roadmap F1).".format(subject_num, subject_num - 1))
    if incomplete:
        notes.append("INCOMPLETE across seeds, pooled over fewer runs: {}".format(
            ", ".join(incomplete)))

    if shared_versions is None:
        notes.append(
            "No versions.txt in any seed directory -- the resolved stack is NOT "
            "recorded for this sweep, so a future discrepancy cannot be attributed "
            "to it (D5, TechStack section 1.1).")
    if shared_preflight is None:
        notes.append(
            "No preflight_fixations.json in any seed directory -- the denominators "
            "below are unverified: what was actually scored is not recorded.")

    result = {
        "log_dir": log_dir,
        "n_seeds": len(runs),
        "seeds": [run["seed"] for run in runs],
        "shared_args": shared_args,
        "environment": shared_versions,
        "preflight": shared_preflight,
        "composites": composite,
        "metrics": metrics,
        "d6_order": [list(_) for _ in D6_ORDER],
        "per_seed": [
            {"seed": run["seed"], "log_file": run["log_file"],
             "composites": run["composites"],
             "headline_printed": run["headline_printed"],
             "has_prediction": run["has_prediction"]}
            for run in runs
        ],
        "notes": notes,
    }

    reference = load_reference(reference_path)
    if reference is not None:
        result["comparison"] = compare_to_reference(result, reference)
        if result["comparison"]["not_transcribed"]:
            notes.append(
                "The reference file leaves {} not transcribed: {}. Those rows show "
                "the paper column as '--', which means UNKNOWN, not "
                "agreement.".format(
                    len(result["comparison"]["not_transcribed"]),
                    ", ".join(result["comparison"]["not_transcribed"])))
    else:
        notes.append(
            "No --reference given, so there is NO comparison to the published row "
            "in this report. Nothing in the repository contains the paper's "
            "numbers (result-images/main-result.png is qualitative), so they must "
            "be transcribed once from the PDF into a reference JSON. Until then "
            "'consistent with the published row' rests on an off-repo hand check "
            "and cannot be re-derived from artefacts.")
    return result


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


def _num(value, places=4):
    return "--" if value is None else "{:.{p}f}".format(value, p=places)


def to_report(result):
    """The full F1 run record, generated from the stored artefacts.

    This is what a hand-written notes.md would have contained, except that every
    number is read back from the run's own outputs on each invocation, so it cannot
    drift from them. The two things it cannot generate -- the published row and the
    verdict -- are named as such rather than quietly left out.
    """
    seeds = ", ".join(str(_) for _ in result["seeds"])
    env = result["environment"] or {}
    pre = result["preflight"] or {}
    args = result["shared_args"]
    out = [
        "# OSIE eval baseline — generated run record",
        "",
        "Generated by `tools/osie_prep/aggregate_seeds.py` from the artefacts under",
        "`{}`. **Do not hand-edit** — regenerate it instead; every number below is".format(
            result["log_dir"]),
        "read back from the run's own stored outputs, which is what makes it",
        "re-derivable without a GPU (D5).",
        "",
        "## Run",
        "",
        "| | |",
        "|---|---|",
        "| seeds | {} (n = {}) |".format(seeds, result["n_seeds"]),
    ]
    for key in ("evaluation_dir", "user_emb_path", "fix_dir", "feat_dir", "emb_dir",
                "subject_num", "fewshot_subject", "num_fewshot", "random_support",
                "eval_repeat_num", "width", "height", "max_length"):
        if key in args:
            out.append("| `{}` | `{}` |".format(key, args[key]))
    out.append("| `prediction.json` kept per seed | {} |".format(
        "yes" if all(_["has_prediction"] for _ in result["per_seed"]) else "NO"))
    out.append("")

    out.append("## Environment (D5 — resolved, not pinned)")
    out.append("")
    if env:
        out.append("| package | resolved |")
        out.append("|---|---|")
        for name, version in env.items():
            out.append("| {} | {} |".format(name, version))
        out.append("")
        out.append("Compare against `ISP/environment.yml` — TechStack §1.1 records why a newer stack")
        out.append("is tolerated here and exactly what that does and does not license.")
    else:
        out.append("*Not recorded — no `versions.txt` in the seed directories.*")
    out.append("")

    out.append("## What was scored (the denominators)")
    out.append("")
    if pre:
        out.append("| | |")
        out.append("|---|---|")
        for key in ("split", "n_images", "n_subjects", "n_cells", "n_records_total",
                    "fewshot_subjects", "origin_size", "t_min_ms", "t_max_ms",
                    "gt_length_min", "gt_length_median", "gt_length_max"):
            if key in pre:
                out.append("| {} | `{}` |".format(key, pre[key]))
        for key in ("oob", "n_short"):
            if key in pre:
                out.append("| {} (D7 counter) | `{}` |".format(key, pre[key]))
        out.append("")
        if pre.get("n_short"):
            out.append(
                "> `n_short` = {} scanpaths shorter than 3 fixations. The frozen evaluator pads "
                "these to length 3 with `(1., 1., 1e-3)` and the padded array then replaces the "
                "original for *all* subsequent metrics in that cell (TechStack §4).".format(
                    pre["n_short"]))
            out.append("")
    else:
        out.append("*Not recorded — no `preflight_fixations.json` in the seed directories.*")
        out.append("")

    out.append("## Metrics")
    out.append("")
    out.append(to_markdown(result))

    comparison = result.get("comparison")
    if comparison:
        out.append("## Versus the published OSIE row")
        out.append("")
        if comparison.get("source"):
            out.append("Source: {}".format(comparison["source"]))
        prov = [_ for _ in (comparison.get("transcribed_by"),
                            comparison.get("transcribed_utc")) if _]
        if prov:
            out.append("Transcribed by {}.".format(", ".join(str(_) for _ in prov)))
        if comparison.get("reference_notes"):
            out.append("")
            out.append(comparison["reference_notes"])
        out.append("")
        # No literal "|" in a header cell -- it would split the column.
        out.append("| metric | ours (mean) | ±(seed) | paper | delta | within seed spread |")
        out.append("|---|---|---|---|---|---|")
        for row in comparison["rows"]:
            label = ("**{}**".format(row["metric_name"])
                     if row["metrics_key"] == "headline"
                     else "{} / {}".format(row["metrics_key"], row["metric_name"]))
            within = ("" if row["within_seed_spread"] is None
                      else ("yes" if row["within_seed_spread"] else "**no**"))
            out.append("| {} | {} | {} | {} | {} | {} |".format(
                label, _num(row["ours_mean"]), _num(row["ours_std"]),
                _num(row["paper"]), _num(row["delta"]), within))
        out.append("")
        out.append("`--` in the paper column means **not transcribed**, not agreement.")
        out.append("")
        out.append("> The last column is a *descriptive* check — whether the gap is smaller than this")
        out.append("> sweep's own re-sampling spread. It is not a pass/fail criterion: at n = 3 the")
        out.append("> spread is a noisy estimate, and what counts as reproducing the row is F7's")
        out.append("> judgement to state, not this tool's to assert.")
    else:
        out.append("## Versus the published OSIE row")
        out.append("")
        out.append("*No `--reference` supplied — this report contains no comparison.* Nothing in the")
        out.append("repository holds the paper's numbers (`result-images/main-result.png` is")
        out.append("qualitative), so they must be transcribed once from the PDF into a reference")
        out.append("JSON, with provenance, before the comparison can be re-derived from artefacts.")
    out.append("")

    out.append("## What this does not settle")
    out.append("")
    out.append("The verdict — what these numbers license, given that the checkpoint was trained on")
    out.append("OSIE subjects 0–9 and the query set is subjects 10–14 — is an argument, not a")
    out.append("measurement, and is deliberately not generated here. It belongs in the spec and in")
    out.append("F7. See `spec/constitution/Mission.md` §5 and Roadmap F7.")
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
    parser.add_argument("--reference", dest="reference_path", default=None,
                        help="JSON holding the paper's published row, transcribed "
                             "once by hand with its provenance; enables the "
                             "comparison table")
    parser.add_argument("--report", dest="report_path", default=None,
                        help="write the full generated run record here (env, "
                             "denominators, metrics, comparison). This replaces a "
                             "hand-written notes.md -- regenerate, never hand-edit")
    args = parser.parse_args(argv)

    try:
        result = aggregate(args.log_dir, reference_path=args.reference_path)
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

    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as handle:
            handle.write(to_report(result))
        sys.stderr.write("aggregate_seeds: wrote {}\n".format(args.report_path))

    sys.stderr.write("aggregate_seeds: OK -- {} seeds {}\n".format(
        result["n_seeds"], result["seeds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
