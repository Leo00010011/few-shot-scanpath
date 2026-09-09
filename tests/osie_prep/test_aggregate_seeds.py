"""Validation for aggregate_seeds -- the FR13.3 seed-sweep aggregator.

The aggregator's whole reason to exist is that pooling the *wrong* runs produces a
plausible, wrong noise band, which is precisely the D7 failure mode. So every guard
case asserts both that OsiePreflightError is raised *and* that the message names the
offending seed / argument -- a bare "check failed" is not a loud failure.

Logs are synthesised here in test.py's exact emitted format rather than captured from
a run: the format is the contract being tested, and a captured log would pin only the
one configuration it happened to come from.
"""

import json
import os
import statistics

import pytest

from osie_prep import OsiePreflightError
from osie_prep.aggregate_seeds import (
    aggregate, composites, parse_headline, parse_run_log, parse_versions,
    to_markdown, to_report,
)

#: logger.py's format. The timestamp's own colons are the reason the prefix has to
#: come off before any `key: value` parsing.
TS = "[09/09/2026 10:11:12 AM - root - INFO] "

BASE_METRICS = {
    "MultiMatch": {"vector": 0.9012, "direction": 0.6543, "length": 0.9101,
                   "position": 0.8432, "duration": 0.5678},
    "ScanMatch": {"w/o duration": 0.4123, "with duration": 0.3456},
    # SED_best/STDE_best mirror SED/STDE exactly, because evaluation.py aliases them
    # (`SED_best_metrics = SED_metrics_rlts`) with no best-of-N selection. F1's seed 0
    # shows 7.3000/7.3000 and 0.8460/0.8460. A fixture with them differing would be
    # testing a branch the frozen code cannot produce.
    "VAME": {"SED": 6.2400, "STDE": 0.7100, "SED_best": 6.2400,
             "STDE_best": 0.7100},
    "retrieval scanmatch w/ duration": {"pmrr": 0.5100, "pr1": 30.0, "pr3": 70.0,
                                        "pr5": 100.0, "rsum": 200.0},
}

BASE_ARGS = {
    "seed": 0, "subject_num": 5, "fewshot_subject": "[10, 11, 12, 13, 14]",
    "eval_repeat_num": 1, "width": 512, "height": 384,
    "fix_dir": "src/data/fixations.json", "feat_dir": "/w/image_features",
    "emb_dir": "src/data/embeddings.npy",
    "user_emb_path": "/w/fewshot_user_embedding_10.pt",
    "evaluation_dir": "src/assets/OSIE-ex-10to15", "max_length": 16,
    "im_h": 24, "im_w": 32, "num_fewshot": 10, "random_support": 0,
}


def render_log(run_args, metrics):
    """Reproduce test.py's two logged blocks byte-for-byte in format."""
    lines = [TS + "The args corresponding to testing process are: "]
    for key, value in run_args.items():
        lines.append(TS + "{key:20}: {value:}".format(key=key, value=value))
    lines.append(TS + "The metrics for best model performance are: ")
    for metrics_key, block in metrics.items():
        for metric_name, value in block.items():
            lines.append(TS + "{metrics_key:10}-{metric_name:15}: {v:.4f}".format(
                metrics_key=metrics_key, metric_name=metric_name, v=value))
    return "\n".join(lines) + "\n"


def _scaled(factor):
    out = {k: {n: round(v * factor, 4) for n, v in b.items()}
           for k, b in BASE_METRICS.items()}
    out["retrieval scanmatch w/ duration"]["pr5"] = 100.0
    return out


def write_seed(log_dir, seed, metrics=None, args_override=None, headline=True,
               dir_name=None, log_name="log_test_subject_10_0.txt"):
    """Materialise one seed<N>/ directory as bash/test_osie.sh would."""
    metrics = metrics if metrics is not None else _scaled(1.0 + 0.002 * seed)
    run_args = dict(BASE_ARGS)
    run_args["seed"] = seed
    run_args.update(args_override or {})

    seed_dir = os.path.join(log_dir, dir_name or "seed{}".format(seed))
    os.makedirs(seed_dir, exist_ok=True)
    with open(os.path.join(seed_dir, log_name), "w", encoding="utf-8") as handle:
        handle.write(render_log(run_args, metrics))

    if headline:
        derived = composites(metrics)
        with open(os.path.join(seed_dir, "stdout.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write("python 3.11.14\nnumpy 2.1.2\n")
            handle.write("SM: {}, MM: {}, SED: {}\n".format(
                round(derived["SM"], 3), round(derived["MM"], 3),
                round(derived["SED"], 3)))
    return seed_dir


def make_sweep(tmp_path, seeds=(0, 1, 2), **kwargs):
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir, exist_ok=True)
    for seed in seeds:
        write_seed(log_dir, seed, **kwargs)
    return log_dir


# --- group 1: parsing the upstream log format -------------------------------

def test_parses_padded_keys_and_names(tmp_path):
    # "ScanMatch" is 9 chars against {:10} so it carries a trailing space before the
    # hyphen, and "w/o duration" is padded to 15. Neither field is ever truncated.
    path = os.path.join(str(tmp_path), "log_test_subject_10_0.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_log(BASE_ARGS, BASE_METRICS))
    run_args, metrics = parse_run_log(path)
    assert metrics["ScanMatch"]["w/o duration"] == pytest.approx(0.4123)
    assert metrics["MultiMatch"]["vector"] == pytest.approx(0.9012)
    assert metrics["VAME"]["STDE_best"] == pytest.approx(0.7100)
    # The retrieval key contains spaces and a slash but no hyphen, which is what
    # makes splitting on the first hyphen unambiguous.
    assert metrics["retrieval scanmatch w/ duration"]["pmrr"] == pytest.approx(0.51)
    assert run_args["seed"] == "0"
    assert run_args["fix_dir"] == "src/data/fixations.json"


def test_last_block_wins_on_an_appended_log(tmp_path):
    # test.py truncates the log at the top of main(), so one block is the norm. A
    # file that was appended to must still not yield a silent mixture of two runs.
    path = os.path.join(str(tmp_path), "log_test_subject_10_0.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_log(BASE_ARGS, _scaled(1.0)))
        handle.write(render_log(dict(BASE_ARGS, seed=7), _scaled(2.0)))
    run_args, metrics = parse_run_log(path)
    assert run_args["seed"] == "7"
    assert metrics["MultiMatch"]["vector"] == pytest.approx(round(0.9012 * 2, 4))


def test_missing_metric_block_raises(tmp_path):
    # A run that died before comprehensive_evaluation_by_subject() has an arg block
    # and nothing else. Pooling it would quietly shrink the sweep.
    path = os.path.join(str(tmp_path), "log_test_subject_10_0.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(TS + "The args corresponding to testing process are: \n")
        handle.write(TS + "{key:20}: {value:}\n".format(key="seed", value=0))
    with pytest.raises(OsiePreflightError) as exc:
        parse_run_log(path)
    assert "metric block" in str(exc.value)


def test_missing_arg_block_raises(tmp_path):
    path = os.path.join(str(tmp_path), "log_test_subject_10_0.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(TS + "The metrics for best model performance are: \n")
    with pytest.raises(OsiePreflightError) as exc:
        parse_run_log(path)
    assert "arg-namespace" in str(exc.value)


def test_headline_absent_returns_none(tmp_path):
    assert parse_headline(os.path.join(str(tmp_path), "nope.txt")) is None


# --- group 2: the composites match test.py's final lines --------------------

def test_composites_follow_the_frozen_formulas():
    derived = composites(BASE_METRICS)
    assert derived["SM"] == pytest.approx(
        statistics.harmonic_mean([0.4123, 0.3456]))
    assert derived["MM"] == pytest.approx(
        statistics.fmean([0.9012, 0.6543, 0.9101, 0.8432, 0.5678]))
    assert derived["SED"] == pytest.approx(6.24)


def test_composites_missing_block_raises():
    with pytest.raises(OsiePreflightError) as exc:
        composites({"MultiMatch": BASE_METRICS["MultiMatch"]})
    assert "ScanMatch" in str(exc.value)


# --- group 3: the pooling guards -------------------------------------------

def test_happy_path_pools_three_seeds(tmp_path):
    result = aggregate(make_sweep(tmp_path))
    assert result["n_seeds"] == 3
    assert result["seeds"] == [0, 1, 2]
    assert result["metrics"]["ScanMatch"]["with duration"]["n"] == 3
    assert result["composites"]["SM"]["std"] is not None
    assert result["shared_args"]["subject_num"] == "5"


def test_across_seed_std_is_the_sample_std_of_the_three_values(tmp_path):
    result = aggregate(make_sweep(tmp_path))
    entry = result["metrics"]["MultiMatch"]["vector"]
    assert entry["std"] == pytest.approx(statistics.stdev(entry["values"]))
    assert entry["mean"] == pytest.approx(statistics.fmean(entry["values"]))
    assert entry["min"] == min(entry["values"])
    assert entry["max"] == max(entry["values"])


def test_seed_directory_disagreeing_with_logged_seed_raises(tmp_path):
    # The exact failure the copy step in test_osie.sh could produce: a seed dir
    # holding another seed's artefacts. Pooling it reports a spread over the wrong
    # runs, and nothing else would notice.
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    write_seed(log_dir, 0)
    write_seed(log_dir, 1)
    write_seed(log_dir, 0, dir_name="seed2")     # dir says 2, log says 0
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    msg = str(exc.value)
    assert "seed2" in msg
    assert "logged seed 0" in msg


def test_differing_invariant_arg_raises_and_names_it(tmp_path):
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    write_seed(log_dir, 0)
    write_seed(log_dir, 1)
    write_seed(log_dir, 2, args_override={"subject_num": 3})
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    msg = str(exc.value)
    assert "subject_num" in msg
    assert "not replicates" in msg


def test_differing_fix_dir_raises(tmp_path):
    # The osie_fixations_update_duration.json trap (TechStack section 3.7) pooled
    # into a sweep alongside the canonical labels.
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    write_seed(log_dir, 0)
    write_seed(log_dir, 1, args_override={
        "fix_dir": "data/osie_fixations_update_duration.json"})
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    assert "fix_dir" in str(exc.value)


def test_seed_varying_is_not_treated_as_a_config_change(tmp_path):
    # seed is deliberately absent from INVARIANT_ARGS -- it is the thing that varies.
    aggregate(make_sweep(tmp_path))


def test_no_seed_dirs_raises(tmp_path):
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    assert "no seed<N>/ directories" in str(exc.value)


def test_absent_log_dir_is_distinguished_from_an_unrun_sweep(tmp_path):
    # Telling someone to re-run three GPU jobs they already ran is the worst
    # possible advice, so a missing directory must not read as a missing sweep.
    missing = os.path.join(str(tmp_path), "log")
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(missing)
    msg = str(exc.value)
    assert "does not exist" in msg
    assert "run the sweep" not in msg


def test_repo_root_invocation_gets_the_branch_relative_hint(tmp_path, monkeypatch):
    # `result/<eval>/log` is anchored to ISP/<DATASET>/GazeformerISP/ by test.py, so
    # a good sweep looks absent when the tool is run from the repo root.
    monkeypatch.chdir(str(tmp_path))
    with pytest.raises(OsiePreflightError) as exc:
        aggregate("result/OSIE-ex-10to15/log")
    msg = str(exc.value)
    assert "GazeformerISP" in msg
    assert "relative to" in msg


def test_no_seed_dirs_lists_what_is_actually_there(tmp_path):
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    open(os.path.join(log_dir, "prediction.json"), "w").close()
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    assert "prediction.json" in str(exc.value)


def test_two_log_files_in_one_seed_dir_raises(tmp_path):
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    write_seed(log_dir, 0)
    write_seed(log_dir, 0, log_name="log_test_subject_10_1.txt")
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    assert "expected 1" in str(exc.value)


# --- group 4: the stdout headline cross-check -------------------------------

def test_headline_agreement_is_recorded_and_within_formatting_noise(tmp_path):
    result = aggregate(make_sweep(tmp_path))
    for run in result["per_seed"]:
        assert run["headline_printed"] is not None
        assert run["headline_printed"]["SM"] == pytest.approx(
            run["composites"]["SM"], abs=1e-3)


def test_mismatched_stdout_and_log_raises(tmp_path):
    # A stdout.txt left behind by a previous seed while the log file is current.
    log_dir = os.path.join(str(tmp_path), "log")
    os.makedirs(log_dir)
    write_seed(log_dir, 0)
    seed_dir = write_seed(log_dir, 1)
    with open(os.path.join(seed_dir, "stdout.txt"), "w", encoding="utf-8") as handle:
        handle.write("SM: 0.900, MM: 0.900, SED: 1.000\n")
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    assert "not from the same run" in str(exc.value)


def test_missing_stdout_still_aggregates(tmp_path):
    # The headline is a cross-check, not the source of truth: an older run without a
    # captured stdout must still pool.
    result = aggregate(make_sweep(tmp_path, headline=False))
    assert result["n_seeds"] == 3
    assert all(run["headline_printed"] is None for run in result["per_seed"])
    assert result["composites"]["MM"]["mean"] > 0


# --- group 5: what the report says about itself -----------------------------

def test_notes_flag_the_across_seed_denominator(tmp_path):
    result = aggregate(make_sweep(tmp_path))
    joined = " ".join(result["notes"])
    assert "ACROSS-SEED" in joined
    assert "cur_metrics_std" in joined


def test_notes_flag_r5_saturation_at_five_subjects(tmp_path):
    result = aggregate(make_sweep(tmp_path))
    assert any("STRUCTURALLY SATURATED" in note for note in result["notes"])


def test_no_r5_saturation_note_above_five_subjects(tmp_path):
    result = aggregate(make_sweep(tmp_path, args_override={"subject_num": 15}))
    assert not any("SATURATED" in note for note in result["notes"])


def test_notes_disclaim_the_coco_fv_r3_defect(tmp_path):
    result = aggregate(make_sweep(tmp_path))
    assert any("rank < 3" in note for note in result["notes"])


def test_markdown_renders_the_d6_block(tmp_path):
    table = to_markdown(aggregate(make_sweep(tmp_path)))
    for name in ("vector", "direction", "length", "position", "duration",
                 "w/o duration", "with duration", "SED", "STDE",
                 "pmrr", "pr1", "pr3", "pr5"):
        assert name in table
    for headline in ("| SM |", "| MM |", "| SED |"):
        assert headline in table
    assert "ACROSS-SEED" in table


def test_result_is_json_serialisable(tmp_path):
    # It is written straight to metrics_sweep.json; a numpy scalar sneaking in would
    # only surface at write time on the cluster.
    json.dumps(aggregate(make_sweep(tmp_path)))


def test_notes_flag_sed_best_as_an_alias(tmp_path):
    # evaluation.py does `SED_best_metrics = SED_metrics_rlts` -- a bare alias with
    # no best-of-N selection -- so SED_best is identically SED in every
    # configuration. Observed on F1's seed 0: 7.3000 / 7.3000.
    result = aggregate(make_sweep(tmp_path))
    assert any("SED_best == SED" in note for note in result["notes"])


def test_no_alias_note_when_the_values_actually_differ(tmp_path):
    # Defensive: the note asserts something about this run's numbers, so it must
    # not fire on a branch where the two genuinely diverge.
    metrics = _scaled(1.0)
    metrics["VAME"]["SED_best"] = metrics["VAME"]["SED"] - 0.5
    result = aggregate(make_sweep(tmp_path, metrics=metrics))
    assert not any("SED_best == SED" in note for note in result["notes"])


def test_sed_best_is_absent_from_the_d6_table(tmp_path):
    # Absent from the table ROWS. The explanatory note below the table names them on
    # purpose, so the whole-document check would be self-defeating.
    table = to_markdown(aggregate(make_sweep(tmp_path)))
    rows = [_ for _ in table.splitlines() if _.startswith("|")]
    assert not any("SED_best" in row for row in rows)
    assert not any("STDE_best" in row for row in rows)
    assert any("SED_best == SED" in _ for _ in table.splitlines())


# --- group 6: the report reads the OTHER stored artefacts -------------------

VERSIONS = ("python 3.11.14 (main, Jan  1 2026, 00:00:00) [GCC 11.4.0]\n"
            "torch 2.10.0+cu126\nnumpy 2.1.2\nscipy 1.14.1\n"
            "skimage 0.19.3\nmultimatch_gaze 0.1.3\ncv2 4.9.0.80\n")

PREFLIGHT = {
    "fix_path": "/w/fixations.json", "split": "test",
    "fewshot_subjects": [10, 11, 12, 13, 14], "origin_size": [600, 800],
    "n_records_total": 10500, "n_rows": 350, "n_images": 70, "n_cells": 350,
    "n_subjects": 5, "t_min_ms": 20, "t_max_ms": 1033, "oob": 0, "n_short": 4,
    "gt_length_min": 1, "gt_length_median": 12, "gt_length_max": 16,
}


def write_extras(seed_dir, versions=VERSIONS, preflight=PREFLIGHT,
                 prediction=True):
    if versions is not None:
        with open(os.path.join(seed_dir, "versions.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write(versions)
    if preflight is not None:
        with open(os.path.join(seed_dir, "preflight_fixations.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(preflight, handle)
    if prediction:
        with open(os.path.join(seed_dir, "prediction.json"), "w",
                  encoding="utf-8") as handle:
            json.dump([], handle)


def full_sweep(tmp_path, seeds=(0, 1, 2), **extras):
    log_dir = make_sweep(tmp_path, seeds=seeds)
    for seed in seeds:
        write_extras(os.path.join(log_dir, "seed{}".format(seed)), **extras)
    return log_dir


def test_environment_and_preflight_are_read_back(tmp_path):
    result = aggregate(full_sweep(tmp_path))
    assert result["environment"]["numpy"] == "2.1.2"
    assert result["environment"]["multimatch_gaze"] == "0.1.3"
    # The python line has spaces in its value; only the first token is the key.
    assert result["environment"]["python"].startswith("3.11.14")
    assert result["preflight"]["n_cells"] == 350
    assert all(run["has_prediction"] for run in result["per_seed"])


def test_differing_stacks_across_seeds_raises(tmp_path):
    log_dir = make_sweep(tmp_path)
    write_extras(os.path.join(log_dir, "seed0"))
    write_extras(os.path.join(log_dir, "seed1"))
    write_extras(os.path.join(log_dir, "seed2"),
                 versions=VERSIONS.replace("numpy 2.1.2", "numpy 1.24.3"))
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    msg = str(exc.value)
    assert "DIFFERENT resolved stacks" in msg
    assert "1.24.3" in msg


def test_differing_preflight_across_seeds_raises(tmp_path):
    log_dir = make_sweep(tmp_path)
    write_extras(os.path.join(log_dir, "seed0"))
    write_extras(os.path.join(log_dir, "seed1"))
    write_extras(os.path.join(log_dir, "seed2"),
                 preflight=dict(PREFLIGHT, n_images=69))
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(log_dir)
    assert "scored DIFFERENT data" in str(exc.value)


def test_absent_extras_are_flagged_not_silently_skipped(tmp_path):
    result = aggregate(make_sweep(tmp_path))          # no versions/preflight
    assert result["environment"] is None
    assert result["preflight"] is None
    joined = " ".join(result["notes"])
    assert "resolved stack is NOT" in joined
    assert "denominators" in joined


def test_report_states_denominators_and_padding(tmp_path):
    report = to_report(aggregate(full_sweep(tmp_path)))
    assert "70" in report and "350" in report          # images, cells
    assert "numpy" in report and "2.1.2" in report
    assert "n_short" in report
    assert "pads these to length 3" in report.replace("\n", " ")


# --- group 7: the comparison against the published row ----------------------

def write_reference(tmp_path, metrics=None, composites_=None, **top):
    ref = {"source": "Xue et al. CVPR 2025, Table 1, OSIE 10-shot",
           "transcribed_by": "tester",
           "metrics": metrics if metrics is not None else {},
           "composites": composites_ if composites_ is not None else {}}
    ref.update(top)
    path = os.path.join(str(tmp_path), "ref.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ref, handle)
    return path


def test_no_reference_says_so_loudly(tmp_path):
    result = aggregate(full_sweep(tmp_path))
    assert "comparison" not in result
    assert any("No --reference given" in note for note in result["notes"])
    assert "no comparison" in to_report(result)


def test_reference_produces_signed_deltas(tmp_path):
    log_dir = full_sweep(tmp_path)
    ours = aggregate(log_dir)["metrics"]["ScanMatch"]["with duration"]["mean"]
    ref = write_reference(
        tmp_path, metrics={"ScanMatch": {"with duration": ours - 0.01}})
    result = aggregate(log_dir, reference_path=ref)
    row = [r for r in result["comparison"]["rows"]
           if r["metric_name"] == "with duration"][0]
    assert row["paper"] == pytest.approx(ours - 0.01)
    assert row["delta"] == pytest.approx(0.01)
    assert row["within_seed_spread"] is False        # 0.01 >> the seed spread


def test_untranscribed_entries_read_as_unknown_never_as_agreement(tmp_path):
    log_dir = full_sweep(tmp_path)
    ref = write_reference(tmp_path, metrics={"VAME": {"SED": None}})
    result = aggregate(log_dir, reference_path=ref)
    row = [r for r in result["comparison"]["rows"]
           if r["metric_name"] == "SED" and r["metrics_key"] == "VAME"][0]
    assert row["paper"] is None
    assert row["delta"] is None
    assert row["within_seed_spread"] is None
    assert "SED" in result["comparison"]["not_transcribed"]
    report = to_report(result)
    assert "means **not transcribed**, not agreement" in report
    # A null must never render as 0.0000 in the paper column. Scope to the
    # comparison section: the metrics table above it has a same-prefixed row whose
    # fourth column is `min`, not `paper`.
    section = report.split("## Versus the published OSIE row", 1)[1]
    paper_cells = [line.split("|")[4].strip() for line in section.splitlines()
                   if line.startswith("| VAME / SED |")]
    assert paper_cells == ["--"]
    assert "0.0000" not in " ".join(
        line for line in section.splitlines()
        if line.startswith("| VAME / SED |"))


def test_missing_reference_file_raises(tmp_path):
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(full_sweep(tmp_path),
                  reference_path=os.path.join(str(tmp_path), "nope.json"))
    assert "not found" in str(exc.value)


def test_malformed_reference_raises(tmp_path):
    path = os.path.join(str(tmp_path), "bad.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([1, 2, 3], handle)
    with pytest.raises(OsiePreflightError) as exc:
        aggregate(full_sweep(tmp_path), reference_path=path)
    assert "metrics" in str(exc.value)


def test_report_carries_provenance_and_refuses_a_verdict(tmp_path):
    ref = write_reference(tmp_path, metrics={"VAME": {"SED": 7.0}})
    report = to_report(aggregate(full_sweep(tmp_path), reference_path=ref))
    assert "Xue et al." in report
    assert "tester" in report
    # The tool reports the gap; it must not declare the run a pass.
    assert "not a pass/fail criterion" in report.replace("\n", " ")
    assert "does not settle" in report


def test_shipped_reference_template_is_valid_and_unfilled(repo_root):
    # The checked-in template must parse and must not have acquired invented
    # numbers: every metric entry starts null until a human transcribes the PDF.
    path = os.path.join(repo_root, "spec", "2026-09-08-osie-eval-baseline",
                        "paper_reference.json")
    with open(path, "r", encoding="utf-8") as handle:
        ref = json.load(handle)
    assert "metrics" in ref and "composites" in ref
    for block in ref["metrics"].values():
        for value in block.values():
            assert value is None or isinstance(value, (int, float))


def test_comparison_table_header_has_no_stray_pipes(tmp_path):
    # A literal "|" inside a header cell splits the column and the table renders
    # with the wrong shape. Every row in the section must have the same cell count.
    ref = write_reference(tmp_path, metrics={"VAME": {"SED": 7.0}})
    report = to_report(aggregate(full_sweep(tmp_path), reference_path=ref))
    section = report.split("## Versus the published OSIE row", 1)[1]
    widths = {len(line.split("|")) for line in section.splitlines()
              if line.startswith("|")}
    assert len(widths) == 1, "ragged comparison table: {}".format(widths)
