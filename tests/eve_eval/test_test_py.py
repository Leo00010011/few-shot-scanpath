"""Group 7 -- ``src/test.py``'s arguments, the batch cap, and the FR9 record.

``src/test.py`` is never imported here: it parses argv at module scope and reaches
``utils/evaluation.py``, whose ``multimatch_gaze`` import is cluster-only. Its parser
and its two pure helpers are executed out of its own AST instead (see the conftest), so
these checks run against the real code rather than a transcription. The parts that only
exist inside ``main()``'s loop are asserted structurally, which is the honest thing to
say about them: the loop needs a GPU and a checkpoint.
"""

import ast
import types

import numpy as np
import pytest

from conftest import TEST_PY

EXPECTED_DEFAULTS = {
    "origin_width": 1920, "origin_height": 1080, "im_h": 24, "im_w": 32,
    "subject_num": 3, "num_fewshot": 10, "batch": 1, "max_batches": -1,
    "width": 512, "height": 384, "max_length": 16, "min_length": 1,
    "action_map_num": 4, "subject_feature_dim": 384, "lm_hidden_dim": 768,
    "eval_repeat_num": 1, "metrics_json": "metrics.json", "heatmap_dir": "",
    "subject_map_path": "", "bridge_report": "", "senet_report": "",
    "feature_report": "",
}


def test_argument_defaults(arg_parser_factory):
    """FR6.1 / FR6.3 -- including the three unchanged ones, which are the metric
    screen and would rescale every coordinate if they moved."""
    parser = arg_parser_factory()
    defaults = {a.dest: a.default for a in parser._actions}
    for key, value in EXPECTED_DEFAULTS.items():
        assert defaults[key] == value, (key, defaults[key], value)


def test_eval_repeat_num_is_pinned_to_one(arg_parser_factory):
    """FR6.5 -- rejected at parse time, with the reason."""
    ok = arg_parser_factory(["--eval_repeat_num", "1"])
    assert ok.eval_repeat_num == 1

    with pytest.raises(SystemExit) as exc:
        arg_parser_factory(["--eval_repeat_num", "2"])
    assert exc.value.code == 2


def test_eval_repeat_num_message_names_the_reason(arg_parser_factory, capsys):
    with pytest.raises(SystemExit):
        arg_parser_factory(["--eval_repeat_num", "2"])
    message = capsys.readouterr().err
    assert "IndexError" in message and "subject_num" in message


def _source():
    with open(TEST_PY, encoding="utf-8") as fh:
        return fh.read()


def test_the_hardcoded_cap_is_gone_and_parameterised():
    """FR6.4 -- the single most dangerous item in F5.

    354 scored images at --batch 1 is 354 batches; the upstream `i_batch > 100` cap
    would stop at 101 (29% of the split) and report a plausible, wrong number.
    """
    source = _source()
    # An AST check, not a text one: the --max_batches help string and the comment
    # above the loop both name `i_batch > 100` deliberately -- saying which cap was
    # replaced, and why it is inert for OSIE and live here, is the point of them.
    caps = [n for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Compare)
            and "i_batch" in ast.dump(n.left)
            and any(isinstance(c, ast.Constant) and c.value == 100
                    for c in n.comparators)]
    assert not caps, "the hardcoded `i_batch > 100` cap still executes (FR6.4)"
    assert "if args.max_batches > 0 and i_batch >= args.max_batches:" in source
    # ...and the belt: a run that covered less than the whole loader must not report.
    assert "if args.max_batches <= 0 and n_batches != len(test_loader):" in source
    assert "the split was truncated (FR6.4)" in source


def test_origin_size_is_passed_explicitly_and_asserted():
    """FR6.2 -- omitting it mis-scales every coordinate by 2.4x / 1.8x (D3)."""
    source = _source()
    assert "origin_size=(args.origin_height, args.origin_width)" in source
    assert "(3.75, 2.8125)" in source


def test_reproducibility_lines_are_unchanged():
    """FR6.6 (D5)."""
    source = _source()
    for line in ("np.random.seed(args.seed)", "torch.manual_seed(args.seed)",
                 "torch.cuda.manual_seed_all(args.seed)",
                 "torch.backends.cudnn.benchmark = False",
                 "torch.backends.cudnn.deterministic = True"):
        assert line in source, line


def test_headline_print_is_left_as_upstream_has_it():
    """FR6.7 -- a bare print() to stdout; the run script tees it."""
    source = _source()
    assert ("print('SM: {}, MM: {}, SED: {}'.format(round(SM, 3), round(MM, 3), "
            "round(SED, 3)))") in source
    # ...and the same three numbers are written into metrics.json, so recovering them
    # never requires parsing formatted text.
    assert '"headline": {"SM": float(SM), "MM": float(MM), "SED": float(SED)}' in source
    assert 'SM = scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))' in source
    assert 'MM = np.mean(list(cur_metrics["MultiMatch"].values()))' in source


def test_headline_formulas_match_D6():
    """D6 -- SM is the harmonic mean of the two ScanMatch values, MM the arithmetic
    mean of the five MultiMatch dimensions. Computed here on a stand-in so the
    definition itself is pinned, not only the source line."""
    import scipy.stats

    scanmatch = {"w/o duration": 0.4, "with duration": 0.3}
    multimatch = {"vector": 0.9, "direction": 0.7, "length": 0.85, "position": 0.8,
                  "duration": 0.75}
    assert scipy.stats.hmean(list(scanmatch.values())) == pytest.approx(
        2 / (1 / 0.4 + 1 / 0.3), abs=1e-12)
    assert np.mean(list(multimatch.values())) == pytest.approx(0.8, abs=1e-12)


def test_metrics_json_is_written_unconditionally():
    """FR9.1 -- including on a run that produced no predictions.

    The write is at main()'s top level, not inside ``if len(predict_results)``; that
    is the structural difference, so it is asserted structurally.
    """
    tree = ast.parse(_source())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = set()
    for node in ast.walk(main):
        if isinstance(node, ast.If) and "predict_results" in ast.dump(node.test):
            guarded.update(id(c) for c in ast.walk(node))
    writes = [n for n in ast.walk(main)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "dump"]
    metrics_writes = [n for n in writes if "metrics_record" in ast.dump(n)]
    assert metrics_writes, "metrics.json is never written"
    for node in metrics_writes:
        assert id(node) not in guarded, (
            "the metrics.json write sits inside `if len(predict_results)` -- FR9.1 "
            "requires it on every run")


def test_per_cell_std_is_null_with_a_reason():
    """FR9.2 -- present and null, never omitted: absence must not read as zero."""
    source = _source()
    assert '"per_cell_std": None' in source
    assert '"per_cell_std_reason"' in source
    assert "F6" in source


def test_notes_carry_every_FR13_annotation(test_py_functions):
    """FR13.1 - FR13.6 and the SED_best/STDE_best alias statement (FR9.3)."""
    class _Args(object):
        subject_num = 3
        num_fewshot = 10

    nan_drops = {"n_dropped": 7, "n_diagonal_cells": 1062}
    notes = test_py_functions["reporting_notes"](_Args(), nan_drops, 10)
    blob = " ".join(notes.values())

    assert "R@3 is STRUCTURALLY SATURATED" in blob            # FR13.1
    assert "IDENTITIES VARYING BY IMAGE" in blob              # FR13.2
    assert "3.75 / 2.8125" in blob and "0.5333" in blob and "0.2667" in blob  # FR13.3
    assert "surplus_trial" in blob and "short_scanpath" in blob               # FR13.4
    assert "OSIE" in blob and "OPEN-7" in blob                # FR13.5
    assert "7 of 1062" in blob                                # FR13.6
    assert "ALIASES" in blob                                  # FR9.3
    assert "-1, NOT NaN" in blob


def test_multimatch_nan_drop_count(test_py_functions):
    """FR13.6 -- the count is recovered from the evaluator's own score_details."""
    n_images, subject_num = 4, 3
    details = np.full((n_images, subject_num, subject_num, 9), -1.0)
    for i in range(n_images):
        for s in range(subject_num):
            details[i, s, s, :5] = [0.9, 0.8, 0.7, 0.6, 0.5]
    details[0, 1, 1, 2] = np.nan            # one dimension NaN -> the whole row drops
    details[2, 0, 0, 4] = np.nan

    out = test_py_functions["multimatch_nan_drops"](details.tolist(), subject_num)
    assert out["n_diagonal_cells"] == n_images * subject_num == 12
    assert out["n_dropped"] == 2
    assert out["per_dimension"]["length"] == 1
    assert out["per_dimension"]["duration"] == 1
    assert out["fraction_dropped"] == pytest.approx(2 / 12)


def test_prediction_key_set_is_verified_against_the_scored_split():
    """FR7.4 -- a missing or duplicated (name, subject) aborts, listing both sides."""
    source = _source()
    assert "prediction.json key set != the scored split's (FR7.4)" in source
    assert "collections.Counter(pred_keys)" in source
    assert "missing = sorted(gt_keys - set(pred_keys))" in source
    assert "extra = sorted(set(pred_keys) - gt_keys)" in source


def test_frozen_file_hashes_are_recorded_at_run_time():
    """D1 / D5 -- the repo test proves the copy; metrics.json proves the machine."""
    source = _source()
    for key in ("evaluation_py", "scanmatch_py", "visual_attention_metrics_py"):
        assert '"{}"'.format(key) in source, key
