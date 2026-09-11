"""Validation Group 6 -- frozen-tree and working-convention compliance.

D1 is the constitution's hardest constraint, so the ISP check is made separately and
explicitly: a blanket "SE-Net/ and ISP/ are clean" assertion can pass while hiding it.
"""

import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(REPO, "tools", "eve_senet")
CONFIG = os.path.join(REPO, "SE-Net", "configs", "eve_useremb.json")


def _git(*args):
    return subprocess.check_output(["git"] + list(args), cwd=REPO).decode()


def _sources():
    return [os.path.join(PKG, f) for f in sorted(os.listdir(PKG)) if f.endswith(".py")]


def test_no_tracked_file_under_se_net_or_isp_is_modified():
    """FR10.2 / D1. A NEW file (eve_useremb.json) is expected and allowed; a modified
    tracked one is not."""
    dirty = [line for line in _git("status", "--porcelain", "SE-Net/", "ISP/").splitlines()
             if not line.startswith("??")]
    assert dirty == [], dirty


def test_the_frozen_metric_files_are_untouched():
    """D1, checked on its own rather than folded into the blanket check above."""
    out = _git("diff", "--stat", "--", "ISP/*/GazeformerISP/src/utils/")
    assert out.strip() == ""


def test_the_package_never_imports_the_bridge():
    """FR10.3 / D2 -- F3 reads the bridge's artefacts, never its code.

    Validation Group 6 words this as ``grep -r "eve_bridge" tools/eve_senet/`` returning
    nothing, which cannot hold literally: the requirements' own CLI block defaults every
    input to ``data/eve_bridge/...``, and ``data/eve_bridge`` is an *artefact directory*,
    not the ``tools/eve_bridge`` package. So the check is on the thing the item is for --
    an import edge between the two packages -- and every remaining mention is asserted to
    be a data path or prose.
    """
    importing = re.compile(r"^\s*(?:from|import)\s+\S*eve_bridge", re.M)
    for path in _sources():
        text = open(path).read()
        assert not importing.search(text), path
        for line in text.splitlines():
            if "eve_bridge" in line:
                assert "data/eve_bridge" in line or line.lstrip().startswith(
                    ("#", "*", '"""', "'''")) or '"' in line or "``" in line, (path, line)


def test_the_package_never_names_the_frozen_metric_code():
    """FR10.1 -- F3 does not touch the ISP tree at all."""
    pattern = re.compile(r"\bevaltools\b|\bevaluation\.py\b|utils\.evaluation")
    for path in _sources():
        assert not pattern.search(open(path).read()), path


def test_no_absolute_path_in_any_source_or_in_the_config():
    """Working convention 4 -- cluster paths live in bash/ and CLI args only."""
    absolute = re.compile(r"""['"](?:[A-Za-z]:[\\/]|/(?:home|scratch|mnt|data)/)""")
    for path in _sources() + [CONFIG]:
        text = open(path).read()
        assert not absolute.search(text), path


def test_the_eve_config_carries_no_hardcoded_dataset_path():
    import json
    cfg = json.load(open(CONFIG))
    assert cfg["Data"]["name"] == "EVE"
    assert cfg["Data"]["image_path"] == ""   # set from --image-dir at runtime
    assert cfg["Data"]["fix_path"] == ""     # F3 loads fixations itself
    # deliberately unchanged from OSIE's: these shaped the released checkpoint's
    # weights, and changing any of them silently mismatches under strict=False
    assert cfg["Data"]["im_w"] == 512 and cfg["Data"]["im_h"] == 320
    assert cfg["Data"]["max_traj_length"] == 20 and cfg["Data"]["TAP"] == "FV"
    assert cfg["Data"]["num_subjects"] == 10  # sizes subject_predictor only (FR6.3)
    assert cfg["Model"]["embedding_dim"] == 384


def test_the_eve_config_is_committable():
    """The blanket *.json ignore would have swallowed it -- the exact failure
    TechStack convention 5 records for paper_reference.json."""
    ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", CONFIG],
                             cwd=REPO)
    tracked_or_untracked = _git("status", "--porcelain", "--", CONFIG)
    assert ignored.returncode != 0 or tracked_or_untracked.strip() != "", (
        "SE-Net/configs/eve_useremb.json is invisible to git")


def test_the_output_directory_is_git_ignored():
    """Working convention 5 -- subject-level derived data never reaches the repo, and
    the !spec/**/*.json negation does not apply here."""
    probe = "data/eve_senet/senet_report.json"
    assert subprocess.run(["git", "check-ignore", "-q", "--no-index", probe],
                          cwd=REPO).returncode == 0


def test_msdeformattn_build_artefacts_are_ignored():
    gitignore = open(os.path.join(REPO, ".gitignore")).read()
    for rule in ("*.so", "**/ops/build/", "*.egg-info/"):
        assert rule in gitignore, rule


def test_no_optimiser_is_constructed_anywhere_in_the_package():
    """D8 -- eval only. builder.build()'s AdamW is exactly what we avoid by not
    calling it."""
    for path in _sources():
        text = open(path).read()
        assert "optim" not in text.replace("optimiser", ""), path


@pytest.mark.parametrize("name", ["__init__.py", "check_env.py", "durations.py",
                                  "dataset.py", "embed.py", "verify_embedding.py"])
def test_every_planned_module_exists(name):
    assert os.path.isfile(os.path.join(PKG, name))


def test_the_run_script_exists_and_is_not_sourced():
    path = os.path.join(REPO, "bash", "embed_eve_subjects.sh")
    text = open(path).read()
    assert "set -euo pipefail" in text
    assert "NEVER `source`" in text
    for tunable in ("SENET_ENV", "SEED", "NUM_FEWSHOT", "BRIDGE_DIR", "OUT_DIR",
                    "CKPT", "CONFIG"):
        assert "{}:-".format(tunable) in text, tunable
