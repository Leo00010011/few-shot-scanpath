"""Group 8 -- the static half of ``bash/test_eve.sh``.

The run itself needs the cluster; these are the parts that are greppable and would
otherwise only be discovered by a failed allocation.
"""

import os
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = os.path.join(REPO_ROOT, "bash", "test_eve.sh")


@pytest.fixture(scope="module")
def script():
    with open(SCRIPT, encoding="utf-8") as fh:
        return fh.read()


def _code(script):
    """The script with comment lines removed.

    Several comments deliberately name the very things these checks forbid --
    `i_batch > 100`, `conda info --base` -- because explaining why they are absent is
    the point of the comment.
    """
    return "\n".join(line for line in script.splitlines()
                     if not line.strip().startswith("#"))


def test_script_is_syntactically_valid():
    try:
        proc = subprocess.run(["bash", "-n", SCRIPT], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    except OSError as exc:
        pytest.skip("bash unavailable: {!r}".format(exc))
    out = proc.stdout.decode(errors="replace")
    if proc.returncode and ("No such file" in out or "cannot open" in out):
        # A WSL bash on PATH cannot read a Windows path; that is not a syntax result.
        pytest.skip("this bash cannot read the script path: {}".format(out.strip()))
    assert proc.returncode == 0, out


def test_every_tunable_is_overridable(script):
    """FR12.2 -- and working convention 4: no absolute path is hardcoded in a .py."""
    for name in ["HOME_DIR", "PROJECT_DIR", "ISP_ENV", "BRIDGE_DIR", "SENET_DIR",
                 "FEATURE_DIR", "WEIGHTS_DIR", "LOCAL_SCRATCH", "STAGE_FEATURES",
                 "SEED", "SUBJECT_NUM", "NUM_FEWSHOT", "N_SUBJECTS", "MAX_BATCHES",
                 "MOUNT_IMAGE", "USER_EMB", "EVE_WORK", "FAST_PREFLIGHT"]:
        assert re.search(r'{0}="\$\{{{0}:-'.format(name), script), name


def test_defaults_live_under_beegfs_and_never_write_to_tmp(script):
    """TechStack section 1.3: /mnt/imagenes is invisible to compute nodes, and /tmp
    is node-local -- fine as a staging area for READ data, ruinous for anything that
    must persist."""
    code = _code(script)
    assert "/mnt/imagenes" not in code
    assert 'HOME_DIR="${HOME_DIR:-/mnt/beegfs/home/' in code
    # The only /tmp default is LOCAL_SCRATCH, and it is only ever read from.
    tmp_lines = [l for l in code.splitlines() if "/tmp" in l]
    assert tmp_lines == [l for l in tmp_lines if "LOCAL_SCRATCH" in l], tmp_lines
    assert 'FEATURE_SRC="$STAGED"' in code


def test_conda_is_sourced_by_explicit_path(script):
    """FR12.3 -- before any env is active, conda need not be on PATH at all."""
    assert "conda info --base" not in _code(script)
    assert 'source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"' in script


def test_set_u_is_lifted_only_across_conda_activate(script):
    """FR12.3 -- TechStack section 1.3's sixth cluster fact."""
    assert "set -euo pipefail" in script
    code = [line for line in script.splitlines()
            if line.strip() and not line.strip().startswith("#")]
    lifted = next(i for i, l in enumerate(code) if l.strip() == "set +u")
    activate = next(i for i, l in enumerate(code)
                    if l.strip().startswith("conda activate"))
    restored = next(i for i, l in enumerate(code)
                    if i > lifted and l.strip() == "set -u")
    assert lifted < activate < restored
    assert "set +e" not in script
    assert "set +o pipefail" not in script


def test_fewshot_subjects_are_generated_not_typed(script):
    """FR12.4 (D4) -- the one that is unrecoverable after the fact."""
    code = _code(script)
    assert 'FEWSHOT_SUBJECTS="$(seq 0 $((N_SUBJECTS - 1)) | tr \'\\n\' \' \')"' in code
    assert "fewshot_subject is not ascending" in script
    # Unquoted at the call site: argparse nargs='+' needs 38 separate words.
    assert "--fewshot_subject $FEWSHOT_SUBJECTS" in code


def test_the_ascending_assertion_rejects_a_permuted_list(tmp_path):
    """Run the embedded python block directly, with a permuted list."""
    checker = (
        "import sys\n"
        "xs = [int(v) for v in sys.argv[1].split()]\n"
        "n = int(sys.argv[2])\n"
        "assert xs == list(range(n)), 'not ascending'\n")
    path = tmp_path / "check.py"
    path.write_text(checker, encoding="utf-8")

    ok = subprocess.run([sys.executable, str(path), "0 1 2 3", "4"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ok.returncode == 0

    for bad in ("3 2 1 0", "1 2 3 0", "0 1 2"):
        proc = subprocess.run([sys.executable, str(path), bad, "4"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.returncode != 0, bad


def test_preconditions_fail_before_any_gpu_work(script):
    """FR12.5 -- and the message says why a fresh checkout has no data/."""
    code = _code(script)
    precondition = code.index('for path in "$BRIDGE_DIR/fixations.json"')
    invocation = code.index("python src/test.py")
    assert precondition < invocation
    assert "data/ is git-ignored" in script
    assert code.index("check_eval.py") < invocation


def test_the_cap_and_the_scales_are_passed_explicitly(script):
    """The three silent-corruption traps, all named on the command line."""
    code = _code(script)
    assert '--max_batches "$MAX_BATCHES"' in code
    assert 'MAX_BATCHES="${MAX_BATCHES:--1}"' in code
    assert "--origin_width 1920 --origin_height 1080" in code
    assert "--width 512 --height 384" in code
    assert "--eval_repeat_num 1" in code


def test_per_seed_artefacts_are_preserved_with_the_seed_guard(script):
    """FR10.2 / FR10.3 -- six files per seed, and the copy asserts the seed."""
    code = _code(script)
    for artefact in ("log_test_subject_*.txt", "prediction.json", "metrics.json",
                     "versions.txt", "preflight.json"):
        assert artefact in code, artefact
    assert 'tee "$SEED_DIR/stdout.txt"' in code
    assert "seed mismatch" in script


def test_no_senet_dependency(script):
    """FR1.1 -- F5 runs in the ISP-side env; F3's embedding is consumed as a file."""
    code = _code(script)
    for forbidden in ("module load CUDA", "detectron2", "MSDeformAttn", "FORCE_CUDA"):
        assert forbidden not in code, forbidden
