"""Validation Group 7's static checks on ``bash/extract_eve_features.sh``.

The run itself needs the cluster; these are the parts that are greppable and that
would otherwise only be discovered by a failed allocation.
"""

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "bash", "extract_eve_features.sh")


@pytest.fixture(scope="module")
def script():
    with open(SCRIPT, encoding="utf-8") as fh:
        return fh.read()


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


def test_set_u_is_lifted_only_across_conda_activate(script):
    """FR9.3 -- TechStack section 1.3's fourth cluster fact."""
    assert "set -euo pipefail" in script
    # Comments are stripped first: the block above the activation explains the hazard
    # and names `conda activate`, so a raw search would match the prose, not the code.
    code = [line for line in script.splitlines()
            if line.strip() and not line.strip().startswith("#")]
    lifted = next(i for i, line in enumerate(code) if line.strip() == "set +u")
    activate = next(i for i, line in enumerate(code)
                    if line.strip().startswith("conda activate"))
    restored = next(i for i, line in enumerate(code)
                    if i > lifted and line.strip() == "set -u")
    assert lifted < activate < restored

    # -e and -o pipefail are never lifted: a genuine activation failure must stop.
    assert "set +e" not in script
    assert "set +o pipefail" not in script


def test_every_tunable_is_overridable(script):
    """FR9.2 -- and working convention 4: no absolute path is hardcoded."""
    for name in ["HOME_DIR", "PROJECT_DIR", "ISP_ENV", "BUNDLE_DIR", "BRIDGE_DIR",
                 "OUT_DIR", "SPLIT", "FORCE_FEATURES", "OSIE_EMB", "LOCAL_SCRATCH",
                 "STAGE_BUNDLE", "BUNDLE_TAR", "MOUNT_IMAGE"]:
        assert re.search(r'{0}="\$\{{{0}:-'.format(name), script), name


def _code(script):
    """The script with comment lines removed.

    Several comments deliberately name the very things these checks forbid --
    `face_crops`, `conda info --base` -- because explaining why they are absent is
    the point of the comment. Stripping them keeps the checks about what executes.
    """
    return "\n".join(line for line in script.splitlines()
                     if not line.strip().startswith("#"))


def _command_lines(script):
    """Non-comment, non-``echo`` lines: the ones that actually do something.

    The `echo`s matter too -- the mount block's message says "conda activate is the
    real check", and the staging block's says "face_crops/ is 11 GB F4 never opens".
    Both are the right thing to tell a reader and the wrong thing to grep, so
    ordering and absence are asserted over commands only.
    """
    return [line for line in _code(script).splitlines()
            if line.strip() and not line.strip().startswith("echo ")]


def _line_index(lines, prefix):
    return next(i for i, line in enumerate(lines) if line.strip().startswith(prefix))


def test_preconditions_precede_activation(script):
    """FR9.4 -- fail fast and legibly, before the env is touched where possible.

    Bundle staging needs no conda, so it too sits above the activation and its
    failures stay legible.
    """
    lines = _command_lines(script)
    activate = _line_index(lines, "conda activate")
    assert any("bundle.h5" in l for l in lines[:activate])
    assert any("df -Pk" in l for l in lines[:activate])
    assert any("tar -xf" in l for l in lines[:activate])
    for path in ["fixations.json", "gt_heatmaps.h5", "subject_id_map.json",
                 "bridge_report.json", "stimuli"]:
        assert path in script


def test_bundle_is_staged_to_node_local_scratch(script):
    """The EyeNet-Pipeline convention: keep bulk read I/O off the network filesystem.

    F4 does 1804 random reads of ~1.3 MB PNGs. `whole_train.sh` rsyncs a tar to
    $LOCAL_SCRATCH and points the run at the extracted copy for exactly this reason.
    """
    code = _code(script)
    assert 'LOCAL_SCRATCH/data/bundle' in code
    assert "rsync" in code and "tar -xf" in code
    # Everything WRITTEN still goes to beegfs -- scratch is node-local and vanishes.
    assert 'tee "$OUT_DIR/stdout.txt"' in code
    assert '$LOCAL_SCRATCH' not in code.split('OUT_DIR="${OUT_DIR:-')[1].split("}")[0]


def test_face_crops_are_never_staged(script):
    """F4 needs bundle.h5 + stimuli/ (4.2 GB), not face_crops/ (11 GB).

    `get_stimulus()` resolves samples_df's "stimuli/<exp_key>.png" against the bundle
    dir; nothing on this path opens a face crop. Staging EyeNet's own tar would move
    11 GB of data F4 never reads.
    """
    assert not [l for l in _command_lines(script) if "face_crops" in l]


def test_env_image_is_mounted_before_activation(script):
    """bash/test_osie.sh's convention: `scanpath` lives inside my_env.ext4.

    Non-fatal, because a second mount in one allocation exits non-zero and `set -e`
    would abort before any precondition ran; `conda activate` is the real check.
    """
    lines = _command_lines(script)
    mount = next(i for i, l in enumerate(lines) if "sudo mount_image.py" in l)
    assert mount < _line_index(lines, "conda activate")
    assert "||" in lines[mount] or "||" in lines[mount + 1], (
        "the mount must be non-fatal -- a second mount in one allocation exits "
        "non-zero and set -e would abort before any precondition ran")


def test_conda_sh_path_is_explicit(script):
    """`conda info --base` cannot run before conda is on PATH.

    test_osie.sh and EyeNet-Pipeline/whole_train.sh both spell the path out.
    """
    lines = _command_lines(script)
    assert any('source "$HOME_DIR/miniconda3/etc/profile.d/conda.sh"' in l
               for l in lines)
    assert not [l for l in lines if "conda info --base" in l]


def test_extraction_is_guarded_by_the_preflight_exit_code(script):
    """FR9.6 -- and the post-check whose failure fails the job."""
    assert "check_features.py" in script
    assert script.count("check_features.py") >= 2
    assert script.index("extract_features.py") < script.rindex("check_features.py")


def test_stdout_is_teed_and_versions_written(script):
    """FR9.5, FR9.7 -- the interactive path has no SLURM log."""
    assert 'tee "$OUT_DIR/stdout.txt"' in script
    assert "versions.txt" in script


def test_no_module_load_cuda(script):
    """FR1.1 -- F4 compiles nothing and inherits none of F3's detectron2 machinery."""
    code = "\n".join(line for line in script.splitlines()
                     if not line.strip().startswith("#"))
    assert "module load" not in code
    assert "detectron2" not in code
    assert "MSDeformAttn" not in code


def test_pythonpath_points_at_the_osie_branch(script):
    """FR4.1 -- ResNetCOCO is imported from upstream, not vendored."""
    assert "ISP/OSIE/GazeformerISP/src" in script
    assert "PYTHONPATH" in script
