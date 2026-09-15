"""FR14.4 and the Data Architecture Integrity keying invariants that are greppable.

Each of these, if violated, produces numbers that look entirely reasonable -- which
is why they are asserted rather than assumed.
"""

import os
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVE_PREP = os.path.join(REPO_ROOT, "tools", "eve_prep")
RUN_SCRIPT = os.path.join(REPO_ROOT, "bash", "extract_eve_features.sh")


def _sources(*dirs):
    out = []
    for d in dirs:
        if os.path.isfile(d):
            out.append(d)
            continue
        for root, _, files in os.walk(d):
            if "__pycache__" in root:
                continue
            out.extend(os.path.join(root, f) for f in files
                       if f.endswith((".py", ".sh")))
    return out


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _code_only(path):
    """Source with comments and string literals removed.

    The greps below look for idioms this package must never *execute*. Its docstrings
    name those idioms deliberately -- explaining why ``str.replace('jpg', 'pth')`` is
    banned is the point of the comment -- so a raw text grep would fail on the
    documentation rather than on the code. Tokenising is what makes the check about
    behaviour.
    """
    source = _read(path)
    if not path.endswith(".py"):
        return "\n".join(line.split("#", 1)[0] for line in source.splitlines())

    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    # Docstrings and comments are gone; ordinary string literals are KEPT, so an
    # actual `x.replace('jpg', 'pth')` in the code would still be caught.
    return ast.unparse(ast.fix_missing_locations(tree))


def test_upstream_is_unmodified():
    """D1, convention 2 -- ResNetCOCO is imported, never edited."""
    # ISP/EVE/ is excluded rather than ISP/ narrowed: F5 ADDS that branch (a new
    # dataset branch mirroring OSIE, working convention 3), and F4's claim is that the
    # branches it consumes are unedited. Every existing branch and the whole of
    # SE-Net stay in scope, so an edit to ResNetCOCO or to any frozen file still fails
    # here.
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "ISP", "SE-Net", ":!ISP/EVE"],
            cwd=REPO_ROOT, stderr=subprocess.STDOUT).decode()
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip("git unavailable: {!r}".format(exc))
    dirty = [line for line in out.splitlines() if not line.endswith(".pyc")]
    assert dirty == [], "upstream files modified: {}".format(dirty)


def test_frozen_metric_code_is_not_on_this_path():
    """D1 -- nothing under utils/ is imported, read or reached by F4."""
    for path in _sources(EVE_PREP):
        source = _code_only(path)
        assert "evaluation" not in source
        assert "evaltools" not in source


def test_no_unanchored_jpg_replace_on_the_eve_path():
    """FR5.2, FR10.1 -- paths are built by concatenation, never by replace.

    Asserted on the *call graph*, not on the text: ``exp_key_filename``'s error
    message quotes the banned idiom on purpose, so a text grep would flag the
    documentation that exists to prevent the defect. What must not appear is an
    executed ``<expr>.replace("jpg", ...)``.
    """
    import ast

    offenders = []
    for path in _sources(EVE_PREP):
        if not path.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(_read(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "replace"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "jpg" in node.args[0].value):
                offenders.append("{}:{}".format(path, node.lineno))
    assert offenders == [], "unanchored jpg replace: {}".format(offenders)


def test_the_bridge_per_name_stimuli_are_not_consumed():
    """FR11.4 -- what prevents a later refactor quietly reintroducing OPEN-6."""
    targets = [EVE_PREP] + ([RUN_SCRIPT] if os.path.isfile(RUN_SCRIPT) else [])
    for path in _sources(*targets):
        source = _code_only(path)
        assert "eve_bridge/stimuli" not in source
        assert "eve_bridge\\stimuli" not in source
