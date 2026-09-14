"""The ``(name, subject) -> exp_key`` mapping, and the filename rule (FR2, FR5.2).

Pure ``h5py`` + ``numpy`` + stdlib -- **no torch**, so this module imports on the
Windows dev machine and is what F5 reuses at dataset construction (FR10.3).

``gt_heatmaps.h5`` is the *authority* for the mapping: it was written by the same code
path that wrote ``fixations.json`` and is hash-bound to it through the
``fixations_sha256`` root attr. The bundle's ``samples_df`` gives an *independent*
derivation, which is the **check** (FR2.4) -- this feature's D4 guard. A trial mapped
to the wrong exp_key hands a participant another participant's screen and every
downstream metric still looks entirely plausible.

``GtHeatmapStore.load()`` is deliberately not used: it materialises the full
``(1804, 16, 24, 32)`` heatmap array (~88 MB) that F4 has no use for. Two vlen string
datasets are read directly instead.
"""

import hashlib
import json
import os
import re
import sys

import h5py

try:
    from . import EvePrepError
except ImportError:  # executed as a script, not as a package member
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eve_prep import EvePrepError

TRIAL_KEY_SEP = "|"

#: FR5.2 -- the realised cohort's 1804 exp_keys all match this and none contains
#: "jpg". Re-asserted on every run rather than trusted (FR7.4).
EXP_KEY_RE = re.compile(r"[A-Za-z0-9_]+\Z")

_MAX_LISTED = 10


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _as_str(v):
    """h5py returns bytes for vlen str on some versions and str on others."""
    return v.decode() if isinstance(v, bytes) else str(v)


def _listing(items):
    shown = [str(_) for _ in items[:_MAX_LISTED]]
    text = ", ".join(shown)
    if len(items) > len(shown):
        text += " ... (+{} more)".format(len(items) - len(shown))
    return "{} total: {}".format(len(items), text)


def parse_trial_key(trial_key):
    """``"<name>|<subject>"`` -> ``(name, int(subject))`` (FR2.2).

    Splits on the **last** separator: a stimulus name containing the separator must
    not be able to corrupt the subject id.
    """
    name, sep, subject = str(trial_key).rpartition(TRIAL_KEY_SEP)
    if not sep or not name or not subject.isdigit():
        raise EvePrepError(
            "malformed trial key {!r}: expected '<name>{}<subject>' (FR2.2)".format(
                trial_key, TRIAL_KEY_SEP))
    return name, int(subject)


def load_trial_exp_keys(heatmaps_path, fixations_path=None):
    """The authoritative mapping ``{(name, subject): exp_key}`` (FR2.1, FR2.3, FR2.5).

    Reads ``trials/trial_key`` and ``trials/exp_key`` only. When ``fixations_path``
    is given, the store's ``fixations_sha256`` attr must match that file -- otherwise
    the mapping and the records describe two different builds.
    """
    with h5py.File(heatmaps_path, "r") as f:
        stored_sha = _as_str(f.attrs["fixations_sha256"])
        grp = f["trials"]
        trial_keys = [_as_str(v) for v in grp["trial_key"][:]]
        exp_keys = [_as_str(v) for v in grp["exp_key"][:]]
        # NOTE: grp["heatmaps"] is never touched -- ~88 MB we have no use for.

    if len(trial_keys) != len(exp_keys):
        raise EvePrepError(
            "{}: trials/trial_key has {} rows but trials/exp_key has {} (FR2.1)".format(
                heatmaps_path, len(trial_keys), len(exp_keys)))

    if len(set(exp_keys)) != len(exp_keys):
        seen, dup = set(), []
        for e in exp_keys:
            if e in seen and e not in dup:
                dup.append(e)
            seen.add(e)
        raise EvePrepError(
            "{}: duplicate exp_key -- two trials would share one screen capture, "
            "which contradicts bridge_report.json's duplicate_trial = 0 (FR2.3); "
            "{}".format(heatmaps_path, _listing(dup)))

    if fixations_path is not None:
        actual = sha256_file(fixations_path)
        if actual != stored_sha:
            raise EvePrepError(
                "fixations.json hash mismatch (FR2.5): the store at {} was built "
                "from {} but {} hashes to {}".format(
                    heatmaps_path, stored_sha, fixations_path, actual))

    mapping = {}
    for t, e in zip(trial_keys, exp_keys):
        key = parse_trial_key(t)
        if key in mapping:
            raise EvePrepError(
                "{}: duplicate trial key {!r} (FR2.3)".format(heatmaps_path, t))
        mapping[key] = e
    return mapping


def derive_trial_exp_keys(samples_df, fixations, subject_id_map):
    """Independently re-derive the mapping from the bundle's ``samples_df`` (FR2.4).

    For each ``fixations.json`` record, the unique ``samples_df`` row with
    ``stimulus_name == name[:-4]`` and ``subject == to_eve[str(subject)]``. Zero or
    several matching rows raise, naming the count -- the ambiguity is the finding.
    """
    to_eve = subject_id_map["to_eve"]
    index = {}
    for row in samples_df.itertuples(index=False):
        index.setdefault((str(row.stimulus_name), str(row.subject)), []).append(
            str(row.exp_key))

    out = {}
    for rec in fixations:
        name = rec["name"]
        dense = int(rec["subject"])
        stem = name[:-4] if name.endswith(".jpg") else name
        if str(dense) not in to_eve:
            raise EvePrepError(
                "dense subject {} is absent from subject_id_map['to_eve'] "
                "(FR2.4)".format(dense))
        eve = to_eve[str(dense)]
        keys = index.get((stem, eve), [])
        if len(keys) != 1:
            raise EvePrepError(
                "{} rows in samples_df for stimulus {!r} subject {!r} "
                "(dense {}), expected exactly 1 (FR2.4)".format(
                    len(keys), stem, eve, dense))
        out[(name, dense)] = keys[0]
    return out


def crosscheck_exp_keys(authoritative, derived):
    """Compare the two derivations trial by trial (FR2.4) -- this feature's D4 gate.

    Three distinguishable failures, because they mean three different upstream
    problems: a trial only the store knows, a trial only the derivation knows, and a
    trial the two map to different captures.
    """
    only_store = sorted(set(authoritative) - set(derived))
    if only_store:
        raise EvePrepError(
            "exp_key cross-check failed (FR2.4): trials present in gt_heatmaps.h5 "
            "but not derivable from samples_df; {}".format(_listing(only_store)))

    only_derived = sorted(set(derived) - set(authoritative))
    if only_derived:
        raise EvePrepError(
            "exp_key cross-check failed (FR2.4): trials derived from samples_df "
            "but absent from gt_heatmaps.h5; {}".format(_listing(only_derived)))

    disagree = ["{} store={} derived={}".format(k, authoritative[k], derived[k])
                for k in sorted(authoritative) if authoritative[k] != derived[k]]
    if disagree:
        raise EvePrepError(
            "exp_key cross-check failed (FR2.4): the store and samples_df disagree "
            "on which capture a trial used -- a participant would be scored against "
            "another participant's screen (D4); {}".format(_listing(disagree)))

    return {"source": "gt_heatmaps.h5", "derived_from": "samples_df",
            "agree": True, "n": len(authoritative)}


def exp_key_filename(exp_key):
    """``exp_key + '.pth'``, after asserting the charset and the no-jpg rule.

    FR5.2, FR7.4. The jpg check is not decorative: the OSIE loader builds its feature
    path with an **unanchored** ``str.replace('jpg', 'pth')`` (TechStack section 3.2),
    so a key containing that substring anywhere would be silently corrupted by any
    call site that reuses the upstream idiom.
    """
    key = str(exp_key)
    if not EXP_KEY_RE.match(key):
        raise EvePrepError(
            "exp_key {!r} is outside [A-Za-z0-9_]+ and is not safe as a filename "
            "(FR5.2)".format(key))
    if "jpg" in key:
        raise EvePrepError(
            "exp_key {!r} contains 'jpg'; the OSIE loader's unanchored "
            "str.replace('jpg', 'pth') would corrupt it (FR5.2, TechStack 3.2)".format(
                key))
    return key + ".pth"


def split_of(fixations):
    """``{(name, subject): split}`` -- one derivation, used by extractor and checker.

    A duplicate ``(name, subject)`` raises: ``fixations.json`` records are addressed
    positionally by every other artefact (working convention 10), so two records for
    one trial would make the index space ambiguous.
    """
    out = {}
    for rec in fixations:
        key = (rec["name"], int(rec["subject"]))
        if key in out:
            raise EvePrepError(
                "duplicate trial {} in fixations.json (FR2.3)".format(key))
        out[key] = rec["split"]
    return out


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
