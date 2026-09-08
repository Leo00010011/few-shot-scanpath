"""``GtHeatmapStore`` -- HDF5 cache of the ground-truth step heatmaps (FR8).

numpy + h5py only. Row ``i`` of every ``/trials/*`` dataset corresponds to record
``i`` of ``fixations.json`` in the FR5.3 order; ``fixations_sha256`` makes a stale
cache impossible to pair silently with a regenerated JSON.
"""

import datetime
import hashlib

import h5py
import numpy as np

from .heatmaps import build_step_heatmaps

VLEN_STR = h5py.special_dtype(vlen=str)

#: FR8.6 -- refuse to assemble more than this many bytes of heatmaps in RAM.
MAX_HEATMAP_BYTES = 2 * 1024 ** 3


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def trial_key(name, subject):
    return "{}|{}".format(name, int(subject))


def _as_str(v):
    return v.decode() if isinstance(v, bytes) else str(v)


class GtHeatmapStore(object):

    def __init__(self, trial_keys, exp_keys, subjects, lengths, action_masks,
                 heatmaps, attrs):
        self.trial_keys = list(trial_keys)
        self.exp_keys = list(exp_keys)
        self.subjects = np.asarray(subjects, dtype=np.int32)
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.action_masks = np.asarray(action_masks, dtype=np.float32)
        self.heatmaps = np.asarray(heatmaps, dtype=np.float32)
        self.attrs = dict(attrs)

        self._key_to_row = {k: i for i, k in enumerate(self.trial_keys)}
        self._exp_to_key = {e: self.trial_keys[i] for i, e in enumerate(self.exp_keys)}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, fixations, exp_keys, subject_id_map, attrs, **hm_kwargs):
        if len(fixations) != len(exp_keys):
            raise ValueError(
                "fixations ({}) and exp_keys ({}) differ in length".format(
                    len(fixations), len(exp_keys)))

        max_length = hm_kwargs.get("max_length", 16)
        action_map = hm_kwargs.get("action_map", (24, 32))
        n = len(fixations)
        n_bytes = n * max_length * action_map[0] * action_map[1] * 4
        if n_bytes > MAX_HEATMAP_BYTES:
            raise ValueError(
                "refusing to assemble {} bytes of heatmaps in RAM (limit {}); a "
                "dataset this large is out of the scope of this feature "
                "(FR8.6)".format(n_bytes, MAX_HEATMAP_BYTES))

        keys = []
        subjects = np.zeros(n, dtype=np.int32)
        lengths = np.zeros(n, dtype=np.int32)
        action_masks = np.zeros((n, max_length), dtype=np.float32)
        heatmaps = np.zeros((n, max_length, action_map[0], action_map[1]),
                            dtype=np.float32)

        for i, rec in enumerate(fixations):
            hm, mask = build_step_heatmaps(rec["X"], rec["Y"], rec["length"],
                                           **hm_kwargs)
            heatmaps[i] = hm
            action_masks[i] = mask
            subjects[i] = int(rec["subject"])
            lengths[i] = min(int(rec["length"]), max_length)
            keys.append(trial_key(rec["name"], rec["subject"]))

        to_dense = subject_id_map["to_dense"]
        dense_sorted = sorted(to_dense.values())
        attrs = dict(attrs)
        attrs.setdefault("origin_size", hm_kwargs.get("origin_size", (1080, 1920)))
        attrs.setdefault("action_map", action_map)
        attrs.setdefault("max_length", max_length)
        attrs.setdefault("blur_sigma", hm_kwargs.get("blur_sigma", 1))
        attrs["n_trials"] = n
        attrs["subject_ids_dense"] = dense_sorted
        attrs["subject_ids_eve"] = [subject_id_map["to_eve"][str(d)]
                                    for d in dense_sorted]

        return cls(keys, list(exp_keys), subjects, lengths, action_masks,
                   heatmaps, attrs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        max_length = self.heatmaps.shape[1]
        action_map = self.heatmaps.shape[2:]
        with h5py.File(path, "w") as f:
            f.attrs["created_utc"] = self.attrs.get(
                "created_utc",
                datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"))
            f.attrs["bundle_dir"] = str(self.attrs.get("bundle_dir", ""))
            f.attrs["origin_size"] = np.asarray(self.attrs["origin_size"],
                                                dtype=np.int32)
            f.attrs["action_map"] = np.asarray(self.attrs["action_map"],
                                               dtype=np.int32)
            f.attrs["max_length"] = np.int32(self.attrs["max_length"])
            f.attrs["blur_sigma"] = np.float32(self.attrs["blur_sigma"])
            f.attrs["fixations_sha256"] = str(self.attrs.get("fixations_sha256", ""))
            f.attrs["n_trials"] = np.int32(self.attrs["n_trials"])
            f.attrs["subject_ids_dense"] = np.asarray(
                self.attrs["subject_ids_dense"], dtype=np.int32)
            f.attrs.create("subject_ids_eve",
                           np.array(self.attrs["subject_ids_eve"], dtype=object),
                           dtype=VLEN_STR)

            grp = f.create_group("trials")
            grp.create_dataset("trial_key",
                               data=np.array(self.trial_keys, dtype=object),
                               dtype=VLEN_STR)
            grp.create_dataset("exp_key",
                               data=np.array(self.exp_keys, dtype=object),
                               dtype=VLEN_STR)
            grp.create_dataset("subject", data=self.subjects)
            grp.create_dataset("length", data=self.lengths)
            grp.create_dataset("action_mask", data=self.action_masks)
            grp.create_dataset("heatmaps", data=self.heatmaps,
                               compression="gzip", compression_opts=4,
                               chunks=(1, max_length, action_map[0], action_map[1]))

    @classmethod
    def load(cls, path, fixations_path=None):
        with h5py.File(path, "r") as f:
            attrs = {
                "created_utc": _as_str(f.attrs["created_utc"]),
                "bundle_dir": _as_str(f.attrs["bundle_dir"]),
                "origin_size": [int(v) for v in f.attrs["origin_size"]],
                "action_map": [int(v) for v in f.attrs["action_map"]],
                "max_length": int(f.attrs["max_length"]),
                "blur_sigma": float(f.attrs["blur_sigma"]),
                "fixations_sha256": _as_str(f.attrs["fixations_sha256"]),
                "n_trials": int(f.attrs["n_trials"]),
                "subject_ids_dense": [int(v) for v in f.attrs["subject_ids_dense"]],
                "subject_ids_eve": [_as_str(v) for v in f.attrs["subject_ids_eve"]],
            }
            grp = f["trials"]
            store = cls(
                [_as_str(v) for v in grp["trial_key"][:]],
                [_as_str(v) for v in grp["exp_key"][:]],
                grp["subject"][:],
                grp["length"][:],
                grp["action_mask"][:],
                grp["heatmaps"][:],
                attrs,
            )

        if fixations_path is not None:
            actual = sha256_file(fixations_path)
            if actual != attrs["fixations_sha256"]:
                raise ValueError(
                    "fixations.json hash mismatch: the store was built from "
                    "{} but {} hashes to {}".format(
                        attrs["fixations_sha256"], fixations_path, actual))
        return store

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _row(self, key):
        if key not in self._key_to_row:
            raise KeyError("trial key {!r} is not in the heatmap store".format(key))
        return self._key_to_row[key]

    def get(self, name, subject):
        return self.heatmaps[self._row(trial_key(name, subject))]

    def get_batch(self, keys):
        rows = [self._row(k) for k in keys]   # raises before any array is touched
        return self.heatmaps[rows]

    def length_of(self, name, subject):
        return int(self.lengths[self._row(trial_key(name, subject))])

    def exp_key_of(self, name, subject):
        return self.exp_keys[self._row(trial_key(name, subject))]

    def trial_key_of(self, exp_key):
        if exp_key not in self._exp_to_key:
            raise KeyError("exp_key {!r} is not in the heatmap store".format(exp_key))
        return self._exp_to_key[exp_key]
