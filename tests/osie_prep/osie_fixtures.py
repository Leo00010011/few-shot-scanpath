"""Record builders for the COCO-FreeView preflight tests.

Deliberately **not** named ``conftest``: ``tests/eve_bridge/conftest.py`` already
claims that module name, and a bare ``from conftest import ...`` resolves to
whichever of the two pytest loaded first when both suites run together.

Everything is built from hand-made in-memory records written to a tmp path, so the
whole preflight layer is testable on the Windows dev machine without the cluster,
the staged data, or a GPU (plan Step 2, validation groups 1-3).
"""

import json


def make_record(task="bottle", name="000000000001.jpg", subject=0, length=4,
                split="test", x0=10.0, y0=10.0):
    """One well-formed fixations.json record."""
    return {
        "name": name,
        "subject": subject,
        "X": [x0 + i for i in range(length)],
        "Y": [y0 + i for i in range(length)],
        "T": [200 + i for i in range(length)],
        "length": length,
        "split": split,
        "task": task,
        "condition": "freeview",
    }


def make_records(n_images=3, subjects=(0, 1, 2), task="bottle", split="test"):
    """A fully-crossed, well-formed fixture: every image seen by every subject."""
    records = []
    for k in range(n_images):
        name = "{:012d}.jpg".format(k + 1)
        for subject in subjects:
            records.append(make_record(task=task, name=name, subject=subject,
                                       split=split))
    return records


def write_fixations(tmp_path, records, filename="fixations.json"):
    path = tmp_path / filename
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def write_images(tmp_path, records, root_name="images"):
    """Create an empty placeholder file for each (task, name).

    Existence is all check_fixations tests -- it never opens a stimulus.
    """
    root = tmp_path / root_name
    for rec in records:
        d = root / rec["task"]
        d.mkdir(parents=True, exist_ok=True)
        (d / rec["name"]).write_bytes(b"")
    return str(root)
