"""Record builders for the OSIE preflight tests.

Deliberately **not** named ``conftest``: ``tests/eve_bridge/conftest.py`` already
claims that module name, and a bare ``from conftest import ...`` resolves to
whichever of the two pytest loaded first when both suites run together.

Everything is built from hand-made in-memory records written to a tmp path, so the
whole preflight layer is testable on the Windows dev machine without the cluster,
the staged data, or a GPU (plan Step 2, validation groups 1-3).

Shapes mirror OSIE's shipped ``fixations.json``: a **flat** stimulus name with no
category component, ``task="none"``, ``condition="freeview"``, and coordinates in
the native 800x600 frame.
"""

import json


def make_record(name="1001.jpg", subject=10, length=4, split="test",
                x0=10.0, y0=10.0):
    """One well-formed fixations.json record."""
    return {
        "name": name,
        "subject": subject,
        "X": [x0 + i for i in range(length)],
        "Y": [y0 + i for i in range(length)],
        "T": [200 + i for i in range(length)],
        "length": length,
        "split": split,
        "task": "none",
        "condition": "freeview",
    }


def make_records(n_images=3, subjects=(10, 11, 12, 13, 14), split="test"):
    """A fully-crossed, well-formed fixture: every image seen by every subject."""
    records = []
    for k in range(n_images):
        name = "{}.jpg".format(1001 + k)
        for subject in subjects:
            records.append(make_record(name=name, subject=subject, split=split))
    return records


def write_fixations(tmp_path, records, filename="fixations.json"):
    path = tmp_path / filename
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def write_images(tmp_path, records, root_name="images"):
    """Create an empty placeholder file for each stimulus name.

    OSIE's stimulus directory is flat -- no ``<category>/`` component. Existence is
    all check_fixations tests; it never opens a stimulus.
    """
    root = tmp_path / root_name
    root.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (root / rec["name"]).write_bytes(b"")
    return str(root)
