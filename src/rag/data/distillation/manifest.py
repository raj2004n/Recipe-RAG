"""The writes a pass leaves behind, and the hash of an input it read.

Both passes write a file whole, by rename, so a reader never meets a half-written
one, and both record the checksum of the input they read in the manifest beside
their output. The split's rows and the manifest that describes them land the same
way, which is what keeps a pass that dies part way through from leaving a partial
file behind for the next one to read.
"""

import hashlib
import json
import os
from pathlib import Path


def file_checksum(path: Path) -> str:
    """A file's bytes, so a manifest names the exact input a pass read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows whole, by rename, so a reader never sees a half-written file."""
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp_path, path)
