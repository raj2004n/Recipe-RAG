"""Manifest writing: one atomic write, and one hash of the input a pass read.

Every pass in this cluster leaves a manifest beside its output. It goes down
whole, by rename, so a reader never meets it half-written, and it names the
checksum of the input it read so a later pass can tell whether it read the
same bytes. Both helpers are small enough to state plainly, and both are used
by passes that live outside this package as well.
"""

import hashlib
import json
import os
from pathlib import Path


def write_manifest(path: Path, document: dict) -> None:
    """Write the manifest whole, by rename, so a reader never sees it half-written."""
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temp_path, path)


def file_checksum(path: Path) -> str:
    """A file's bytes, so a manifest names the exact input a pass read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()
