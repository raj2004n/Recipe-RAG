"""One hash of a file's bytes, for the manifests and the registry that name an input.

A pass records the checksum of every file it read, so a later one can tell whether it read the same
bytes, and the dataset registry records the same thing for the corpus a build was made against. They
all want one value in one format, ``sha256:`` followed by the hex digest, so they share a single
implementation rather than carrying one apiece.
"""

import hashlib
from pathlib import Path


def file_checksum(path: Path) -> str:
    """A file's bytes, so a manifest names the exact input a pass read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()
