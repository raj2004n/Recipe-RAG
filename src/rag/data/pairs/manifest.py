"""Manifest writing: one atomic write.

Every pass in this cluster leaves a manifest beside its output. It goes down
whole, by rename, so a reader never meets it half-written, and it names the
checksum of the input it read so a later pass can tell whether it read the same
bytes. The hash itself belongs to `core.checksum`, which the passes that need it
import directly.
"""

import json
import os
from pathlib import Path


def write_manifest(path: Path, document: dict) -> None:
    """Write the manifest whole, by rename, so a reader never sees it half-written."""
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temp_path, path)
