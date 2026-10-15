"""Manifest writing, and the DVC pointer that stands beside it.

Every pass in this cluster leaves a manifest beside its output. It goes down
whole, by rename, so a reader never meets it half-written, and it names the
checksum of the input it read so a later pass can tell whether it read the same
bytes. The hash itself belongs to `core.checksum`, which the passes that need it
import directly.

These passes run outside the build graph, so no stage moves the `.dvc` pointer
that stands in for their artefacts in git. `add_to_dvc` is what puts the artefact
and its pointer down in the same run.
"""

import json
import os
import subprocess
from pathlib import Path


def write_manifest(path: Path, document: dict) -> None:
    """Write the manifest whole, by rename, so a reader never sees it half-written."""
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temp_path, path)


def add_to_dvc(path: Path) -> None:
    """Record `path` with DVC, so the run that wrote the artefact moves its pointer.

    Called once the artefact is whole on disk, and before the manifest that calls
    the pass complete, so a manifest reading complete means the pointer beside it
    describes the bytes that are there. A failing add raises rather than leaving a
    pointer that has fallen behind the file.
    """
    subprocess.run(["dvc", "add", str(path)], check=True)
