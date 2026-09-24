"""The writes a pass leaves behind.

Both passes write a file whole, by rename, so a reader never meets a half-written
one. The split's rows and the manifest that describes them land the same way,
which is what keeps a pass that dies part way through from leaving a partial
file behind for the next one to read. The checksum a manifest records comes from
`core.checksum`.
"""

import json
import os
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows whole, by rename, so a reader never sees a half-written file."""
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp_path, path)
