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
