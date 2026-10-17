import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from .paths import SnapshotPaths
from . import registry

from .types import BM25IndexedChunkSet, IndexedChunkSet
from .indices import load_bm25_index_chunk_sets, load_indexed_chunk_sets


@dataclass
class Snapshot:
    created_at: str | None
    retrieval_strategy: str
    indexed_chunk_sets: list[IndexedChunkSet]  # carries chunks in IndexedChunkSet.chunks
    bm25_indexed_chunk_sets: list[BM25IndexedChunkSet]  # carries chunks in BM25IndexedChunkSet.chunks
    base_config: dict[str, any]  # configs for initialising retriever components for inference, in pipeline
    prompts_config: dict[str, any]
    dataset_key: str  # registry key for the dataset this snapshot was built against


# Strategy -> ordered list of index loaders. Each loader returns
# (chunk_sets, chunks_by_set_name). Adding a strategy = add a row.
_STRATEGY_LOADERS = {
    "dense":  [load_indexed_chunk_sets],
    "hybrid": [load_indexed_chunk_sets, load_bm25_index_chunk_sets],
}


def _resolve_snapshot_dir(folder: str | Path) -> Path:
    """Return the absolute path to a snapshot folder, raising if it is missing.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Snapshot folder not found at {folder_path}")
    return folder_path


def _enforce_complete(registry_file: Path, folder: str) -> None:
    """Refuse to load a snapshot whose registry status is not `"complete"`.
    """
    status = registry.get_status_for_folder(registry_file, folder)
    if status != "complete":
        raise RuntimeError(
            f"Cannot load snapshot {folder!r}: registry status is {status!r}, "
            f"expected 'complete'. Run finalise_build to seal the snapshot."
        )


def load_snapshot(folder: str | Path) -> Snapshot:
    """Load a snapshot directly from a folder path.

    Looks up the folder's name in the registry and refuses to load if the
    status is not `"complete"`. Use `load_snapshot_by_target` to resolve a
    target name (e.g. `"latest_build"`) to a folder.
    """
    folder_path = _resolve_snapshot_dir(folder)
    registry_file = SnapshotPaths.from_paths_config({}).registry_file
    _enforce_complete(registry_file, folder_path.name)

    with open(folder_path / "configs" / "base_config.yaml") as f:
        base_config = yaml.safe_load(f)
    with open(folder_path / "configs" / "prompts_config.yaml") as f:
        prompts_config = yaml.safe_load(f)

    with open(folder_path / "metadata.json") as f:
        metadata = json.load(f)

    strategy = metadata["strategy"]
    try:
        loaders = _STRATEGY_LOADERS[strategy]
    except KeyError as e:
        raise ValueError(
            f"Unknown retrieval strategy {strategy!r}; "
            f"expected one of {sorted(_STRATEGY_LOADERS)}"
        ) from e

    indexed_chunk_sets: list[IndexedChunkSet] = []
    bm25_indexed_chunk_sets: list[BM25IndexedChunkSet] = []

    for loader in loaders:
        chunk_sets, chunks_by_set_name = loader(
            folder_path / "index",
            chunks_save_path=folder_path / "chunks",
        )
        if loader is load_bm25_index_chunk_sets:
            bm25_indexed_chunk_sets = chunk_sets
            for indexed in bm25_indexed_chunk_sets:
                indexed.chunks = chunks_by_set_name.get(indexed.name, {})
        else:
            indexed_chunk_sets = chunk_sets
            for indexed in indexed_chunk_sets:
                indexed.chunks = chunks_by_set_name.get(indexed.name, {})

    return Snapshot(
        created_at=metadata.get("created_at"),
        retrieval_strategy=strategy,
        indexed_chunk_sets=indexed_chunk_sets,
        bm25_indexed_chunk_sets=bm25_indexed_chunk_sets,
        base_config=base_config,
        prompts_config=prompts_config,
        dataset_key=metadata["dataset_key"],
    )


def load_snapshot_by_target(target: str = "latest_build") -> Snapshot:
    """Resolve a target name (e.g. `'latest_build'`) to a snapshot folder and load it.

    Raises:
        FileNotFoundError: If the target has no folder mapped in the registry.
        RuntimeError: If the resolved snapshot's status is not `"complete"`.
    """
    paths = SnapshotPaths.from_paths_config({})
    folder = registry.get_folder_for_target(paths.registry_file, target)
    if not folder:
        raise FileNotFoundError(
            f"No snapshot folder mapped to target {target!r} in registry"
        )

    folder_path = paths.snapshots_root / folder
    return load_snapshot(folder_path)
