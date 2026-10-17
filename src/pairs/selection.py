import json
from pathlib import Path

from ..snapshot.datasets.resolver import LocalDatasetResolver
from ..snapshot import registry
from ..snapshot.paths import SnapshotPaths
from ..corpus.preprocessor import Recipe


def load_pairs(path: Path) -> list[dict]:
    """Every pair row, keyed once, in the order the file holds it.

    A repeated (primary, partner) pair would otherwise be designed twice and pay twice, so the
    first row wins and the repeat is reported by the manifest rather than silently dropped.
    """
    rows, seen = [], set()
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["primary_id"], row["partner_id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def dataset_key_of(pairs: list[dict]) -> str:
    """The dataset key the pairs were chosen against, which the rows all carry."""
    keys = {row.get("dataset_key") for row in pairs}
    if len(keys) != 1 or None in keys:
        raise ValueError(f"the pair rows do not agree on one dataset key: {sorted(map(str, keys))}")
    return keys.pop()


def resolve_corpus(dataset_key: str, override: str | None) -> dict:
    """The registered corpus the recipe ids resolve in, and its registry entry.

    The ids are read out of the corpus the pairs were chosen against rather than out of whatever
    file sits at the default path, so a re-registered or moved corpus is a hard stop instead of a
    quiet redesign from different recipes.
    """
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return {"path": str(path), "checksum": None, "registered_at": None, "source": "override"}

    paths = SnapshotPaths.from_paths_config({})
    entry = registry.get_dataset(paths.registry_file, dataset_key)
    if entry is None:
        raise ValueError(
            f"the registry does not know dataset key {dataset_key!r}; "
            "pass --recipes to read the corpus from a file instead"
        )
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    resolved = resolver.resolve(dataset_key)
    return {
        "path": str(resolved.path),
        "checksum": resolved.checksum,
        "registered_at": entry.get("created_at"),
        "source": "registry",
    }


def resolve_snapshot_corpus(snapshot):
    """The registered corpus behind the snapshot's dataset key, and its registry entry.

    The registry entry is read alongside the resolved path so the manifest can
    carry the size the corpus was registered at, which the resolver does not
    expose.
    """
    paths = SnapshotPaths.from_paths_config({})
    resolver = LocalDatasetResolver(
        registry_path=paths.registry_file,
        registry_get_dataset=registry.get_dataset,
    )
    return resolver.resolve(snapshot.dataset_key), registry.get_dataset(paths.registry_file, snapshot.dataset_key)


def load_recipes(path: Path) -> dict[int, Recipe]:
    """The cleaned corpus, indexed by recipe id."""
    with path.open() as handle:
        return {recipe.recipe_id: recipe for recipe in (Recipe(**row) for row in json.load(handle))}
