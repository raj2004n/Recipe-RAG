"""Project-wide configuration singletons."""

from .utils import load_config
from .snapshot.paths import SnapshotPaths

_PATHS_CONFIG_PATH = "configs/paths_config.yaml"
_PATHS_CONFIG = load_config(_PATHS_CONFIG_PATH)

BASE_CONFIG: dict = load_config(_PATHS_CONFIG["configs"]["base_config"])
PROMPTS_CONFIG: dict = load_config(_PATHS_CONFIG["configs"]["prompts_config"])
PATHS_CONFIG: dict = _PATHS_CONFIG

# BASE_CONFIG sub-pieces
ENCODER_CONFIG: dict = BASE_CONFIG["encoder"]
INDEXER_CONFIG: dict = BASE_CONFIG["indexer"]
RAG_STRATEGY: str = BASE_CONFIG["rag"]["strategy"]
EVAL_DATASET_CONFIG: dict = BASE_CONFIG["evaluation dataset generator"]

# PATHS_CONFIG sub-pieces
DATA_PATHS: dict = PATHS_CONFIG["data"]


# Snapshots no longer have a single fixed location - each build gets a
# versioned folder under snapshots/, and the registry tracks which folder
# is 'current' or 'latest'. Callers that need paths for the active build
# look up the versioned name in the registry first, then call this helper.
def get_snapshot_paths(versioned_name: str | None = None) -> SnapshotPaths:
    """Return a SnapshotPaths resolved for the given versioned folder name.

    When `versioned_name` is `None`, the legacy `snapshots/building/` root
    is used (kept for any code path that has not yet migrated).
    """
    return SnapshotPaths.from_paths_config(PATHS_CONFIG, versioned_name=versioned_name)
