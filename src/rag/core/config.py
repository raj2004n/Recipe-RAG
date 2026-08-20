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
BUILDING_PATHS: dict = PATHS_CONFIG["building"]

# used when sealed snapshot directory needed
SNAPSHOT_PATHS: SnapshotPaths = SnapshotPaths.from_paths_config(PATHS_CONFIG)
