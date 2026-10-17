from typing import Any

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

# PATHS_CONFIG sub-pieces
DATA_PATHS: dict = PATHS_CONFIG["data"]


BASE_CONFIG_FILE = "base_config.yaml"
PATHS_CONFIG_FILE = "paths_config.yaml"

# The config keys each build stage takes its values from, and the only ones it may
# read. Every row mirrors the named `params:` block the same stage declares in
# dvc.yaml, so an edit to a key outside a stage's row leaves that stage's outputs
# alone, and tests/test_dvc_graph.py holds the two lists to each other. The
# PATHS_CONFIG_FILE entry every stage carries is `configs`, the block that names
# the base config file itself: a stage reads that at import, before it is known
# which stage it is, so it is declared rather than checked on read.
STAGE_CONFIG_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    "set_strategy": {
        BASE_CONFIG_FILE: ("rag.strategy",),
        PATHS_CONFIG_FILE: ("configs", "data.cleaned_recipes"),
    },
    "chunk": {
        PATHS_CONFIG_FILE: ("configs",),
    },
    "transform": {
        BASE_CONFIG_FILE: (
            "encoder.model_id",
            "encoder.batch_size",
            "encoder.normalize_embeddings",
        ),
        PATHS_CONFIG_FILE: ("configs",),
    },
    "index": {
        BASE_CONFIG_FILE: ("indexer.type", "indexer.dimension"),
        PATHS_CONFIG_FILE: ("configs",),
    },
    "finalise_build": {
        PATHS_CONFIG_FILE: ("configs",),
    },
}

_CONFIG_FILES: dict[str, dict] = {
    BASE_CONFIG_FILE: BASE_CONFIG,
    PATHS_CONFIG_FILE: PATHS_CONFIG,
}


class UndeclaredConfigKey(RuntimeError):
    """A build stage read a config key it did not declare."""


class StageConfig:
    """The config values one build stage declared, checked as they are read.

    A build stage takes what it works from through this object rather than from
    the module-level singletons, so a stage that starts reading a new key fails on
    the run that introduces it instead of changing its output while DVC still sees
    an unchanged dependency.
    """

    def __init__(self, stage: str, declared: dict[str, tuple[str, ...]]) -> None:
        self._stage = stage
        self._declared = declared

    def get(self, file: str, key: str) -> Any:
        """Return `key` from `file`, both as the stage's row spells them.

        Args:
            file: the config file's name, e.g. `base_config.yaml`.
            key: the dotted path into it, e.g. `encoder.batch_size`.

        Raises:
            UndeclaredConfigKey: when the stage's row names neither the file nor
                the key.
        """
        if file not in self._declared:
            raise UndeclaredConfigKey(
                f"stage {self._stage} reads {file}, which it does not declare"
            )
        if key not in self._declared[file]:
            raise UndeclaredConfigKey(
                f"stage {self._stage} reads {file}: {key}, which it does not declare"
            )

        value: Any = _CONFIG_FILES[file]
        for part in key.split("."):
            value = value[part]
        return value


def stage_config(stage: str) -> StageConfig:
    """Return the config view for one build stage.

    Args:
        stage: the stage's name, as `dvc.yaml` and `STAGE_CONFIG_KEYS` spell it.
    """
    return StageConfig(stage, STAGE_CONFIG_KEYS[stage])


# The staging tree is where a build in flight writes, and each sealed build
# gets its own versioned folder under snapshots/. The registry tracks which
# versioned folder is 'latest_build' or 'serve'. A build stage calls this
# helper with no name to reach the staging tree, and a reader calls it with the
# folder name the registry resolved.
def get_snapshot_paths(versioned_name: str | None = None) -> SnapshotPaths:
    """Return a SnapshotPaths resolved for the given versioned folder name.

    When `versioned_name` is `None`, the staging tree at `build/staging/` is
    resolved, which is where every build stage reads and writes.
    """
    return SnapshotPaths.from_paths_config(PATHS_CONFIG, versioned_name=versioned_name)
