from functools import lru_cache
from pathlib import Path

import yaml

from ...core.config import DATA_PATHS
from .types import QUERY_TYPES, QUERY_TYPE_SPECS


@lru_cache(maxsize=1)
def _load() -> dict:
    path = Path(DATA_PATHS["eval_system_messages"])
    with path.open() as f:
        return yaml.safe_load(f)


def _section(name: str) -> dict[str, str]:
    messages = _load()[name]
    missing = set(QUERY_TYPES) - set(messages)
    if missing:
        raise RuntimeError(
            f"eval_system_messages.yaml missing {name} types: {sorted(missing)}"
        )
    return messages


def prompts_drafter_messages() -> dict[str, str]:
    """Per-type system messages for the prompts drafter, labels formatted."""
    labels = {spec.name: spec.label for spec in QUERY_TYPE_SPECS if spec.label}
    return {
        name: message.replace("{label}", labels[name]) if name in labels else message
        for name, message in _section("prompts_drafter").items()
    }


def answers_drafter_messages() -> dict[str, str]:
    """Per-type system messages for the answers drafter."""
    return dict(_section("answers_drafter"))
