from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import yaml

from ..core.config import DATA_PATHS

_JUDGE_PERSONAS = ("core", "expectedness")


@lru_cache(maxsize=1)
def _load() -> dict:
    path = Path(DATA_PATHS["eval_system_messages"])
    with path.open() as f:
        return yaml.safe_load(f)


def judge_messages() -> dict[str, str]:
    """The judge system messages, keyed by persona name."""
    messages = _load().get("judge", {})
    missing = [name for name in _JUDGE_PERSONAS if name not in messages]
    if missing:
        raise RuntimeError(
            f"eval_system_messages.yaml is missing judge personas: {missing}"
        )
    return {name: messages[name] for name in _JUDGE_PERSONAS}


def judge_prompt_hash() -> str:
    """A stable digest of the judge personas, logged per eval run."""
    blob = "\n\n".join(judge_messages()[name] for name in _JUDGE_PERSONAS)
    return sha256(blob.encode()).hexdigest()[:16]