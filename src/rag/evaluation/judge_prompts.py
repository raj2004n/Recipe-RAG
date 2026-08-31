"""Load the generation-judge personas from configs/eval_system_messages.yaml.

The judge has two personas so the reference answer reaches the judge only
when it is being scored: the core judge sees the query, the reference
recipes, and the generated recipe, while the expectedness judge additionally
sees the reference answer. The messages are read from the live config at eval
time rather than carried by the snapshot, so judge prompts can evolve without
a new build; every eval run logs the prompt hash so a score is always
traceable to the exact prompts that produced it.

Each message is a full standalone string (persona, task, and JSON line baked
in); the user-message templates and the sampling options stay in code.
"""

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