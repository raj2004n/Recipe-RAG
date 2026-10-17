import json
import os
from typing import Iterable

from fastapi import Header

from .errors import ForbiddenError, UnauthenticatedError


def _load_api_keys() -> dict[str, str]:
    """Parse `RECIPE_RAG_API_KEYS` into a `{key: role}` mapping.

    Raises RuntimeError on missing or malformed input - the lifespan catches
    this and exits 2 so the service refuses to start with a bad config.
    """
    raw = os.environ.get("RECIPE_RAG_API_KEYS")
    if not raw:
        raise RuntimeError(
            "RECIPE_RAG_API_KEYS env var is missing. The service refuses to "
            "start without API keys configured."
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"RECIPE_RAG_API_KEYS is not valid JSON: {e}. Refusing to start."
        ) from e

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "RECIPE_RAG_API_KEYS must be a JSON object mapping keys to roles."
        )

    out: dict[str, str] = {}
    for key, role in parsed.items():
        if not isinstance(key, str) or not isinstance(role, str):
            raise RuntimeError(
                "RECIPE_RAG_API_KEYS entries must be string keys and string roles."
            )
        out[key] = role

    if not out:
        raise RuntimeError(
            "RECIPE_RAG_API_KEYS parsed to an empty mapping. Refusing to start."
        )

    return out


def _roles_granted(key_role: str, required: Iterable[str]) -> bool:
    """A role grants access to itself only. Admin is not implicit for public."""
    return key_role in set(required)


UI_KEY_ENV_VAR = "RAMSAY_UI_KEY"


def _resolve_ui_key(keys: dict[str, str]) -> str:
    """Pick the configured key the browser UI's forwarders act as.

    The page itself cannot hold a key, so the server injects one when it
    forwards to the keyed routes. `RAMSAY_UI_KEY` names that key; when it
    is unset the first key with the `public` role is used, which keeps a
    deployment that configures a single key working with no extra setup.

    Raises RuntimeError when the named key is absent from the map, or when
    nothing was named and the map carries no `public` key. The lifespan
    turns that into exit 2: a UI whose every forwarder would come back 401
    is not a state worth serving.
    """
    named = os.environ.get(UI_KEY_ENV_VAR)
    if named:
        if named not in keys:
            raise RuntimeError(
                f"{UI_KEY_ENV_VAR} does not name a key in RECIPE_RAG_API_KEYS. "
                "Refusing to start."
            )
        return named

    for key, role in keys.items():
        if role == "public":
            return key

    raise RuntimeError(
        "RECIPE_RAG_API_KEYS has no 'public' key and RAMSAY_UI_KEY is unset, "
        "so the UI has no key to forward with. Refusing to start."
    )


def require_role(*allowed: str):
    """Return a FastAPI dependency that enforces one of the allowed roles.

    `require_role("public")` lets either `public` or `admin` keys through
    because admin outranks public. `require_role("admin")` lets only admin.
    """

    allowed_set = set(allowed)
    allowed_with_admin = allowed_set | {"admin"}

    async def _check(x_api_key: str | None = Header(default=None)) -> str:
        keys = _load_api_keys()
        if x_api_key is None or x_api_key not in keys:
            raise UnauthenticatedError(
                "X-API-Key header is missing or unknown."
            )
        role = keys[x_api_key]
        if not _roles_granted(role, allowed_with_admin):
            raise ForbiddenError(
                f"Key role {role!r} lacks access; need one of {sorted(allowed_set)}."
            )
        return role

    return _check
