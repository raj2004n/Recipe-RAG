"""Ship the promoted baseline to the local Docker daemon as a running service.

The baseline is promoted by the gate into the registry's serve target, but a
promotion only records which snapshot is the deployment baseline; the runnable
form is the serve image with that snapshot baked in. The deployment target is
this machine, so the ship is a pull followed by a local run: it pulls the serve
image for the baseline snapshot from ghcr (or a repository given with
--image-repo), starts a container with the GPU passed through and the service
port mapped to the host, and waits until the service reports healthy before
returning.

The script fails closed: no serve target in the registry, a baseline whose
status is not complete, a missing promotion record, an image that cannot be
pulled, missing or malformed RECIPE_RAG_API_KEYS (the service refuses to start
without it), a failed docker call, or a service that never reports healthy each
stop the ship and exit non-zero. A baseline the gate recorded as failing is
not one of those: it ships, because the promotion that put it there was the
human's decision, and the checks it went ahead against are logged as a warning
first so an override is visible in the deploy log rather than silent. A
pre-existing recipe-rag container is replaced, so re-running the script ships a
newer baseline with the same command.
Docker access is expected to be available as plain `docker`; if your user is
not in the docker group, either prefix the script's docker calls with sudo or
add the user to the group once (root-equivalent access; effective on the next
login).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request

from src.rag.core.logging import configure_logging
from src.rag.core.snapshot import registry
from src.rag.core.snapshot.paths import SnapshotPaths

logger = logging.getLogger(__name__)

_IMAGE_REPO = "ghcr.io/raj2004n/recipe-rag"
_CONTAINER = "recipe-rag"
_API_KEYS_ENV = "RECIPE_RAG_API_KEYS"
_HOST_PORT = 8000
_HEALTH_PATH = "/health"
_HEALTH_TIMEOUT_S = 180
_HEALTH_POLL_S = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull and run the serve image for the promoted baseline on "
                    "this machine")
    parser.add_argument(
        "--image-repo", default=_IMAGE_REPO,
        help="image repository to pull the baseline from; pass e.g. 'recipe-rag' "
             "to use an image built locally (default ghcr)",
    )
    parser.add_argument(
        "--no-pull", action="store_true",
        help="skip the pull and expect an image tagged <image-repo>:<serve folder> "
             "to exist locally",
    )
    parser.add_argument(
        "--port", type=int, default=_HOST_PORT,
        help=f"host port mapped to the service port 8000 (default {_HOST_PORT})",
    )
    return parser.parse_args()


def _api_keys() -> str:
    """The API-keys JSON from the environment, or a hard stop if missing."""
    raw = os.environ.get(_API_KEYS_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"error: {_API_KEYS_ENV} is not set. The service refuses to start "
            f"without it; export a JSON object of key to role, e.g. "
            f"export {_API_KEYS_ENV}='{{\"dev-admin-key\":\"admin\",\"dev-public-key\":\"public\"}}'"
        )
    return raw


def _probe_key(api_keys: str) -> str:
    """A valid key for the readiness probe, preferring a public-role key."""
    mapping = json.loads(api_keys)
    for key, role in mapping.items():
        if role == "public":
            return key
    return next(iter(mapping))


def _failed_check_labels(record) -> list[str]:
    """One readable label per recorded check a promotion went ahead against."""
    labels = []
    for check in record.get("checks") or []:
        if check.get("passed"):
            continue
        labels.append(
            f"{check.get('scope')} {check.get('metric')} "
            f"{check.get('value')} {check.get('operator')} {check.get('bar')}"
        )
    return labels


def _serve_folder(paths) -> str:
    """The promoted folder from the registry, or a hard stop if none."""
    folder = registry.get_folder_for_target(paths.registry_file, "serve")
    if not folder:
        raise SystemExit(
            "error: no serve target in the registry. Run the promotion gate "
            "(scripts.promote_snapshot) on a passing snapshot first."
        )
    status = registry.get_status_for_folder(paths.registry_file, folder)
    if status != "complete":
        raise SystemExit(
            f"error: serve target {folder!r} has status {status!r}, not 'complete'; "
            f"refusing to ship an unsealed snapshot."
        )
    record = registry.get_promotion_record(paths.registry_file, folder)
    if record is None:
        raise SystemExit(
            f"error: no promotion record found for {folder!r}; the snapshot may "
            f"be complete but was never gated in."
        )
    if record.get("verdict") == "fail":
        failed = _failed_check_labels(record)
        logger.warning(
            "%s was promoted against the gate: %d of %d checks failed (%s); "
            "shipping it anyway",
            folder,
            len(failed),
            len(record.get("checks") or []),
            "; ".join(failed) or "no per-check outcome recorded",
        )
    return folder


def _pull_image(tag: str) -> None:
    logger.info("pulling %s", tag)
    proc = subprocess.run(["docker", "pull", tag])
    if proc.returncode != 0:
        raise SystemExit(
            f"error: could not pull {tag}. Is the image in ghcr? Push it first "
            f"(run the build workflow), or pass --image-repo to point at an "
            f"image built locally, e.g. --image-repo recipe-rag."
        )


def _run_container(tag: str, api_keys: str, port: int) -> None:
    subprocess.run(
        ["docker", "rm", "-f", _CONTAINER],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    logger.info("starting %s on port %s with the GPU passed through", tag, port)
    subprocess.run(
        ["docker", "run", "--gpus", "all", "-d", "--restart", "unless-stopped",
         "--name", _CONTAINER, "-p", f"{port}:8000",
         "-e", f"{_API_KEYS_ENV}={api_keys}", tag],
        check=True,
    )


def _wait_healthy(port: int, probe_key: str) -> None:
    url = f"http://127.0.0.1:{port}{_HEALTH_PATH}"
    request = urllib.request.Request(url, headers={"X-API-Key": probe_key})
    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("%s is healthy", url)
                    return
        except Exception:
            pass
        time.sleep(_HEALTH_POLL_S)
    subprocess.run(["docker", "logs", _CONTAINER], check=False)
    raise SystemExit(
        f"error: {url} never reported healthy within {_HEALTH_TIMEOUT_S}s; "
        f"container logs are printed above."
    )


def main() -> None:
    configure_logging()
    args = parse_args()

    paths = SnapshotPaths.from_paths_config({})
    folder = _serve_folder(paths)
    tag = f"{args.image_repo}:{folder}"

    if not args.no_pull:
        _pull_image(tag)
    else:
        logger.info("--no-pull set; expecting an existing %s image", tag)

    api_keys = _api_keys()
    _run_container(tag, api_keys, args.port)
    _wait_healthy(args.port, _probe_key(api_keys))

    print()
    print(f"Container {_CONTAINER} is up and healthy on port {args.port}.")
    print()
    print(f"  curl -H 'X-API-Key: <a public key from {_API_KEYS_ENV}>' \\")
    print("       -H 'content-type: application/json' \\")
    print(f"       -d '{{\"query\":\"quick chicken dinner\"}}' http://127.0.0.1:{args.port}/query")
    print()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"error: a subprocess returned non-zero ({exc.returncode}): {exc}", file=sys.stderr)
        sys.exit(1)
