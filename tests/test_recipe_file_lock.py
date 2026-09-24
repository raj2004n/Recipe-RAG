"""Tests for the advisory recipe-file lock.

The lock is the runtime guard that makes a mid-build preprocessing swap
visible: while a build holds it, a concurrent preprocess that respects it
cannot overwrite the recipes file. We don't enforce it (the protection
comes from pinning dataset identity at set_strategy), so these tests
verify the lock semantics rather than the swapping behaviour directly.
"""

import threading
from pathlib import Path

import pytest

from src.rag.core.snapshot import lifecycle


def _make_recipes(tmp_path: Path, name: str = "recipes.json") -> Path:
    path = tmp_path / name
    path.write_text("[]")
    return path


def test_lock_acquires_and_releases(tmp_path: Path):
    recipes = _make_recipes(tmp_path)
    lock_path = recipes.with_suffix(recipes.suffix + ".buildlock")

    with lifecycle.recipe_file_lock(recipes):
        assert lock_path.exists()
    # Released on exit: the lock file is removed.
    assert not lock_path.exists()


def test_lock_releases_on_exception(tmp_path: Path):
    recipes = _make_recipes(tmp_path)
    lock_path = recipes.with_suffix(recipes.suffix + ".buildlock")

    with pytest.raises(RuntimeError):
        with lifecycle.recipe_file_lock(recipes):
            assert lock_path.exists()
            raise RuntimeError("simulated stage failure")
    assert not lock_path.exists()


def test_concurrent_lock_blocks(tmp_path: Path):
    """Two holders should not both think they own the lock. We acquire it
    in the main thread, spawn a second thread that tries to acquire with
    LOCK_EX (exclusive) and a short timeout, then release. The second
    attempt must time out - that is what makes the lock observable."""
    import fcntl
    recipes = _make_recipes(tmp_path)
    lock_path = recipes.with_suffix(recipes.suffix + ".buildlock")
    lock_path.touch()

    results: list[str] = []

    def contender():
        fd = open(lock_path, "r")
        try:
            # Non-blocking attempt; should fail immediately because the
            # main thread already holds the exclusive lock.
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                results.append("acquired")
            except BlockingIOError:
                results.append("blocked")
        finally:
            fd.close()

    with lifecycle.recipe_file_lock(recipes):
        t = threading.Thread(target=contender)
        t.start()
        t.join(timeout=2.0)

    assert results == ["blocked"]
