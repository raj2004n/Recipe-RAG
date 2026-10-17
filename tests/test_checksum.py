import hashlib
from pathlib import Path

from src.rag.core.checksum import file_checksum


def test_file_checksum_matches_a_plain_sha256(tmp_path: Path):
    payload = tmp_path / "input.json"
    payload.write_bytes(b'{"a": 1}\n')
    assert file_checksum(payload) == "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()


def test_file_checksum_spans_its_read_blocks(tmp_path: Path):
    """A file larger than one read block hashes the same as its whole bytes.

    The digest is accumulated block by block, so the block size must not be
    observable in the result.
    """
    payload = tmp_path / "large.bin"
    payload.write_bytes(bytes(range(256)) * 8192)
    assert payload.stat().st_size > (1 << 20)
    assert file_checksum(payload) == "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()
