import json
import os
from dataclasses import asdict

from ..domain.types import (
    Chunk,
    ChunkSet,
    EncodedChunk,
    EncodedChunkSet,
    TokenizedChunk,
    TokenizedChunkSet,
)


def _read_json(path: str) -> any:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, payload: any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_chunk_sets(chunk_sets: list[ChunkSet], save_path: str):
    os.makedirs(save_path, exist_ok=True)
    _write_json(
        os.path.join(save_path, "chunk_sets.json"),
        [asdict(cs) for cs in chunk_sets],
    )


def load_chunk_sets(save_path: str) -> list[ChunkSet]:
    data = _read_json(os.path.join(save_path, "chunk_sets.json"))

    return [
        ChunkSet(
            name=cs["name"],
            chunks=[Chunk(**chunk) for chunk in cs["chunks"]],
        )
        for cs in data
    ]


def save_encoded_chunk_sets(encoded_chunk_sets: list[EncodedChunkSet], save_path: str):
    os.makedirs(save_path, exist_ok=True)
    _write_json(
        os.path.join(save_path, "encoded_chunk_sets.json"),
        [asdict(cs) for cs in encoded_chunk_sets],
    )


def save_tokenized_chunk_sets(tokenized_chunk_sets: list[TokenizedChunkSet], save_path: str):
    os.makedirs(save_path, exist_ok=True)
    _write_json(
        os.path.join(save_path, "tokenized_chunk_sets.json"),
        [asdict(cs) for cs in tokenized_chunk_sets],
    )


def load_encoded_chunk_sets(save_path: str) -> list[EncodedChunkSet]:
    data = _read_json(os.path.join(save_path, "encoded_chunk_sets.json"))

    return [
        EncodedChunkSet(
            name=chunk_set["name"],
            encoded_chunks=[
                EncodedChunk(**chunk)
                for chunk in chunk_set["encoded_chunks"]
            ],
        )
        for chunk_set in data
    ]


def load_tokenized_chunk_sets(save_path: str) -> list[TokenizedChunkSet]:
    data = _read_json(os.path.join(save_path, "tokenized_chunk_sets.json"))

    return [
        TokenizedChunkSet(
            name=chunk_set["name"],
            tokenized_chunks=[
                TokenizedChunk(
                    chunk=Chunk(**tokenized_chunk["chunk"]),
                    tokens=tokenized_chunk["tokens"],
                )
                for tokenized_chunk in chunk_set["tokenized_chunks"]
            ],
        )
        for chunk_set in data
    ]
