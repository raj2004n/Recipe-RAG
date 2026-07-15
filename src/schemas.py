from dataclasses import dataclass
from src.retrieval.indexer import Indexer

@dataclass
class Chunk:
    text: str
    chunk_type: str
    recipe_id: int
    chunk_id: int

@dataclass
class EncodedChunk:
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, any]

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: list[float]) -> "EncodedChunk":
        return cls(
            id=f"recipa_{chunk.recipe_id}_chunk_{chunk.chunk_id}",
            vector=vector,
            text=chunk.text,
            metadata={
                "recipe_id": chunk.recipe_id,
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type
            }
        )

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float

@dataclass
class Snapshot:
    # ids
    snapshot_id: str | None
    created_at: str | None

    # retriever
    index: Indexer
    chunks: list[dict]

    # configs
    base_config: dict[str, any]
    prompts_config: dict[str, any]