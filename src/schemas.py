from dataclasses import dataclass
from src.retrieval.indexer import Indexer

@dataclass
class Chunk:
    text: str
    chunk_type: str
    recipe_id: int
    chunk_id: int

@dataclass
class EncodedResult:
    id: int
    vector: list[float]
    text: str
    metadata: dict[str, any]

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