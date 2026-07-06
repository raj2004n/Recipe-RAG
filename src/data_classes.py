from dataclasses import dataclass
from src.retrieval.indexer import Indexer

# TODO: Consider modifying RetrieverResult
@dataclass
class RetrievalResult:
    chunk_id: str
    recipe_id: str
    chunk: dict
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