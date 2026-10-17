from dataclasses import dataclass, field


@dataclass
class Recipe:
    recipe_id: int = -1
    url: str = ""
    recipe_name: str = ""
    servings: float = 0.0
    info: list[dict[str, any]] = field(default_factory=list)
    ingredient_groups: list[dict[str, any]] = field(default_factory=list)
    instruction_groups: list[dict[str, any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    # Globally unique within a snapshot. Used as the FAISS id for dense
    # indices, so collisions across chunk sets would silently resolve a
    # hit to the wrong chunk.
    chunk_id: int
    recipe_id: int
    chunk_type: str
    text: str


@dataclass
class ChunkSet:
    name: str
    chunks: list[Chunk]


# A chunk id mapped back to the Chunk it names, the shape the indexers carry
# and the shape a snapshot restores when it loads.
ChunkById = dict[int, Chunk]


@dataclass
class EncodedChunk:
    vector: list[float]
    text: str
    recipe_id: int
    chunk_id: int
    # Carried from Chunk so the indexer can rebuild a full Chunk on the
    # other side without needing to re-look-up the source chunk.
    chunk_type: str = ""

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: list[float]) -> "EncodedChunk":
        return cls(
            vector=vector,
            text=chunk.text,
            recipe_id=chunk.recipe_id,
            chunk_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
        )


@dataclass
class EncodedChunkSet:
    name: str
    encoded_chunks: list[EncodedChunk]


@dataclass
class TokenizedChunk:
    chunk: Chunk
    tokens: list[str]

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        tokens: list[str],
    ) -> "TokenizedChunk":
        return cls(
            chunk=chunk,
            tokens=tokens,
        )


@dataclass
class TokenizedChunkSet:
    name: str
    tokenized_chunks: list[TokenizedChunk]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class FusedChunk:
    """A chunk after fusion + dedupe; ready for the reranker."""

    chunk: Chunk
    fused_score: float
    sources: tuple[str, ...] = field(default_factory=tuple)
