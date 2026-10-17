from dataclasses import dataclass, field

import faiss
from rank_bm25 import BM25Okapi

from ..domain.types import ChunkById, TokenizedChunk


@dataclass
class IndexedChunkSet:
    name: str
    index: faiss.Index
    chunks: ChunkById = field(default_factory=dict)
    # chunks is populated by index_embeddings (from EncodedChunk.chunk) or by
    # set_indexed_sets (from a snapshot's chunks.json)
    # required for search() to resolve hits back to Chunks


@dataclass
class BM25IndexedChunkSet:
    name: str
    index: BM25Okapi
    tokenized_chunks: list[TokenizedChunk] = field(default_factory=list)
    chunks: ChunkById = field(default_factory=dict)
    # chunks is populated by build() from tokenized_chunks[i].chunk, keyed by chunk_id
    # required for search() to resolve hits back to Chunks
