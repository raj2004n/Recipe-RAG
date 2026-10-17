from rank_bm25 import BM25Okapi
from ..domain.types import ChunkById, TokenizedChunk, TokenizedChunkSet
from ..snapshot.types import BM25IndexedChunkSet


class BM25Index:
    def __init__(self, bm25: BM25Okapi, documents: list[TokenizedChunk]):
        self.bm25 = bm25
        self.documents = documents


class BM25Indexer:
    def build(self, chunk_set: TokenizedChunkSet) -> list[BM25IndexedChunkSet]:
        chunks_by_id: ChunkById = {
            tc.chunk.chunk_id: tc.chunk for tc in chunk_set.tokenized_chunks
        }
        tokenized_documents = [chunk.tokens for chunk in chunk_set.tokenized_chunks]

        bm25 = BM25Okapi(tokenized_documents)

        return [BM25IndexedChunkSet(
            name=chunk_set.name,
            index=bm25,
            tokenized_chunks=chunk_set.tokenized_chunks,
            chunks=chunks_by_id,
        )]