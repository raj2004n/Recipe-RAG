from rank_bm25 import BM25Okapi
from .tokenizer import Tokenizer
from ..chunker import Chunk

class BM25Index:
    def __init__(self, bm25: BM25Okapi, documents: list[Chunk]):
        self.bm25 = bm25
        self.documents = documents

class BM25Indexer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def build(self, chunks: list[Chunk]) -> BM25Index:
        tokenized_documents = [self.tokenizer.tokenize(chunk.text) for chunk in chunks]

        bm25 = BM25Okapi(tokenized_documents)

        return BM25Index(
            bm25=bm25,
            documents=chunks,
        )
