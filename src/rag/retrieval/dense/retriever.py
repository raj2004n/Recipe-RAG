from dataclasses import dataclass
from ..chunker import Chunk
from .encoder import Encoder
from .indexer import Indexer


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
            self,
            encoder: Encoder,
            indexer: Indexer,
            top_k: int,
        ):
        """ Initialise a Retriever object.

        Args:
            encoder: Encoder used to encode the query.
            indexer: Indexer used to search for nearest neighbors. The
                     indexer owns chunk resolution — the retriever just
                     consumes its hits.
            top_k: default number of results to return.

        The encoder turns the query into a dense vector in the same space
        as the indexed recipe vectors. The indexer searches for nearest
        neighbors and resolves them back to Chunks.
        """
        self.encoder = encoder
        self.indexer = indexer
        self.top_k = top_k


    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.top_k

        # encode query into dense vector
        query_vector = self.encoder.encode_query(query)

        # indexer handles the k-cut, score-threshold, and chunk resolution
        hits = self.indexer.search(query_vector, top_k)[0]

        results: list[RetrievalResult] = []
        for score, chunk in hits:
            if chunk is None:
                # FAISS returned an id the indexer doesn't have a Chunk for
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                )
            )
        return results
