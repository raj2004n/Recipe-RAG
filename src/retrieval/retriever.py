from dataclasses import dataclass
from .chunker import Chunk
from .encoder import Encoder
from .indexer import Indexer

# TODO: Should have save and load method here for consistency.
# Consider a serparate save methods for the index and the chunks for clarity.

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float

class Retriever:
    def __init__(
            self,
            chunks: list[Chunk],
            encoder: Encoder,
            indexer: Indexer,
            top_k: int,
            score_threshold: float
        ):
        """ Initialise a Retriever object.

            Args:
                encoder (Encoder): The encoder object used to encode documents.
                index (Indexer): The index object used to search for nearest neighbors.
                chunks (list[dict]): A list of dictionaries containing chunk information.
                config (dict): The configuration dictionary.

        The encoder turns the query into a dense vector in the same space as the recipe vectors.
        The index is used to search for the nearest neighbors of the query vector (in the encoder space of the recipe vectors).
        The chunks maps the retrieved indices (top k) to the corresponding recipe chunks.
        """
        self.encoder = encoder
        self.indexer = indexer
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.chunks = {c.id: c for c in chunks}

    def retrieve(self, query: str, top_k : int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.top_k
        
        # encode query into dense vector
        query_vector = self.encoder.encode_query(query)

        # search recipe index for nearest neighbors of query vector
        scores, chunk_ids = self.indexer.search(query_vector, top_k)

        # build results
        results: list[RetrievalResult] = []

        for score, chunk_id in zip(scores[0], chunk_ids[0]):
            if chunk_id is None:
                continue

            if self.score_threshold and score < self.score_threshold:
                continue

            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

            # break if enough results are found
            if len(results) >= top_k:
                break
        return results