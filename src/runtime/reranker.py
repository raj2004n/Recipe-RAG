
import torch

from sentence_transformers import CrossEncoder

from ..domain.types import FusedChunk


class CrossEncoderReranker:
    """Rerank fused chunks with a cross-encoder."""

    def __init__(self, model_id: str, max_length: int = 512):
        self.model_id = model_id
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    # defer the model until first rerank to minimise cost of construction
    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_id,
                max_length=self.max_length,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[FusedChunk],
        top_n: int,
    ) -> list[FusedChunk]:
        """Return the top_n chunks by cross-encoder score.

        Each returned FusedChunk has `fused_score` replaced by the
        cross-encoder score, and "reranker" appended to `sources`.
        """
        if not chunks or top_n <= 0:
            return []

        model = self._ensure_model()
        pairs = [(query, fc.chunk.text) for fc in chunks]
        scores = model.predict(pairs, batch_size=32)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda cs: cs[1],
            reverse=True,
        )[:top_n]

        return [
            FusedChunk(
                chunk=fc.chunk,
                fused_score=float(score),
                sources=fc.sources + ("reranker",),
            )
            for fc, score in ranked
        ]
