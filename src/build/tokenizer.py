# tokenizer.py

import re

from ..domain.types import ChunkSet, TokenizedChunk, TokenizedChunkSet


class Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def transform(
        self,
        chunk_sets: list[ChunkSet],
    ) -> list[TokenizedChunkSet]:
        return [
            TokenizedChunkSet(
                name=chunk_set.name,
                tokenized_chunks=[
                    TokenizedChunk.from_chunk(
                        chunk=chunk,
                        tokens=self.tokenize(chunk.text),
                    )
                    for chunk in chunk_set.chunks
                ],
            )
            for chunk_set in chunk_sets
        ]