import json
from pathlib import Path

from src.rag.retrieval.chunker import chunk_recipes
from src.rag.retrieval.sparse.bm25retriever import BM25Retriever
from src.rag.retrieval.sparse.bm25indexer import BM25Indexer
from src.rag.retrieval.sparse.tokenizer import Tokenizer


RECIPES_PATH = Path("data/processed/cleaned_recipes.json")
TOP_K = 10

def test(query: str = "chicken") -> None:
    # chunk recipes for the sparse retriever (instructions + ingredients)
    chunk_sets = chunk_recipes("instructions_ingredients", RECIPES_PATH)
    chunks = chunk_sets.chunks

    # build the BM25 index
    tokenizer = Tokenizer()
    indexer = BM25Indexer(tokenizer)
    index = indexer.build(chunks)

    # retrieve
    retriever = BM25Retriever(index, tokenizer)
    results = retriever.retrieve(query, k=TOP_K)

    # load raw recipes for display
    with open(RECIPES_PATH) as f:
        recipes = json.load(f)

    print(f"Query: {query}")
    print(f"Number of indexed chunks: {len(chunks)}")
    print(f"Top {len(results)} results:\n")

    for rank, r in enumerate(results, 1):
        recipe = recipes[r.chunk.recipe_id]
        print(
            f"{rank:>2}. score={r.score:.4f} "
            f"recipe_id={r.chunk.recipe_id} "
            f"type={r.chunk.chunk_type} "
            f"name={recipe['recipe_name']}"
        )
        print(f"     {r.chunk.text[:120]}{'...' if len(r.chunk.text) > 120 else ''}") # truncate excess text


if __name__ == "__main__":
    test()
