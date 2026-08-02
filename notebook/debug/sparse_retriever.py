import json

from src.rag.retrieval.chunker import chunk_recipes, ChunkSet, Chunk
from src.rag.retrieval.sparse.tokenizer import Tokenizer
from src.rag.retrieval.sparse.bm25indexer import BM25Indexer
from src.rag.retrieval.sparse.bm25retriever import BM25Retriever

def test_bm25_retriever(query: str):
    # load recipes
    recipes_path = "data/processed/cleaned_recipes.json"

    # chunk recipes 
    chunk_sets : list[ChunkSet] = chunk_recipes(
        strategy="instructions_ingredients",
        recipes_path=recipes_path
    )
    chunks : list[Chunk] = chunk_sets[0].chunks

    # build index
    tokenizer = Tokenizer()
    indexer = BM25Indexer(tokenizer)
    index = indexer.build(chunks)

    # build retriever
    retriever = BM25Retriever(index, tokenizer)

    # retrieve
    results = retriever.retrieve(query, top_k=10)
    
    recipe_ids = [r.chunk.recipe_id for r in results]
    chunk_types = [r.chunk.chunk_type for r in results]
    texts = [r.chunk.text for r in results]

    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)

    retrieved_recipes = [recipes[recipe_id]["recipe_name"] for recipe_id in recipe_ids]

    print(f"Retrieved recipes: {retrieved_recipes}")

if __name__ == "__main__": 
    test_bm25_retriever(
        query="chicken"
    )