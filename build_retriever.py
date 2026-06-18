import mlflow
import yaml
import json
import logging
from src.models.encoder import Encoder
from src.models.indexer import Indexer
from src.models.retriever import Retriever
from src.chunking.chunker import Chunker

def build_retriever(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)
    
    chunker = Chunker(config_path)
    # chunk
    chunks = []
    for recipe in recipes:
        recipe_chunks = chunker.chunk(recipe)
        chunks.extend(recipe_chunks)
    
    # load encoder
    encoder = Encoder(config_path)
    
    # encode documents
    texts = [c["text"] for c in chunks]
    embeddings = encoder.encode_documents(texts)
    
    # index
    indexer = Indexer(
        dimension=embeddings.shape[1],
        index_type=config["index"]["type"]
    )

    # add vectors to index
    indexer.add(embeddings, ids=None)
    
    # build retriever
    retriever = Retriever(
        encoder=encoder,
        indexer=indexer,
        chunks=chunks,
        config_path=config_path
    )
    
    # save retriever
    retriever.save("models/retriever")
    
    return retriever


if __name__ == "__main__":
    retriever = build_retriever("configs/base_config.yaml")
    
    # Test
    results = retriever.retrieve("chicken curry")
    for r in results:
        print(f"[{r.score:.3f}] {r.chunk['metadata']['recipe_name']}")