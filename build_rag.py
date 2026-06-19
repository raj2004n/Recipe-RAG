import mlflow
import yaml
import json
import logging
from src.models.encoder import Encoder
from src.models.indexer import Indexer
from src.models.retriever import Retriever
from src.chunking.chunker import Chunker
from src.models.generator import Generator
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

#TODO: Implement logger
#TODO: Implement MLFlow for tracking
#TODO: Create DVC pipeline

# config path
config_path = "configs/base_config.yaml"

def build_retriever(config: dict) -> tuple[Retriever, list[dict]]:
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
        index=indexer,
        chunks=chunks,
        config=config
    )
    
    # save retriever
    retriever.save("models/retriever")
    
    return retriever, recipes


def build_generator(config: dict, retriever: Retriever, recipes: list) -> Generator:
    model_name = config["generator"]["model_name"]
    generator = Generator(model_name)
    generator.save("models/generator")
    return generator


def build_rag(config_path: str = "configs/base_config.yaml"):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # build retriever
    retriever, recipes = build_retriever(config)
    
    # build generator
    generator = build_generator(config, retriever, recipes)
    
    # create RAG pipeline
    pipeline = RAGPipeline(retriever, generator)
    
    return pipeline


if __name__ == "__main__":
    pipeline = build_rag()
    
    # test
    print("Testing RAG Pipeline:")
    
    result = pipeline.query("How do I make chicken curry?")
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")