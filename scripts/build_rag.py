import mlflow
import yaml
import json
from src.models.encoder import Encoder
from src.models.indexer import Indexer
from src.models.retriever import Retriever
from src.chunking.chunker import Chunker
from src.models.generator import Generator
from src.pipeline.rag_pipeline import RAGPipeline

# TODO: Add logging
#TODO: Implement MLFlow for tracking
#TODO: Create DVC pipeline
#TODO: Choose to pass config or config path, and keep it consistent

# conect to remote MlFlow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-rag-experiment")



def flatten_dict(d, parent_key="", sep="."):
    items = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v

    return items

def build_retriever(config: dict) -> tuple[Retriever, list[dict]]:
    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)
    
    chunker = Chunker(config)

    # chunk
    chunks = []
    for recipe in recipes:
        recipe_chunks = chunker.chunk(recipe)
        chunks.extend(recipe_chunks)
    
    # load encoder
    encoder = Encoder(config)
    
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


def build_rag(config_path: str = "configs/base_config.yaml") -> RAGPipeline:
    # load config, move to better place later
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with mlflow.start_run():
        # build retriever
        retriever, recipes = build_retriever(config)
        
        # build generator
        generator = build_generator(config, retriever, recipes)
        
        # create RAG pipeline
        pipeline = RAGPipeline(retriever, generator)

        # log params
        flat_config = flatten_dict(config)
        mlflow.log_params(flat_config)

        # log artifact
        mlflow.log_artifact(config_path)

    return pipeline


if __name__ == "__main__":
    pipeline = build_rag()
    
    # testing, remove later
    #print("Testing RAG Pipeline:")
    
    #result = pipeline.query("How do I make chicken curry?")
    #print(f"\nQuestion: {result['question']}")
    #print(f"\nAnswer: {result['answer']}")