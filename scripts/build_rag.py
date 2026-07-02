import mlflow
import yaml
import json
import numpy as np
from src.models.encoder import Encoder
from src.models.indexer import Indexer
from src.models.retriever import Retriever
from src.chunking.chunker import Chunker
from src.prompting.prompt_builder import PromptBuilder
from src.models.generator import Generator
from src.pipeline.rag_pipeline import RAGPipeline
from transformers import AutoTokenizer
import os
import copy

#TODO: Add logging
#TODO: Seperate the build script into R and AG parts

# connect to remote MlFlow server
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


def build_retriever(config: dict) -> Retriever:
    # load recipes
    with open("data/processed/cleaned_recipes.json") as f:
        recipes = json.load(f)
    
    # initialise chunker
    chunker = Chunker(config)

    # chunk recipes
    chunks = []
    for recipe in recipes:
        recipe_chunks = chunker.chunk(recipe)
        chunks.extend(recipe_chunks)

    # initialise transformer encoder
    encoder = Encoder(config)
    
    # encode documents into dense vectors
    texts = [c["text"] for c in chunks]
    embeddings = encoder.encode_documents(texts)

    # initialise FAISS index
    indexer = Indexer(
        dimension=embeddings.shape[1],
        index_type=config["index"]["type"]
    )
    
    # assign ids to chunks (preserves order)
    ids = np.arange(len(chunks))

    # add vectors to index
    indexer.add(embeddings, ids)
    
    # build retriever
    retriever = Retriever(
        encoder=encoder,
        index=indexer,
        chunks=chunks,
        config=config
    )

    return retriever


def save_retriever_artifats(save_path: str, chunks: list[dict], indexer: Indexer, config: dict):
    # create path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # save FAISS index
    indexer.save(os.path.join(save_path, "index.faiss"))
    
    # save chunks
    with open(os.path.join(save_path, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    resolved_config = copy.deepcopy(config)

    # save config
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(resolved_config, f, sort_keys=False)


def load_retriever(artifact_path: str):
    # load config
    with open(os.path.join(artifact_path, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # load chunks
    with open(os.path.join(artifact_path, "chunks.json")) as f:
        chunks = json.load(f)

    encoder = Encoder(config)

    indexer = Indexer(
        dimension=config["index"]["dimension"],
        index_type=config["index"]["type"]
    )

    indexer.load(os.path.join(artifact_path, "index.faiss"))

    retriever = Retriever(
        encoder=encoder,
        index=indexer,
        chunks=chunks,
        config=config
    )

    return retriever


def build_generator(config: dict) -> Generator:
    model_name = config["generator"]["model_name"]
    generator = Generator(model_name)
    return generator


def build_rag(config_path: str = "configs/base_config.yaml", save: bool = True, load: bool = False, artifact_path: str = "models/retriever") -> RAGPipeline:
    # load config, move to better place later
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with mlflow.start_run():

        if load:
            retriever = load_retriever(artifact_path)

        # build retriever
        retriever = build_retriever(config)

        # save retriever
        if save:
            save_retriever_artifats(artifact_path, retriever.chunks, retriever.indexer, config)
        
        # build generator
        generator = build_generator(config)
        # build tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["generator"]["model_name"])
        # build prompt builder
        prompt_builder = PromptBuilder(config, tokenizer)
        # create RAG pipeline
        pipeline = RAGPipeline(retriever, prompt_builder, generator, tokenizer)

        # log params
        mlflow.log_params(flatten_dict(config))
        # log artifact
        mlflow.log_artifact(config_path)

    return pipeline


if __name__ == "__main__":
    pipeline = build_rag()
    
    # testing, remove later
    print("Testing RAG Pipeline:")
    
    result = pipeline.query("How do I make chicken curry?")
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")