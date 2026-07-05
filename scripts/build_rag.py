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
from src.utils import flatten_config_dict, load_config, build_generator
import os
import copy
from pathlib import Path

#TODO: Add logging

# connect to remote MlFlow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-rag-experiment")

def get_chunks(recipes_path: Path, chunker: Chunker):
    # load recipes
    with open(recipes_path) as f:
        recipes = json.load(f)
    
    # chunk recipes
    chunks = chunker.chunk(recipes)
    for r in recipes:
        recipe_chunks = chunker.chunk(r)
        chunks.extend(recipe_chunks)
    return chunks

def encode_chunks(chunks: list[dict], encoder: Encoder) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = encoder.encode_documents(texts)    
    return embeddings

def assign_ids(embeddings: np.ndarray, indexer: Indexer) -> Indexer:
    # assign ids to chunks (preserves order)
    ids = np.arange(len(embeddings))
    # add vectors to index
    indexer.add(embeddings, ids)
    return indexer

def save_config(directory: Path, config: dict):
    path = directory / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

def run():
    model_config = load_config("configs/base_config.yaml")
    prompts_config = load_config("configs/prompts_config.yaml")

    paths = model_config["paths"]
    encoder_config = model_config["encoder"]
    indexer_config = model_config["indexer"]
    retriever_config = model_config["retrieval"]
    generator_config = model_config["generator"]

    prompt_config = prompts_config["config"]


    with mlflow.start_run():    
        chunker = Chunker(model_config["chunking"]["strategy"])

        encoder = Encoder(
            encoder_config["model"],
            encoder_config["batch_size"],
            encoder_config["normalize_embeddings"]
        )
        
        indexer = Indexer(
            indexer_config["type"],
            indexer_config["dimension"]
        )
    
        chunks = get_chunks(paths["cleaned_recipes"], chunker)
        embeddings = encode_chunks(chunks, encoder)
        indexer = assign_ids(embeddings, indexer) # note: indexer contains the index

        retriever = Retriever(
            chunks,
            encoder,
            indexer,
            retriever_config["top_k"],
            retriever_config["score_threshold"]
        )

        tokenizer = AutoTokenizer.from_pretrained(generator_config["model"])

        prompt_builder = PromptBuilder(
            tokenizer,
            prompt_config["max_context_tokens"]
        )

        generator = build_generator(
            generator_config["model"],
            generator_config["adapter_path"]
        )

        
        # build RAG pipeline
        pipeline = RAGPipeline(retriever, prompt_builder, generator)

        # save chunks, indexer
        retriever.save_chunks(paths["directories"]["retriever_artifacts"])
        retriever.save_indexer(paths["directories"]["retriever_artifacts"])

        # save config
        save_config(paths["directories"]["retriever_artifacts"], model_config)

        # encoder, tokenizer (in prompt builder) are pre-loaded, so no need to save

        # log params, artifacts
        mlflow.log_params(flatten_config_dict(model_config))
        mlflow.log_artifact(paths["base_config"])

    return

if __name__ == "__main__":
    run()