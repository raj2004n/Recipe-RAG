import os
import json
import mlflow
import argparse
from src.rag.retrieval.encoder import Encoder, EncodedChunkSet
from src.rag.retrieval.indexer import Indexer
from src.rag.retrieval.retriever import Retriever
from src.rag.retrieval.chunker import chunk_recipes, Chunk, ChunkSet
from src.rag.core.utils import flatten_config_dict, load_config
from src.rag.core.snapshot import Snapshot, save_snapshot, save_chunks, save_encoded_chunks, load_chunks

#TODO: Add logging

# connect to remote MlFlow server
#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_experiment("first-rag-experiment")

MODEL_CONFIG = load_config("configs/base_config.yaml")
PROMPTS_CONFIG = load_config("configs/prompts_config.yaml")
PATHS_CONFIG = load_config("configs/paths_config.yaml")

FILES = PATHS_CONFIG["files"]
DIRECTORIES = PATHS_CONFIG["directories"]

ENCODER_CONFIG = MODEL_CONFIG["encoder"]
INDEXER_CONFIG = MODEL_CONFIG["indexer"]
RETRIEVER_CONFIG = MODEL_CONFIG["retriever"]
GENERATOR_CONFIG = MODEL_CONFIG["generator"]

PROMPT_TEMPLATES = PROMPTS_CONFIG["template"]
PROMPT_SETTINGS = PROMPTS_CONFIG["settings"]

# TODO: Model strategy
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        choices=["chunk", "encode", "index", "retrieve", "build"],
                        help="The RAG pipeline step to run.")
    return parser.parse_args()

def chunk():
    chunk_set_list = chunk_recipes(MODEL_CONFIG["rag"]["strategy"], FILES["cleaned_recipes"])
    save_chunks(
        chunk_set_list=chunk_set_list, 
        save_path=PATHS_CONFIG["building"]["chunks"]
    )

def encode_chunk():
    chunks_path = PATHS_CONFIG["building"]["chunks"]
    
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks directory not found at {chunks_path}")

    chunks_set_list = load_chunks(chunks_path)

    encoder = Encoder(
        model_id=ENCODER_CONFIG["model_id"],
        batch_size=ENCODER_CONFIG["batch_size"],
        normalize_embeddings=ENCODER_CONFIG["normalize_embeddings"]
    )

    encoded_chunks_set_list = encoder.encode_chunks(chunks_set_list)

    save_encoded_chunks(
        encoded_chunks_set_list=encoded_chunks_set_list,
        save_path=PATHS_CONFIG["building"]["encoded_chunks"]
    )


def index_embeddings(embeddings: list[list[float]]) -> None:
    indexer = Indexer(
        type=INDEXER_CONFIG["type"],
        dimension=INDEXER_CONFIG["dimension"]
    )

    # strategy logic

    return indexer.index_embeddings(embeddings)

def run():
    model_config = load_config("configs/base_config.yaml")
    prompts_config = load_config("configs/prompts_config.yaml")

    files = model_config["paths"]["files"]
    directories = model_config["paths"]["directories"]

    # retriever configs
    chunker_config = model_config["chunking"]
    encoder_config = model_config["encoder"]
    indexer_config = model_config["indexer"]
    retriever_config = model_config["retriever"]

    # generator configs
    prompt_templates = prompts_config["template"]
    prompt_settings = prompts_config["settings"]
    generator_config = model_config["generator"]


    with mlflow.start_run():
        # initialize components
        encoder = Encoder(
            encoder_config["model"],
            encoder_config["batch_size"],
            encoder_config["normalize_embeddings"]
        )
        indexer = Indexer(
            indexer_config["type"],
            indexer_config["dimension"]
        )

        # chunk, encode, and index data
        chunks = chunk_recipes(
            strategy=chunker_config["strategy"],
            recipes_path=files["cleaned_recipes"]
        )
        embeddings = encoder.encode_chunks(chunks)
        indexer.index_embeddings(embeddings)

        # initialize retriever
        retriever = Retriever(
            chunks,
            encoder,
            indexer,
            retriever_config["top_k"],
            retriever_config["score_threshold"]
        )
        
        # unnecessary for now, since loaded generator is pre-trained
        """
        tokenizer = AutoTokenizer.from_pretrained(generator_config["model"])
        
        prompt_builder = PromptBuilder(
            tokenizer,
            prompt_config["max_context_tokens"],
            prompt_templates
        )
        generator = Generator(
            generator_config["model"],
            generator_config["adapter_path"]
        )
        
        # build RAG pipeline, not necessary for now
        pipeline = RAGPipeline(retriever, prompt_builder, generator)
        """
        # create snapshot
        snapshot = Snapshot(
            snapshot_id=None,
            created_at=None,
            index=retriever.indexer,
            chunks=chunks,
            base_config=model_config,
            prompts_config=prompts_config
        )

        # save snapshot
        save_snapshot(snapshot, directories["snapshots"])

        #TODO: Consider calling eval here.
        
        # encoder, tokenizer (in prompt builder) are pre-loaded, so no need to save

        # log params, artifacts
        mlflow.log_params(flatten_config_dict(model_config))
        mlflow.log_artifact(files["base_config"])
        mlflow.log_artifact(files["prompts_config"])

    return

if __name__ == "__main__":
    args = parse_args()
    if args.run == "chunk":
        chunk()
    elif args.run == "encode":
        encode_chunk()