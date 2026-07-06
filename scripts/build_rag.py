import mlflow
from src.retrieval.encoder import Encoder
from src.retrieval.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import Chunker
#from src.generation.prompt_builder import PromptBuilder
#from src.generation.generator import Generator
#from src.pipeline.rag_pipeline import RAGPipeline
#from transformers import AutoTokenizer
from src.data_classes import Snapshot
from src.utils import flatten_config_dict, save_config, load_config, save_snapshot
from pathlib import Path

#TODO: Add logging

# connect to remote MlFlow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-rag-experiment")

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
        chunker = Chunker(chunker_config["strategy"])
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
        chunks = chunker.get_chunks(files["cleaned_recipes"])
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
            chunks=retriever.chunks,
            base_config=model_config,
            prompts_config=prompts_config
        )

        # save snapshot
        save_snapshot(snapshot, directories["snapshots"])

        # encoder, tokenizer (in prompt builder) are pre-loaded, so no need to save

        # log params, artifacts
        mlflow.log_params(flatten_config_dict(model_config))
        mlflow.log_artifact(files["base_config"])
        mlflow.log_artifact(files["prompts_config"])

    return

if __name__ == "__main__":
    run()