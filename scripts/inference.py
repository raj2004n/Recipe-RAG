from pathlib import Path
from transformers import AutoTokenizer
from src.utils import load_config, load_chunks, load_indexer
from src.retrieval.encoder import Encoder
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.pipeline.rag_pipeline import RAGPipeline

def run():
    # load config
    config = load_config("models/retriever/config.yaml")
    prompts_config = load_config("models/retriever/prompts_config.yaml")

    encoder_config = config["encoder"]
    retriever_config = config["retriever"]
    generator_config = config["generator"]

    prompt_templates = prompts_config["template"]
    prompt_settings = prompts_config["settings"]

    # load paths
    paths = config["paths"]

    # load components (initialze Encoder since pre trained hence was not saved)
    chunks = load_chunks(Path(paths["files"]["chunks"]))
    encoder = Encoder(
        encoder_config["model"],
        encoder_config["batch_size"],
        encoder_config["normalize_embeddings"]
    )
    indexer = load_indexer(Path(paths["files"]["indexer"]))

    # initialize retriever
    retriever = Retriever(
        chunks,
        encoder,
        indexer,
        retriever_config["top_k"],
        retriever_config["score_threshold"]
    )
    # initialize generator
    generator = Generator(
        generator_config["model"],
        generator_config["adapter_path"]
    )
    # initialize prompt builder
    tokenizer = AutoTokenizer.from_pretrained(generator_config["model"]) 
    prompt_buidler = PromptBuilder(
        tokenizer=tokenizer,
        max_context_length=prompt_settings["max_context_tokens"],
        template=prompt_templates
    )

    # build RAG pipeline
    pipeline = RAGPipeline(retriever, prompt_buidler, generator)
    return pipeline



if __name__ == "__main__":
    
    # Load the pre built system
    pipeline = run()
    
    print("Recipe RAG System Ready!")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nWhat would you like to cook? > ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = pipeline.query(question)
        print(f"\n📝 Answer:\n{result['answer']}")