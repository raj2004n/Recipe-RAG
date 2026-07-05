import yaml
import logging
from src.retrieval.encoder import Encoder
from src.retrieval.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator
from src.pipeline.rag_pipeline import RAGPipeline

def load_rag(config_path: str = "configs/base_config.yaml"):
    """Load pre-built RAG system from disk"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load encoder
    encoder = Encoder(config)
    
    # Load index
    indexer = Indexer(
        dimension=config["index"]["dimension"],
        index_type=config["index"]["type"]
    )
    
    # Load retriever
    retriever = Retriever.load(
        path="models/retriever",
        encoder=encoder,
        index=indexer,
        config=config
    )
    
    # Load generator (if exists, else None for retrieval-only mode)
    generator = None
    generator_path = "models/generator" # can add to config
    model_name = "microsoft/phi-2" # can add to config
    try:
        generator = Generator(config)
        generator.load(model_name, generator_path)
    except Exception as e:
        raise e
    
    # Create pipeline
    pipeline = RAGPipeline(retriever, generator)
    
    return pipeline


if __name__ == "__main__":
    
    # Load the pre built system
    pipeline = load_rag()
    
    print("Recipe RAG System Ready!")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nWhat would you like to cook? > ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = pipeline.query(question)
        print(f"\n📝 Answer:\n{result['answer']}")