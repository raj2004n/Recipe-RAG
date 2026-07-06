from pathlib import Path
from transformers import AutoTokenizer
from src.utils import load_snapshot
from src.retrieval.encoder import Encoder
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.pipeline.rag_pipeline import RAGPipeline
from src.data_classes import Snapshot

def run():
    # load snapshot
    snapshot = load_snapshot("snapshots/latest")
    rag = RAGPipeline(snapshot)
    return rag



if __name__ == "__main__":
    
    # Load the pre built system
    rag = run()
    
    print("Recipe RAG System Ready!")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nWhat would you like to cook? > ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = rag.query(question)
        print(f"\n📝 Answer:\n{result['answer']}")