""" 
Test script for dense strategy inference. Uses dense retriever.
"""
from src.rag.core.snapshot import load_snapshot
from src.rag.pipeline.dense_pipeline import DensePipeline

def run():
    # load snapshot
    snapshot = load_snapshot("temp/latest")
    rag = DensePipeline(snapshot)
    return rag

if __name__ == "__main__":
    
    # load the pre built system
    rag = run()
    
    print("Recipe RAG System Ready!")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nWhat would you like to cook? > ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        result = rag.query(question)
        print(f"\n📝 Answer:\n{result['answer']}")