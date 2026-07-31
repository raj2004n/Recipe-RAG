from src.rag.core.snapshot import load_snapshot
from src.rag.pipeline.rag_pipeline import RAGPipeline

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