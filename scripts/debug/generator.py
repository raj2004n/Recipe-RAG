import argparse
import time
from src.generation.prompt_builder import PromptBuilder
from src.utils import load_config, load_retriever, build_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False, default="How do I make chicken curry?")
    parser.add_argument("--top_k", type=int, required=False, default=5)
    return parser.parse_args()

def debug_generator(query: str, top_k: int = 5):
    config = load_config()

    retriever, _, _, _ = load_retriever(config)
    generator = build_generator(config)

    prompt_builder = PromptBuilder(config, generator.tokenizer)

    results = retriever.retrieve(query, top_k)
    
    prompt = prompt_builder.build(query, results)

    print(f"="*50)
    print(f"Query: {query}")
    print(f"="*50)

    # print etrieved response
    print(f"Retrieved Results:")
    for r in results:
        print(f"Recipe ID: {r.recipe_id}, Chunk ID: {r.chunk_id}, Score: {r.score}")
        print(f"Recipe Name: {r.chunk['metadata']['recipe_name']}")
        print(f"-"*20)
        #print(f"Text: {r.chunk['text']}")

    print(f"="*50)
    print(f"Prompt: \n{prompt}")
    print(f"="*50)

    # generate response
    start = time.time()
    response = generator.generate(prompt)
    end = time.time()

    print(f"Response: \n{response}")
    print(f"-"*50)
    print(f"Time taken: {end - start}")

if __name__ == "__main__":
    args = parse_args()
    debug_generator(args.query, args.top_k)