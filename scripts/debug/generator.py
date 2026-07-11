import argparse
import mlflow
from src.utils import load_snapshot, flatten_config_dict
from src.pipeline.rag_pipeline import RAGPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

""" 
Idea:
Retriever gather relevant chunks (potentially from different recipes)
The generator can piece together the chunks, basically; here is the relevant pieces, now make a recipe.
"""
HARD_PROMPT = "How do I make a chicken curry, without using any meat?" # does not answer well to this prompt. Outputs multiple recipes, and uses meat.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=False, default="How do I make a chicken curry?") 
    parser.add_argument("--top_k", type=int, required=False, default=3)
    return parser.parse_args()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-generator-experiment")

def debug_generator(query: str, top_k: int):
    # load snapshot
    snapshot = load_snapshot("snapshots/latest")

    # modify generator
    gen_config = snapshot.base_config["generator"]
    gen_config["model"] = "google/gemma-3-1b-it" # max input tokens 32k, max new tokens 8192
    gen_config["max_input_tokens"] = 5000
    gen_config["max_new_tokens"] = 2000
    gen_config["epochs"] = 3
    gen_config["adapter_path"] = None
    gen_config["truncation"] = True
    gen_config["return_tensors"] = "pt"

    # overwrite snapshot to alter generator
    snapshot.base_config["generator"] = gen_config

    prompts_settings = snapshot.prompts_config["settings"]
    prompts_settings["max_context_tokens"] = 4000
    snapshot.prompts_config["settings"] = prompts_settings

    # TODO: if too low, then say that I don't have a suitable recipe for that.
    # TODO: Why does it sometimes give nutrients. Evaluate consistecny of the responses.
    generation_prompt = snapshot.prompts_config["template"]["generation"]
    generation_prompt = """Use the RELEVANT RECIPES to answer USER QUERY.
    
    RELEVANT RECIPES: {context}

    USER QUERY: {query}

    Rules:
    - If generating a recipe, provide a list of ingreidents, instructions, and tips.
    - Do not state anything about being provided recipes.
    """

    snapshot.prompts_config["template"]["generation"] = generation_prompt
    
    with mlflow.start_run():
        
        # build rag pipeline with loaded retriever, and new generator
        rag = RAGPipeline(snapshot)

        results, response, prompt = rag.debug_query(query, top_k)

        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(gen_config["model"])
        model = AutoModelForCausalLM.from_pretrained(
            gen_config["model"],
            torch_dtype="auto",
            device_map="auto"
        )

        # prepare the model input
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # conduct text completion
        gen_params = {
            "max_new_tokens": gen_config["max_new_tokens"],
            "do_sample": False,
        }

        generated_ids = model.generate(
            **model_inputs,
            **gen_params
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True) # can set special tokens to False. Handly for debugging

        mlflow.log_param("query", query)
        mlflow.log_param("top_k", top_k)
        mlflow.log_text(prompt, "prompt.txt")
        mlflow.log_text(content, "response.txt")
        
        mlflow.log_params(gen_params)
        mlflow.log_params(flatten_config_dict(snapshot.base_config))
        
        mlflow.log_artifact(snapshot.base_config["paths"]["files"]["base_config"])
        mlflow.log_artifact(snapshot.base_config["paths"]["files"]["prompts_config"])
        mlflow.log_artifact(snapshot.base_config["paths"]["files"]["indexer"])
        mlflow.log_artifact(snapshot.base_config["paths"]["files"]["chunks"])

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
    print(f"Response: \n{content}")
    print(f"="*50)

if __name__ == "__main__":
    args = parse_args()
    debug_generator(args.query, args.top_k)