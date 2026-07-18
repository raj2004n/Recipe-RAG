import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from functools import lru_cache

MODEL_ID = "google/gemma-3-4b-it"

# TODO: Still does not handle "Recipes" well.
# Some seasonings is interpretted as exclusion
SYSTEM_PROMPT = """
You are a query parser for a recipe retrieval system.

The user's input is a recipe search query.
Extract only explicit constraints from the query.

Your task:
1. Identify things the user wants (inclusions):
   - ingredients
   - dishes
   - cuisines
   - flavors
   - textures
   - cooking methods
   - dietary preferences
   - meal characteristics

2. Identify things the user wants to avoid (exclusions):
   - ingredients
   - flavors
   - textures
   - cooking methods
   - cuisines
   - dietary restrictions

Rules:
- Only extract information explicitly mentioned by the user.
- Do not infer missing ingredients.
- Do not infer cuisine from a dish unless explicitly stated.
- Do not add common recipe ingredients.
- Do not add meal types unless explicitly requested.
- Do not add dietary tags unless explicitly stated.
- Preserve negations.
- If nothing is specified, output "any" or "none".

Output format (strict):
Inclusions: <comma-separated list or "any">
Exclusions: <comma-separated list or "none">
"""

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-query_decomp-experiment")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def load_model():    
    # load base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

def decompose_query(query: str) -> str:
    """ LLM to decompose a single query into inclusion and exclusion."""
    model, tokenizer = load_model()
    
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
    
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False
            )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
    decomposed_query = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # free memory
    del generated_ids
    del model_inputs
    del model
    del tokenizer

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return decomposed_query

def decompose_queries(queries: list[str]):
    """ LLM to decompose query into multiple queries.
     - Dense retrievers do not handle negation. 
      - By decomposing query, can seperately search for negation query and remove those recipes. """
    
    model, tokenizer = load_model()

    results = {}
    
    for query in queries:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
        model_inputs = tokenizer(
            [text],
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False
            )

        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        results[query] = response
    
    for key, value in results.items():
        print(key)
        print(f"-"*50)
        print(value)
        print(f"="*50)

    with mlflow.start_run():
        mlflow.log_param("system_prompt", SYSTEM_PROMPT)
        mlflow.log_param("model_id", MODEL_ID)
        mlflow.log_dict(results, "results.json")

if __name__ == "__main__":
    queries = [
        "I want to make a quick chicken curry. But I do not have any onions, and I don't want anything spicy",
        "I want recipes that do not use meat",
        "Hmm I dont really know what i wanna eat. Tho it is very hot today so maybe consider that",
        "Recipes that do not use meat",
        "Recipes",
        "i wanna eat mexican food",
        "i wanna eat some dish from brazil",
        "i wanna eat something from jupiter",
        "i want the secret formula for the krabby patty",
        "I only have potatoes, some seasonings, and chicken stock. Give me some recipes that i can make with that"
        ]
    decomposed_query = decompose_query(queries[0])
    print(decomposed_query)
