import mlflow
import pprint
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache


MODEL_ID = "google/gemma-3-1b-it"

SYSTEM_PROMPT = """
Extract what the user wants and does not want from the query. 
If what the user does not want is not explicitly mentioned, exclude nothing.

Output as such:
Include: ...

Exclude: ...

"""

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("first-query_decomp-experiment")

@lru_cache()
def load_model():    
    # load base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def decompose_query(queries: list[str]):
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
                max_new_tokens=128,
                do_sample=False
            )

        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        results[query] = response
    
    pprint.pprint(results)
    with mlflow.start_run():
        mlflow.log_param("system_prompt", SYSTEM_PROMPT)
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
        ]
    decompose_query(queries)