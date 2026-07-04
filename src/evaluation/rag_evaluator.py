from src.utils import load_retriever, load_config, load_eval_dataset, build_generator
from src.prompting.prompt_builder import PromptBuilder
from src.models.generator import Generator
from src.models.retriever import Retriever
from transformers import AutoTokenizer
from typing import Sequence
from ollama import chat
import json

JUDGE_MODEL = "gemma3:4b"

def judge(recipe: dict, query: str, response: str, config_file: str = "models/retriever/config.yaml"):
    """ Judges a response against a recipe. """
    # load config
    config = load_config(config_file)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["generator"]["model_name"])

    # load prompt builder
    prompt_builder = PromptBuilder(config, tokenizer)

    # build judge prompt
    prompt = prompt_builder.build_judge_prompt(recipe, query, response)

    response = chat(JUDGE_MODEL, messages=[{"role": "user", "content": prompt}])

    content = response["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "query_relevance": -1,
            "faithfulness": -1,
            "error": content
        }

def evaluate_rag(retriever: Retriever,
                 config_file: str = "models/retriever/config.yaml",
                 evaluation_path: str = "data/evaluation/evaluation_dataset.json",
                 ks: Sequence[int] = (1, 3, 5)
                 ):

    results = []
    aggregate_scores = {
        "query_relevance": [],
        "faithfulness": []
    }

    # load config
    config = load_config(config_file)

    # load evaluation dataset
    evaluation_dataset = load_eval_dataset(evaluation_path)

    for i, eval in enumerate(evaluation_dataset):
        query = eval["query"]
        ground_truth_id = eval["recipe_id"]

        retrieved_results= retriever.retrieve(query, top_k=1) # retrieve only one result

        context = retrieved_results[0].chunk["text"]

        # build generator
        generator = build_generator(config)
        # build tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["generator"]["model_name"])
        
        # build prompt builder
        prompt_builder = PromptBuilder(config, tokenizer)
        
        # build prompt
        prompt = prompt_builder.build(query, retrieved_results)

        # generate response
        response = generator.generate(prompt)
        
        # judge response
        judge_result = judge(context, query, response, config_file)

        results.append(judge_result)

    return results