from pathlib import Path
from src.evaluation.dataset_generator import generate_dataset
from src.utils import load_config

def run() -> None:
    config = load_config("configs/base_config.yaml")

    paths = config["paths"]
    eval_config = config["evaluation dataset generator"]

    generate_dataset(
        recipes_path=Path(paths["cleaned_recipes"]),
        output_path=Path(paths["evaluation_queries"]),
        model_name=eval_config["model"],
        max_workers=eval_config["max_workers"],
        request_timeout=eval_config["request_timeout"],
        max_retries=eval_config["max_retries"], 
        backoff_base=eval_config["backoff_base"]
    )

if __name__ == "__main__":
    run()