from pathlib import Path
from src.rag.data.eval_dataset_generator import generate_dataset
from src.rag.core.config import DATA_PATHS, EVAL_DATASET_CONFIG


def run() -> None:
    generate_dataset(
        recipes_path=Path(DATA_PATHS["cleaned_recipes"]),
        output_path=Path(DATA_PATHS["eval_llm_generated"]),
        model_name=EVAL_DATASET_CONFIG["model"],
        max_workers=EVAL_DATASET_CONFIG["max_workers"],
        request_timeout=EVAL_DATASET_CONFIG["request_timeout"],
        max_retries=EVAL_DATASET_CONFIG["max_retries"],
        backoff_base=EVAL_DATASET_CONFIG["backoff_base"]
    )

if __name__ == "__main__":
    run()