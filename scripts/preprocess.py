from src.rag.core.utils import load_config
from src.rag.data.preprocessor import preprocess

def run():
    config = load_config("configs/paths_config.yaml")
    files = config["files"]

    preprocess(
        input_path=files["raw_recipes"],
        output_path=files["cleaned_recipes"]
    )

if __name__ == "__main__":
    run()