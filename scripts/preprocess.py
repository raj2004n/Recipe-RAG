from src.utils import load_config
from src.data.preprocessor import preprocess

def run():
    config = load_config()
    paths = config["paths"]

    preprocess(
        input_path=paths["raw_recipes"],
        output_path=paths["cleaned_recipes"]
    )

if __name__ == "__main__":
    run()