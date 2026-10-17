from src.core.config import DATA_PATHS
from src.corpus.preprocessor import preprocess

def run():
    preprocess(
        input_path=DATA_PATHS["raw_recipes"],
        output_path=DATA_PATHS["cleaned_recipes"]
    )

if __name__ == "__main__":
    run()