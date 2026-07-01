from pathlib import Path
from src.evaluation.dataset_generator import generate_dataset

generate_dataset(
    recipes_file=Path("data/processed/cleaned_recipes.json"),
    output_file=Path("data/evaluation/evaluation_dataset.json"),
)