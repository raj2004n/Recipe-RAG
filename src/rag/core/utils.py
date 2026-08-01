import yaml
import json
from pathlib import Path

def load_config(config_file: str = "configs/base_config.yaml") -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    return dataset

def flatten_config_dict(d, parent_key="", sep="."):
    items = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_config_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v

    return items