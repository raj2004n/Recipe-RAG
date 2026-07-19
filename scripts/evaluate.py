from typing import Sequence
import mlflow
from src.rag.core.utils import load_retriever, load_config, load_eval_dataset
from src.rag.evaluation.retriever_evaluator import evaluate_strict

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("evaluation-experiment") #TODO Should not be a new experiment. Should be the same experiment as the one used for building.

def evaluate_retriever(config_file: str = "models/retriever/config.yaml", evaluation_path: str = "data/evaluation/evaluation_dataset.json", ks: Sequence[int] = (1, 3, 5)):
    """ Evaluates a retriever on a dataset of queries and ground truths. """
    # load saved config
    config = load_config(config_file)
    
    with mlflow.start_run():
        retriever, _, _, _ = load_retriever(config)
        evaluation_dataset = load_eval_dataset(evaluation_path)

        results = evaluate_strict(
            retriever=retriever,
            evaluation_dataset=evaluation_dataset,
            ks=ks,
        )
        
        print("Strict Recall@k:")
        for metric, value in results.items():
            print(f"{metric}: {value}")

        mlflow.log_params(config)
        mlflow.log_artifact(config_file)
        mlflow.log_artifact(evaluation_path)
        mlflow.log_metrics(results)

if __name__ == "__main__":
    evaluate_retriever()