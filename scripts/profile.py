from src.rag.core.profiler import profile_model_memory, profile_dataset_tokenization
from src.rag.core.utils import load_snapshot
import mlflow
import argparse

TARGET_MODEL = "google/gemma-3-1b-it"
INPUT_TOKENS = 8000

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("profiling")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=False, default=TARGET_MODEL)
    parser.add_argument("--input_tokens", type=int, required=False, default=INPUT_TOKENS)
    return parser.parse_args()

def run(model_id: str, input_tokens: int) -> None:
    # load chunks
    snapshot = load_snapshot("snapshots/latest")
    chunks = [c for s in snapshot.chunk_sets for c in s.chunks]
    
    with mlflow.start_run():
        
        memory_metrics =profile_model_memory(model_id, input_tokens)
        dataset_metrics = profile_dataset_tokenization(model_id, chunks)
        
        status = memory_metrics.pop("status")
        
        mlflow.set_tag("status", status)
        mlflow.log_param("input_tokens", input_tokens)

        mlflow.log_metrics(memory_metrics)
        mlflow.log_metrics(dataset_metrics)

if __name__ == "__main__":
    args = parse_args()
    run(args.model_id, args.input_tokens)