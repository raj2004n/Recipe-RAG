from src.profiler import profile_model_memory, profile_dataset_tokenization
from src.utils import load_snapshot
import mlflow

TARGET_MODEL = "google/gemma-3-1b-it"
INPUT_TOKENS = 4096

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("profiling")

def run() -> None:
    # load chunks
    snapshot = load_snapshot("snapshots/latest")
    chunks = snapshot.chunks
    
    with mlflow.start_run():
        
        memory_metrics =profile_model_memory(TARGET_MODEL, INPUT_TOKENS)
        dataset_metrics = profile_dataset_tokenization(TARGET_MODEL, chunks)

        mlflow.log_metrics(memory_metrics)
        mlflow.log_metrics(dataset_metrics)

if __name__ == "__main__":
    run()