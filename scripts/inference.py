from src.rag.core.snapshot.loader import load_snapshot, Snapshot
from src.rag.core.config import DATA_PATHS
from src.rag.pipeline import DensePipeline, HybridPipeline


# map snapshot retrieval_strategy to pipeline
_PIPELINES = {
    "dense": DensePipeline,
    "hybrid": HybridPipeline,
}


def _build_pipeline(snapshot: Snapshot):
    """Pick the right Pipeline class for `snapshot.retrieval_strategy`."""
    try:
        pipeline_cls = _PIPELINES[snapshot.retrieval_strategy]
    except KeyError as exc:
        raise ValueError(
            f"No pipeline registered for retrieval_strategy "
            f"{snapshot.retrieval_strategy!r}; "
            f"known: {sorted(_PIPELINES)}"
        ) from exc

    if pipeline_cls is HybridPipeline:
        return pipeline_cls(
            snapshot,
            recipes_path=DATA_PATHS["cleaned_recipes"],
        )
    return pipeline_cls(snapshot)


def run():
    # load snapshot
    snapshot = load_snapshot("snapshots/latest")
    return _build_pipeline(snapshot)


if __name__ == "__main__":
    rag = run()

    print(f"Recipe RAG System Ready!  (strategy: {rag._snapshot.retrieval_strategy})")
    print("Type 'quit' to exit")

    while True:
        question = input("\nWhat would you like to cook? > ")
        if question.lower() in ['quit', 'exit', 'q']:
            break

        result = rag.query(question)
        print(f"\n📝 Answer:\n{result['answer']}")
