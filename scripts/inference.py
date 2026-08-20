import logging
from src.rag.core.snapshot.loader import load_snapshot, Snapshot
from src.rag.core.config import SNAPSHOT_PATHS, DATA_PATHS
from src.rag.core.logging import configure_logging
from src.rag.pipeline import DensePipeline, HybridPipeline

logger = logging.getLogger(__name__)


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
    snapshot = load_snapshot(str(SNAPSHOT_PATHS.latest))
    return _build_pipeline(snapshot)


if __name__ == "__main__":
    configure_logging()
    rag = run()

    logger.info("Recipe RAG System Ready!  (strategy: %s)", rag._snapshot.retrieval_strategy)
    logger.info("Type 'quit' to exit")

    while True:
        question = input("\nWhat would you like to cook? > ")
        if question.lower() in ['quit', 'exit', 'q']:
            break

        result = rag.query(question)
        logger.info("\U0001f4dd Answer:\n%s", result['answer'])
