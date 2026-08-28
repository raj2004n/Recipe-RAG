from pathlib import Path

from src.rag.core.config import DATA_PATHS
from src.rag.evaluation.quality_reporter import (
    format_table,
    report_quality,
    write_json,
)


def run() -> int:
    """Print the dataset-quality report to stdout and write it as JSON.

    Returns 0 on success. The corpus file must exist (see _corpus_size); a
    missing corpus raises FileNotFoundError and the script exits non-zero
    because the coverage metric would otherwise be wrong.
    """
    report = report_quality(
        golden_path=DATA_PATHS["eval_golden_prompts"],
        llm_generated_path=DATA_PATHS["eval_llm_generated"],
        corpus_path=DATA_PATHS["eval_corpus_recipes"],
    )
    print(format_table(report), end="")
    output_path = Path(DATA_PATHS["eval_golden_prompts"]).parent / ".quality_report.json"
    write_json(report, str(output_path))
    print(f"  Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
