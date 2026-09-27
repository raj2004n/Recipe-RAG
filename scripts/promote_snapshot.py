"""The promotion gate: the script that decides the deployment baseline.

Reads a candidate snapshot's end-to-end MLflow run, applies the bars from
configs/promotion.yaml (slice means, per-query-type floors, judge-disagreement
caps, and the structure and retrieval rates), prints a pass/fail report whose
header carries the sample-size caveat, and only on explicit human approval
records the promotion in the snapshot registry: targets["baseline"] plus a
promoted record with the governing run's identity and the per-check outcome.

It never promotes automatically. A failed gate exits non-zero and offers no
approval. Promotion is a registry write, not a deployment; carrying the
baseline to the GPU host is a separate later step.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow

from src.rag.core.logging import configure_logging
from src.rag.core.snapshot import registry
from src.rag.core.snapshot.paths import SnapshotPaths
from src.rag.evaluation.harness import MLFLOW_TRACKING_URI, resolve_snapshot_folder
from src.rag.evaluation.promotion import (
    PromotionConfigError,
    evaluate,
    load_promotion_config,
)


def _default_config_path() -> Path:
    """configs/promotion.yaml at the repo root, next to this script."""
    return Path(__file__).resolve().parents[1] / "configs" / "promotion.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Promotion gate for snapshots")
    parser.add_argument(
        "--snapshot", type=str, default="latest",
        help="Snapshot target ('latest' or 'current') or folder name to gate.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to the criteria file (default configs/promotion.yaml).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Approve the promotion non-interactively. The human decision is still yours; this only skips the prompt.",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Print the report and stop; never write to the registry.",
    )
    return parser.parse_args()


def find_latest_run(folder: str, experiment: str, marker_param: str):
    """The most recent end-to-end MLflow run for the snapshot, or None.

    Runs are marked with the snapshot folder name under marker_param, so the
    candidate is found by filtering the experiment on that param and taking
    the newest by start time. The gate must fail closed when the tracking
    store is unreachable or the experiment has no recorded runs. When the
    tracking URI is an HTTP server, the client is bounded to a short timeout
    and a single retry instead of mlflow's default no-timeout backoff, so a
    down server fails fast as a readable error rather than hanging.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    try:
        experiment_obj = mlflow.get_experiment_by_name(experiment)
    except Exception as exc:
        raise RuntimeError(_server_missing_message(experiment)) from exc
    if experiment_obj is None:
        raise RuntimeError(_server_missing_message(experiment))
    runs = mlflow.search_runs(
        experiment_ids=[experiment_obj.experiment_id],
        filter_string=f"params.{marker_param} = '{folder}'",
        output_format="list",
    )
    if not runs:
        return None
    return max(runs, key=lambda run: run.info.start_time or 0)


def _server_missing_message(experiment: str) -> str:
    return (
        f"could not reach the tracking store or experiment {experiment!r} has "
        f"no recorded runs. The default store is sqlite:///mlflow.db (override "
        f"with RECIPE_RAG_MLFLOW_TRACKING_URI if you use an HTTP server). Run "
        f"scripts.evaluate_end_to_end --snapshot <target> first to record a "
        f"run in this experiment."
    )


def _format_value(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_report(folder: str, run, gate: dict, checks: list) -> None:
    """The gate's human-readable report: header, checks, verdict."""
    caveat = gate["sample_size_caveat"]
    params = dict(run.data.params)
    print(f"Promotion gate for snapshot: {folder}")
    print(f"  run name:      {run.info.run_name or '(unnamed)'}")
    print(f"  run id:        {run.info.run_id}")
    print(f"  rows:          {params.get('rows', 'n/a')}")
    if params.get("limit"):
        print("  WARNING: this run was bounded by --limit; it is not a full golden read")
    if params.get("judge_model_id"):
        print(f"  judge:         {params.get('judge_model_id')} "
              f"(temp {params.get('judge_temperature', 'n/a')})")
    if params.get("generator_model_id"):
        adapter = params.get("generator_adapter_path") or "none"
        print(f"  generator:     {params.get('generator_model_id')} "
              f"(adapter {adapter})")
    print(f"  caveat:        {caveat}")
    print()

    current_scope = None
    for check in checks:
        if check.scope != current_scope:
            current_scope = check.scope
            heading = "slice" if current_scope == "slice" else current_scope.removeprefix("query_type:")
            print(f"[{heading}]")
        mark = "PASS" if check.passed else "FAIL"
        print(
            f"  {check.metric:<36} {_format_value(check.value):>7} "
            f"{check.operator} {check.bar:.2f}?  {mark}"
        )
    print()


def main():
    configure_logging()
    args = parse_args()

    config_path = Path(args.config) if args.config else _default_config_path()
    try:
        config = load_promotion_config(config_path)
    except PromotionConfigError as exc:
        print(f"error: {exc}")
        sys.exit(2)

    try:
        folder = resolve_snapshot_folder(args.snapshot)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        sys.exit(2)

    paths = SnapshotPaths.from_paths_config({})
    status = registry.get_status_for_folder(paths.registry_file, folder)
    if status != "complete":
        print(f"error: snapshot {folder!r} has status {status!r}, not 'complete'")
        sys.exit(2)

    gate = config["gate"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        run = find_latest_run(folder, gate["experiment"], gate["run_marker_param"])
    except RuntimeError as exc:
        print(f"error: {exc}")
        sys.exit(2)

    if run is None:
        print(
            f"error: no {gate['experiment']!r} run found for snapshot {folder!r}. "
            f"Run scripts.evaluate_end_to_end --snapshot {folder} first."
        )
        sys.exit(2)

    params = dict(run.data.params)
    if params.get("limit") is not None:
        print(
            f"error: the run for {folder!r} was bounded by --limit (rows={params.get('rows', 'n/a')}); "
            f"re-run the evaluation without --limit for a promotable read."
        )
        sys.exit(2)

    checks = evaluate(dict(run.data.metrics), config)
    print_report(folder, run, gate, checks)

    passed = all(check.passed for check in checks)
    if not passed:
        print(f"[VERDICT] FAIL - snapshot {folder!r} did not clear the bars; nothing promoted.")
        sys.exit(1)

    print(f"[VERDICT] PASS - snapshot {folder!r} cleared the bars.")
    if args.report_only:
        print("--report-only set: no registry write; snapshot not promoted.")
        return

    if not args.yes:
        answer = input(f"Promote {folder!r} as the deployment baseline? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Not promoted (no approval). The registry is unchanged.")
            return

    registry.record_promotion(
        paths.registry_file,
        folder,
        run_id=run.info.run_id,
        run_name=run.info.run_name or "",
        checks=[check.as_dict() for check in checks],
    )
    print(f"Promoted {folder!r}: targets.baseline now points at it, recorded in the registry.")


if __name__ == "__main__":
    main()
