"""Rank student adapters by having the judge read what they write.

A trained adapter's loss says how well it predicts the teacher's words, not whether the recipes it
writes are any good. This script answers the second question for a handful of candidate adapters by
making them write, on the same held-out rows, from the same prompt, and having the judge read every
answer with the two personas the rest of the evaluation already uses.

The prompt is taken straight from the tokenised rows the trainer read, as the ids the row was built
with, so nothing is re-rendered and no tokenisation drift can creep in between training and judging.
Each adapter writes greedily from its own copy of the base checkpoint, the answers are then read in
parallel, and the run ends with a table ranking the adapters by their mean core score alongside the
paired count of rows one adapter won over the other. A margin under ``TIE_MARGIN`` is reported as a
tie rather than resolved, because picking a winner on a difference the judge cannot see is worse
than saying the two are level.

``--probe`` is the cheap way in: it writes nothing, generates for a handful of rows on one adapter,
and reports the measured seconds a row and peak memory so the full sweep can be projected from real
numbers instead of an estimate.
"""

import argparse
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.train_student import EMPTY_REASONING
from src.rag.core.logging import configure_logging
from src.rag.data.pairs import file_checksum, write_manifest
from src.rag.evaluation.generator_evaluator import structural_checks
from src.rag.evaluation.harness import begin_mlflow_run, build_judge, judge_config
from src.rag.evaluation.scoring import judgment

logger = logging.getLogger(__name__)

STUDENT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
TOKENISED_ROWS = Path("data/pairs/tokenised_rows_20260919.pt")
PROMPTS_FILE = Path("data/pairs/recipe_prompts_20260918.jsonl")
ADAPTERS_ROOT = Path("artifacts") / "student"
ARTIFACTS_DIR = Path("artifacts") / "judge"
MLFLOW_EXPERIMENT = "rag-student-judge"

DEFAULT_SYSTEMS = "epoch_01,epoch_02"
MAX_NEW_TOKENS = 1024
JUDGE_CALLS = 2
TIE_MARGIN = 0.05
PROBE_ROWS = 8
ASSISTANT_OPEN = "<|im_start|>assistant\n"
USER_OPEN = "<|im_start|>user\n"


def stamp() -> str:
    """The run's name, in the same shape the trainer stamps its own runs with."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def load_rows(split: str, limit: int | None) -> list[dict]:
    """The tokenised rows for one split, in the order the split file wrote them."""
    payload = torch.load(TOKENISED_ROWS, weights_only=False)
    rows = [r for r in payload["rows"] if r["split"] == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no rows for split {split!r}")
    return rows


def load_requests() -> dict[tuple[int, int], str]:
    """The request text per (primary, partner) pair, which is what the judge is shown as the query."""
    requests: dict[tuple[int, int], str] = {}
    with PROMPTS_FILE.open() as handle:
        for line in handle:
            row = json.loads(line)
            requests[(row["primary_id"], row["partner_id"])] = row["prompt"]
    return requests


def user_turn(rendered: str) -> str:
    """The user message out of a rendered prompt, which carries both recipes."""
    start = rendered.rindex(USER_OPEN) + len(USER_OPEN)
    end = rendered.rindex(ASSISTANT_OPEN)
    return rendered[start:end].strip()


def teacher_answer(tokenizer, labels: torch.Tensor, masked: int) -> str:
    """The teacher's answer for a row, with the empty reasoning block and the turn end removed."""
    text = tokenizer.decode(labels[masked:].tolist())
    if text.startswith(EMPTY_REASONING):
        text = text[len(EMPTY_REASONING):]
    return text.removesuffix("<|im_end|>").strip()


def adapter_dir(system: str) -> Path:
    """The newest run folder that holds an adapter for a system name, so a rerun needs no edit."""
    candidates = sorted(
        folder / system
        for folder in ADAPTERS_ROOT.glob("*")
        if (folder / system / "adapter_config.json").is_file()
    )
    if not candidates:
        raise SystemExit(f"no adapter folder for {system!r} under {ADAPTERS_ROOT}")
    return candidates[-1]


def build_student(adapter: Path | None, device: torch.device):
    """The base checkpoint in bf16, with an adapter on top when one is named."""
    base = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, dtype=torch.bfloat16)
    base.to(device)
    base.eval()
    if adapter is not None:
        base = PeftModel.from_pretrained(base, adapter)
        base.eval()
    return base


def write_answer(model, tokenizer, prompt_ids: torch.Tensor, device: torch.device) -> dict:
    """One greedy answer from the prompt's own token ids, plus how it was produced.

    Greedy rather than sampled on purpose: the teacher wrote its answers at temperature zero, and a
    comparison between adapters should not be able to come down to which one drew a luckier sample.
    Decoding stops at the turn end token or at the ceiling, and a response that reached the ceiling
    is flagged, since a truncated answer is judged as written.
    """
    inputs = prompt_ids.to(device).long().unsqueeze(0)
    attention = torch.ones_like(inputs)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=attention,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    seconds = time.monotonic() - started
    new_ids = output[0, inputs.shape[1]:]
    text = tokenizer.decode(new_ids.tolist()).removesuffix("<|im_end|>").strip()
    return {
        "response": text,
        "new_tokens": int(new_ids.shape[0]),
        "seconds": round(seconds, 2),
        "hit_ceiling": bool(new_ids.shape[0] >= MAX_NEW_TOKENS),
    }


def judge_one(client, task: dict) -> dict:
    """Every judge call for one answer: both personas, called the configured number of times."""
    core = [client.score_core(task["query"], task["context"], task["response"])
            for _ in range(JUDGE_CALLS)]
    expected = [
        client.score_expectedness(task["query"], task["reference"], task["response"])
        for _ in range(JUDGE_CALLS)
    ] if task["reference"] else []
    scores, disagreement = judgment(core, expected)
    return {
        "core_judge": [v.model_dump() if v else None for v in core],
        "expectedness_judge": [v.model_dump() if v else None for v in expected],
        "scores": scores,
        "disagreement": disagreement,
    }


def core_total(scores: dict) -> float | None:
    """The core persona's three dimensions added up, or None if none of them scored."""
    parts = [scores.get(name) for name in
             ("constraint_satisfaction", "query_relevance", "faithfulness")]
    present = [p for p in parts if p is not None]
    if not present:
        return None
    return sum(present)


def summarise(system: str, records: list[dict]) -> dict:
    """One adapter's row of the result table."""
    answer_records = [r for r in records if r["system"] == system]
    totals = [core_total(r["scores"]) for r in answer_records]
    present = [t for t in totals if t is not None]
    dimensions = ("constraint_satisfaction", "query_relevance", "faithfulness", "expectedness")
    disagreement = {}
    for name in dimensions:
        flags = [r["disagreement"][name] for r in answer_records
                 if r["disagreement"][name] is not None]
        if flags:
            disagreement[name] = sum(flags) / len(flags)
    return {
        "system": system,
        "rows": len(answer_records),
        "scored_rows": len(present),
        "core_total": sum(present) / len(present) if present else None,
        "core_stdev": statistics.pstdev(present) if len(present) > 1 else None,
        "dimensions": {
            name: (
                sum(r["scores"][name] for r in answer_records if r["scores"][name] is not None)
                / max(1, sum(1 for r in answer_records if r["scores"][name] is not None))
            )
            for name in dimensions
        },
        "disagreement": disagreement,
        "structure_pass_rate": (
            sum(1 for r in answer_records if r["structural"]["structure_pass"])
            / len(answer_records) if answer_records else None
        ),
        "mean_new_tokens": (
            sum(r["new_tokens"] for r in answer_records) / len(answer_records)
            if answer_records else None
        ),
        "ceiling_hits": sum(1 for r in answer_records if r["hit_ceiling"]),
        "mean_seconds": (
            sum(r["seconds"] for r in answer_records) / len(answer_records)
            if answer_records else None
        ),
    }


def head_to_head(records: list[dict], systems: list[str]) -> dict:
    """How often each system beat the other on the same row, which a mean can hide.

    Rows where either side failed to score are left out rather than counted as a draw, so the count
    is over rows the judge actually read twice.
    """
    by_row: dict[int, dict[str, float]] = {}
    for record in records:
        total = core_total(record["scores"])
        if total is not None:
            by_row.setdefault(record["row_id"], {})[record["system"]] = total
    wins = {system: 0 for system in systems}
    decided = 0
    for row_scores in by_row.values():
        if len(row_scores) < len(systems):
            continue
        ordered = sorted(systems, key=lambda s: row_scores[s], reverse=True)
        top = row_scores[ordered[0]]
        second = row_scores[ordered[1]]
        if top == second:
            continue
        decided += 1
        wins[ordered[0]] += 1
    return {"decided_rows": decided, "wins": wins}


def print_probe(measurements: list[dict], rows_total: int, systems: int) -> None:
    """What a handful of real rows says about the full sweep."""
    tokens = sum(m["new_tokens"] for m in measurements)
    seconds = sum(m["seconds"] for m in measurements)
    peak = max(m["peak_gib"] for m in measurements)
    per_row = seconds / len(measurements)
    print(f"probe: {len(measurements)} rows, {tokens} new tokens, {seconds:.1f}s, "
          f"peak {peak:.2f} GiB")
    print(f"probe: {per_row:.2f}s a row, {tokens / seconds:.1f} new tokens a second")
    print(f"probe: {rows_total} rows x {systems} systems projects to "
          f"{per_row * rows_total * systems / 60:.0f} minutes of generation")
    print(f"probe: judging is {rows_total * systems * 4} calls, not measured here")
    print(f"probe: first measurement carried {measurements[0]['seconds']:.1f}s, "
          "which is where the kernels compile")


def run_probe(args, rows: list[dict], tokenizer, device: torch.device) -> None:
    """Time generation on a few rows and write nothing."""
    system = args.systems[0]
    adapter = adapter_dir(system)
    print(f"probe: adapter {adapter}")
    model = build_student(adapter, device)
    measurements = []
    for row in rows[:args.probe_rows]:
        masked = int(row["n_masked_tokens"])
        torch.cuda.reset_peak_memory_stats(device)
        answer = write_answer(model, tokenizer, row["input_ids"][:masked], device)
        answer["peak_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        measurements.append(answer)
        print(f"probe: row {row['row_id']} prompt {masked} new {answer['new_tokens']} "
              f"{answer['seconds']:.2f}s peak {answer['peak_gib']:.2f} GiB")
    print_probe(measurements, len(rows), len(args.systems))
    del model
    torch.cuda.empty_cache()


def generate_all(args, rows: list[dict], requests: dict, tokenizer, device) -> list[dict]:
    """Every system writes for every row, one system at a time so one model is on the card."""
    records = []
    for system in args.systems:
        adapter = None if system == "base" else adapter_dir(system)
        logger.info("generating with %s from %s", system, adapter or "the base checkpoint")
        model = build_student(adapter, device)
        for index, row in enumerate(rows, start=1):
            masked = int(row["n_masked_tokens"])
            answer = write_answer(model, tokenizer, row["input_ids"][:masked], device)
            records.append({
                "row_id": row["row_id"],
                "system": system,
                "primary_id": row["primary_id"],
                "partner_id": row["partner_id"],
                "primary_category": row["primary_category"],
                "prompt_tokens": masked,
                **answer,
            })
            if index % 10 == 0 or index == len(rows):
                logger.info("%s: %d/%d rows written", system, index, len(rows))
        del model
        torch.cuda.empty_cache()
    return records


def judge_all(args, rows: list[dict], records: list[dict], requests: dict, tokenizer) -> list[dict]:
    """Both personas read every answer, on as many threads as were asked for."""
    by_id = {row["row_id"]: row for row in rows}
    tasks = []
    for record in records:
        row = by_id[record["row_id"]]
        masked = int(row["n_masked_tokens"])
        rendered = tokenizer.decode(row["input_ids"][:masked].tolist())
        context = user_turn(rendered)
        tasks.append({
            "row_id": record["row_id"],
            "system": record["system"],
            "query": requests[(row["primary_id"], row["partner_id"])],
            "context": context,
            "reference": teacher_answer(tokenizer, row["labels"], masked),
            "response": record["response"],
        })
    client = build_judge(judge_config())
    logger.info("judging %d answers with %d workers", len(tasks), args.judge_workers)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
        verdicts = list(pool.map(lambda task: judge_one(client, task), tasks))
    logger.info("judging took %.1f minutes", (time.monotonic() - started) / 60)
    judged = []
    for record, task, verdict in zip(records, tasks, verdicts):
        judged.append({
            **record,
            "scores": verdict["scores"],
            "disagreement": verdict["disagreement"],
            "core_judge": verdict["core_judge"],
            "expectedness_judge": verdict["expectedness_judge"],
            "structural": structural_checks(task["response"], [task["context"]]),
            "reference_answer": task["reference"],
        })
    return judged


def rank(summaries: list[dict]) -> dict:
    """The table's verdict, with a margin too small to call reported as a tie."""
    scored = [s for s in summaries if s["core_total"] is not None]
    scored.sort(key=lambda s: s["core_total"], reverse=True)
    if not scored:
        return {"winner": None, "tie": None, "order": []}
    winner = scored[0]
    tie = None
    if len(scored) > 1 and winner["core_total"] - scored[1]["core_total"] < TIE_MARGIN:
        tie = [winner["system"], scored[1]["system"]]
    return {
        "winner": winner["system"] if tie is None else None,
        "tie": tie,
        "margin": (
            winner["core_total"] - scored[1]["core_total"] if len(scored) > 1 else None
        ),
        "margin_for_tie": TIE_MARGIN,
        "order": [s["system"] for s in scored],
    }


def print_table(summaries: list[dict], verdict: dict, head: dict) -> None:
    """The result of the sweep, in the terminal where the run was started."""
    print()
    print(f"{'system':<12}{'core':>7}{'sd':>7}{'expected':>10}{'struct':>8}"
          f"{'ceiling':>9}{'rows':>7}")
    for summary in summaries:
        core = f"{summary['core_total']:.3f}" if summary["core_total"] is not None else "n/a"
        spread = f"{summary['core_stdev']:.3f}" if summary["core_stdev"] is not None else "n/a"
        expected = summary["dimensions"]["expectedness"]
        struct = summary["structure_pass_rate"]
        print(f"{summary['system']:<12}{core:>7}{spread:>7}"
              f"{(f'{expected:.3f}' if expected is not None else 'n/a'):>10}"
              f"{(f'{struct:.3f}' if struct is not None else 'n/a'):>8}"
              f"{summary['ceiling_hits']:>9}{summary['rows']:>7}")
    if verdict["tie"]:
        print(f"\ntie: {verdict['tie'][0]} and {verdict['tie'][1]} are within "
              f"{verdict['margin_for_tie']} of each other, margin {verdict['margin']:.4f}")
    elif verdict["winner"]:
        print(f"\nwinner: {verdict['winner']}, margin {verdict['margin']:.4f}")
    print(f"head to head: {head['wins']} over {head['decided_rows']} decided rows")


def run_judge(args) -> int:
    """Write the sweep out, log it, and leave a manifest behind."""
    if not torch.cuda.is_available():
        raise SystemExit("this script needs the card; no CUDA device is visible")
    device = torch.device("cuda")
    rows = load_rows(args.split, args.limit)
    requests = load_requests()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    logger.info("%d %s rows, systems %s", len(rows), args.split, args.systems)

    if args.probe:
        run_probe(args, rows, tokenizer, device)
        return 0

    records = generate_all(args, rows, requests, tokenizer, device)
    judged = judge_all(args, rows, records, requests, tokenizer)
    summaries = [summarise(system, judged) for system in args.systems]
    verdict = rank(summaries)
    head = head_to_head(judged, args.systems)
    print_table(summaries, verdict, head)

    run_dir = Path(args.out_dir) / f"{args.split}_rank_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_row = run_dir / "per_row.jsonl"
    with per_row.open("w") as handle:
        for record in judged:
            handle.write(json.dumps(record) + "\n")
    for system in args.systems:
        answers = run_dir / f"answers_{system}.jsonl"
        with answers.open("w") as handle:
            for record in judged:
                if record["system"] == system:
                    handle.write(json.dumps({
                        "row_id": record["row_id"],
                        "response": record["response"],
                        "new_tokens": record["new_tokens"],
                        "hit_ceiling": record["hit_ceiling"],
                        "scores": record["scores"],
                    }) + "\n")
    document = {
        "state": "complete",
        "run": args.split,
        "systems": args.systems,
        "adapter_folders": {
            system: str(adapter_dir(system))
            for system in args.systems if system != "base"
        },
        "rows_source": str(TOKENISED_ROWS),
        "rows_checksum": file_checksum(TOKENISED_ROWS),
        "requests_source": str(PROMPTS_FILE),
        "requests_checksum": file_checksum(PROMPTS_FILE),
        "rows": len(rows),
        "judge": judge_config(),
        "judge_calls": JUDGE_CALLS,
        "judge_workers": args.judge_workers,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decode": "greedy",
        "seed": args.seed,
        "summaries": summaries,
        "verdict": verdict,
        "head_to_head": head,
    }
    write_manifest(run_dir / "summary.json", document)
    logger.info("wrote %s", run_dir)

    if args.no_mlflow:
        return 0
    with begin_mlflow_run(MLFLOW_EXPERIMENT, run_dir.name) as run:
        mlflow.log_params({
            "split": args.split,
            "systems": ",".join(args.systems),
            "rows": len(rows),
            "judge_model": judge_config()["model_id"],
            "judge_calls": JUDGE_CALLS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "verdict": json.dumps(verdict),
        })
        for summary in summaries:
            prefix = f"{summary['system']}"
            for name, value in summary["dimensions"].items():
                if value is not None:
                    mlflow.log_metric(f"{prefix}.{name}", value)
            if summary["core_total"] is not None:
                mlflow.log_metric(f"{prefix}.core_total", summary["core_total"])
            if summary["structure_pass_rate"] is not None:
                mlflow.log_metric(f"{prefix}.structure_pass_rate", summary["structure_pass_rate"])
            mlflow.log_metric(f"{prefix}.ceiling_hits", summary["ceiling_hits"])
        mlflow.log_artifact(str(per_row))
        mlflow.log_artifact(str(run_dir / "summary.json"))
        logger.info("mlflow run %s finished", run.info.run_id)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev",
                        help="which tokenised rows to read (default: dev)")
    parser.add_argument("--systems", default=DEFAULT_SYSTEMS,
                        help="comma-separated adapter folder names, or 'base' for none")
    parser.add_argument("--out-dir", default=str(ARTIFACTS_DIR),
                        help="where the run folder is written")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the rows read, for a smoke run")
    parser.add_argument("--probe", action="store_true",
                        help="time generation on a few rows and write nothing")
    parser.add_argument("--probe-rows", type=int, default=PROBE_ROWS,
                        help="how many rows the probe measures")
    parser.add_argument("--judge-workers", type=int, default=4,
                        help="threads used for the judge calls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true",
                        help="skip the MLflow run and keep the printed table")
    args = parser.parse_args(argv)
    args.systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not args.systems:
        raise SystemExit("at least one system is needed")
    return args


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    return run_judge(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
