"""Draw the distillation split, then write the tokenised rows a trainer reads.

The student learns from rows the teacher wrote, so the eval that judges it has to be drawn from
rows it never trains on. A recipe that carries a primary slot carries it in exactly two rows, and a
hold-out made of whole recipes can only ever give even counts, while the settled eval is 297
savoury rows against 297 sweet. Rows are drawn instead, out of four pools: a recipe's category,
crossed with whether that recipe also turns up as the partner of some other row.

The eval takes its savoury half from the savoury recipes that are never a partner, which is the part
of the corpus no other row leads back into, and its sweet half from the sweet recipes on both sides
of that line. What is left is the training rows, and the dev slice is drawn from those, shaped like
the rows it is drawn from. No third bucket is held aside.

The draw is seeded, so a manifest describes a split that can be drawn again from the same numbers.
The leak figures in the manifest are measured after the draw rather than assumed, because a held-out
row can still share a dish with training through the other side of its pair, and a reader should see
how much of the eval that is before trusting what it scores.

Tokenising pins the student's prompt through the parity assertions in train_student, renders each
row as the service would hand it in, and appends the teacher's answer. The prompt is masked and the
answer carries the loss, the empty reasoning block included, so a student served without that
opening still writes a complete assistant turn. Rows are written unpadded, since padding belongs to
the batch a trainer builds.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.train_student import (
    CHAT_TEMPLATE,
    EMPTY_REASONING,
    STUDENT_MODEL_ID,
    STUDENT_SYSTEM,
    assert_template_matches,
    render_student_prompt,
)
from src.rag.core.logging import configure_logging
from src.rag.data.pairs import write_manifest
from src.rag.data.pairs.selection import (
    dataset_key_of,
    load_pairs,
    load_recipes,
    resolve_corpus,
)
from src.rag.data.source_block import user_message

logger = logging.getLogger(__name__)

# The answers the teacher wrote and the category sidecar that sorts their primaries. The answers
# file's own line order is the row address the split and the tokenised rows are keyed by.
ANSWERS_JSONL = Path("data/pairs/recipe_answers_20260918.jsonl")
CATEGORIES_JSONL = Path("data/pairs/recipe_categories.jsonl")
PAIRS_JSONL = Path("data/pairs/recipe_pairs_20260918.jsonl")
OUTPUT_DIR = Path("data/pairs")

# The draw's seed. The split is a function of this number and the corpus, so the manifest is enough
# to rebuild it.
SEED = 42

# How many rows each pool gives to the eval. The three sizes are what makes the eval 297 savoury
# against 297 sweet, and they were settled before anything was written.
EVAL_DRAW = {
    "savoury_primary_only": 297,
    "sweet_primary_only": 74,
    "sweet_with_partners": 223,
}

# The dev slice, drawn from the training rows by primary category so it mirrors them.
DEV_DRAW = {"savoury": 186, "sweet": 14}

# Every count the split has to land on, checked rather than trusted.
EXPECTED_POOLS = {
    "savoury_primary_only": 908,
    "savoury_with_partners": 1586,
    "sweet_primary_only": 148,
    "sweet_with_partners": 324,
}
EXPECTED_SPLIT = {"train": 2172, "dev": 200, "eval": 594}
EXPECTED_CATEGORY = {
    "train": {"savoury": 2011, "sweet": 161},
    "dev": {"savoury": 186, "sweet": 14},
    "eval": {"savoury": 297, "sweet": 297},
}

# The sequence cap. The longest row in the corpus is 3,964 tokens, so nothing is truncated and the
# cap is a guard against a render that has gone wrong rather than a limit on the data.
MAX_SEQ_LEN = 4096

# Rows are tokenised in blocks, so a corpus sized pass never holds every id list in Python objects
# at once.
TOKENISE_BATCH = 256

# How a row's primary is judged for the leak figures. Both readings are reported, because the split
# was priced against the first one and the second is the one that decides how much of the eval is
# really unseen.
LEAK_RULE = (
    "a row's partner is trained when that recipe appears in a training row on either side. A row's "
    "primary is absent from the partner slot when that recipe appears in no training row's partner "
    "position, which is the reading the split was priced against, and it is separately counted as "
    "present in training whenever that recipe appears in any training row at all, which is what the "
    "model has actually read about the dish"
)


def load_answers(path: Path) -> list[dict]:
    """The teacher's answers in file order, which is the row address everything else refers to."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_categories(path: Path) -> dict[int, str]:
    """The category sidecar, by recipe id."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["recipe_id"]: row["category"] for row in rows}


def pool_name(category: str, is_partner: bool) -> str:
    """The pool a row's primary belongs to: its category, and whether it doubles as a partner."""
    return f"{category}_{'with_partners' if is_partner else 'primary_only'}"


def build_split(rows: list[dict], categories: dict[int, str],
                rng: random.Random) -> tuple[list[str], dict[str, list[int]], dict[int, str]]:
    """Sort every answer row into train, dev or eval, and return the pools it was drawn from.

    The eval leaves first, out of the three pools whose sizes were settled, and the dev slice is
    then drawn from what remains within each category. The draw is made over row positions held in
    file order, so one seed rebuilds one split.
    """
    partners = Counter(row["partner_id"] for row in rows)
    pools: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        category = categories.get(row["primary_id"])
        if category is None:
            raise ValueError(f"the category sidecar does not cover recipe {row['primary_id']}")
        pools[pool_name(category, row["primary_id"] in partners)].append(position)

    for name, expected in EXPECTED_POOLS.items():
        if len(pools[name]) != expected:
            raise ValueError(f"pool {name} holds {len(pools[name])} rows, expected {expected}")

    labels = ["train"] * len(rows)
    for name, size in EVAL_DRAW.items():
        for position in rng.sample(pools[name], size):
            labels[position] = "eval"

    remaining: dict[str, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        if label == "train":
            remaining[categories[rows[position]["primary_id"]]].append(position)
    for category, size in DEV_DRAW.items():
        for position in rng.sample(remaining[category], size):
            labels[position] = "dev"

    pool_of = {position: name for name, members in pools.items() for position in members}
    return labels, dict(pools), pool_of


def count_split(rows: list[dict], categories: dict[int, str], labels: list[str]) -> dict:
    """The split's own counts, and a refusal to write a split that is not the one settled."""
    counts = Counter(labels)
    by_category: dict[str, Counter] = defaultdict(Counter)
    for row, label in zip(rows, labels):
        by_category[label][categories[row["primary_id"]]] += 1

    for name, expected in EXPECTED_SPLIT.items():
        if counts[name] != expected:
            raise ValueError(f"{name} holds {counts[name]} rows, expected {expected}")
    for name, expected in EXPECTED_CATEGORY.items():
        if dict(by_category[name]) != expected:
            raise ValueError(f"{name} holds {dict(by_category[name])}, expected {expected}")

    return {
        "rows": len(rows),
        **{name: counts[name] for name in ("train", "dev", "eval")},
        "by_category": {name: dict(by_category[name]) for name in ("train", "dev", "eval")},
    }


def leak_report(rows: list[dict], categories: dict[int, str], labels: list[str],
                training: set[int]) -> dict:
    """What a held-out row still shares with the rows behind the given training positions.

    A partner can be trained while the primary it was paired with is held out, and a primary can be
    absent from training while the dish itself is known through some other row. Both readings are
    counted, so a reader can tell a genuinely unseen dish from a row that only looks unseen.
    """
    seen = {rows[position]["primary_id"] for position in training}
    seen |= {rows[position]["partner_id"] for position in training}
    partner_slot = {rows[position]["partner_id"] for position in training}
    evaluated = [position for position, label in enumerate(labels) if label == "eval"]
    sweet = [position for position in evaluated if categories[rows[position]["primary_id"]] == "sweet"]

    partner_trained = sum(1 for position in evaluated if rows[position]["partner_id"] in seen)
    primary_in_partner_slot = sum(1 for position in evaluated if rows[position]["primary_id"] in partner_slot)
    primary_anywhere = sum(1 for position in evaluated if rows[position]["primary_id"] in seen)

    return {
        "eval_rows": len(evaluated),
        "rows_with_a_trained_partner": partner_trained,
        "rows_whose_primary_is_absent_from_the_partner_slot": len(evaluated) - primary_in_partner_slot,
        "rows_whose_primary_appears_anywhere_in_training": primary_anywhere,
        "sweet_eval_rows": len(sweet),
        "sweet_rows_whose_primary_appears_in_training": sum(
            1 for position in sweet if rows[position]["primary_id"] in seen),
    }


def file_checksum(path: Path) -> str:
    """A file's bytes, so a manifest names the exact input a pass read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def dated_paths(stem: str, suffix: str, stamp: str | None = None) -> tuple[Path, Path]:
    """This pass's output and its manifest, dated so a later pass lands beside it."""
    day = stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"{stem}_{day}{suffix}"
    return out_path, out_path.with_name(f"{out_path.stem}_manifest.json")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows whole, by rename, so a reader never sees a half-written file."""
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp_path, path)


def run_split(args: argparse.Namespace) -> int:
    answers_path, categories_path = Path(args.answers), Path(args.categories)
    out_path, manifest_path = dated_paths("distillation_split", ".jsonl", args.stamp)
    if args.out is not None:
        out_path = Path(args.out)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")

    rows = load_answers(answers_path)
    categories = load_categories(categories_path)
    labels, pools, pool_of = build_split(rows, categories, random.Random(args.seed))
    counts = count_split(rows, categories, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = {name: [i for i, label in enumerate(labels) if label == name]
                 for name in ("train", "dev", "eval")}
    pool_document = {
        name: {"rows": len(members), "recipes": len({rows[i]["primary_id"] for i in members})}
        for name, members in sorted(pools.items())
    }

    document = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": "split",
        "output": str(out_path),
        "seed": args.seed,
        "answers_file": str(answers_path),
        "answers_checksum": file_checksum(answers_path),
        "categories_file": str(categories_path),
        "categories_checksum": file_checksum(categories_path),
        "rule": (
            "rows are drawn, not recipes, because a recipe carries two primary rows and a recipe "
            "level hold-out can only give even counts. The eval takes 297 rows from the savoury "
            "recipes that are never a partner, 74 from the sweet recipes that are never a partner "
            "and 223 from the sweet recipes that are, which makes it 297 savoury rows against 297 "
            "sweet. The dev slice is then drawn from the rows left within each category, shaped "
            "like the training rows it comes from. Everything else trains."
        ),
        "draw": {"eval": dict(EVAL_DRAW), "dev": dict(DEV_DRAW)},
        "pools": pool_document,
        "counts": counts,
        "leak": {
            "measured_against": "the training rows the model learns from",
            "trained_rows": {
                **leak_report(rows, categories, labels, set(positions["train"])),
                "rows_behind_this_reading": len(positions["train"]),
            },
            "also_measured_for_comparison": {
                "measured_against": "everything the eval does not hold, dev included",
                **leak_report(rows, categories, labels,
                              set(positions["train"]) | set(positions["dev"])),
                "rows_behind_this_reading": len(positions["train"]) + len(positions["dev"]),
            },
            "definitions": LEAK_RULE,
        },
    }

    # The record goes down before the rows, so a pass that dies part way through still says which
    # split it was drawing.
    write_manifest(manifest_path, {**document, "state": "running"})
    write_jsonl(out_path, [
        {
            "row_id": position,
            "primary_id": row["primary_id"],
            "partner_id": row["partner_id"],
            "primary_category": categories[row["primary_id"]],
            "pool": pool_of[position],
            "split": labels[position],
        }
        for position, row in enumerate(rows)
    ])
    write_manifest(manifest_path, {**document, "state": "complete"})

    logger.info("wrote %d rows to %s", len(rows), out_path)
    print(f"train {counts['train']}  dev {counts['dev']}  eval {counts['eval']}  of {counts['rows']}")
    print("by category: " + "  ".join(
        f"{name} savoury={counts['by_category'][name]['savoury']} sweet={counts['by_category'][name]['sweet']}"
        for name in ("train", "dev", "eval")))
    print("pools: " + "  ".join(f"{name}={body['rows']}" for name, body in pool_document.items()))
    return 0


def context_template() -> str:
    """The two-message template without the empty reasoning block, so the block can be supervised.

    The block is the student's own opening rather than the prompt's, so the ids it tokenises to are
    carried by the target and trained on, and the context ends at the assistant turn's first line.
    """
    if not CHAT_TEMPLATE.endswith(EMPTY_REASONING):
        raise AssertionError("the chat template no longer ends with the empty reasoning block")
    template = CHAT_TEMPLATE[: -len(EMPTY_REASONING)]
    if template + EMPTY_REASONING != CHAT_TEMPLATE:
        raise AssertionError("splitting the template changed it")
    return template


def check_tokenizer(tokenizer, template: str) -> list[int]:
    """The facts the row layout rests on, each one compared rather than assumed.

    The checkpoint's own template has to be the string in train_student, the end of an answer has to
    be the sequence the model stops on, and the join between the assistant opening and the reasoning
    block has to be clean. That last one is a property of two constants rather than of any row, so
    it is checked once, on the same probe the template parity uses.
    """
    assert_template_matches(tokenizer)
    block = tokenizer(EMPTY_REASONING, add_special_tokens=False)["input_ids"]
    end = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    if end != [tokenizer.eos_token_id]:
        raise AssertionError(f"<|im_end|> tokenises to {end}, not to the eos token alone")

    probe = template.format(system="SYS", user="USER")
    joined = tokenizer(probe + EMPTY_REASONING, add_special_tokens=False)["input_ids"]
    if joined != tokenizer(probe, add_special_tokens=False)["input_ids"] + block:
        raise AssertionError("the assistant opening and the reasoning block do not tokenise apart")
    return block


def load_split(path: Path) -> list[dict]:
    """The split, in the row order the answers file holds."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if [row["row_id"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path} does not cover every answer row exactly once")
    counts = Counter(row["split"] for row in rows)
    if {name: counts[name] for name in EXPECTED_SPLIT} != EXPECTED_SPLIT:
        raise ValueError(f"{path} holds {dict(counts)}, not the settled split")
    return rows


def newest_split() -> Path:
    """The split a tokenising pass reads when it is not told which one."""
    candidates = sorted(OUTPUT_DIR.glob("distillation_split_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no split in {OUTPUT_DIR}; run --run split first")
    return candidates[-1]


def check_provenance(split_path: Path, answers_path: Path) -> None:
    """Refuse to tokenise a split whose answers file has moved since the split was drawn.

    A split row addresses the answers file by line, so an answers file that was rewritten would
    point every held-out row at a different pair. The manifest's checksum catches that before a row
    is rendered; the pair named on each row is the check behind it.
    """
    manifest_path = split_path.with_name(f"{split_path.stem}_manifest.json")
    if not manifest_path.exists():
        logger.warning("%s carries no manifest, so its inputs cannot be checked", split_path)
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("answers_checksum") != file_checksum(answers_path):
        raise ValueError(f"{answers_path} is not the file {split_path} was drawn from")


def run_tokenise(args: argparse.Namespace) -> int:
    split_path = Path(args.split_file) if args.split_file is not None else newest_split()
    out_path, manifest_path = dated_paths("tokenised_rows", ".pt", args.stamp)
    if args.out is not None:
        out_path = Path(args.out)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")

    split = load_split(split_path)
    answers = load_answers(Path(args.answers))
    check_provenance(split_path, Path(args.answers))

    # The prompts are rendered from the corpus the pairs were chosen against, so a recipe that has
    # moved or changed since is a hard stop rather than a quietly different training row.
    pairs_path = Path(args.pairs)
    dataset_key = dataset_key_of(load_pairs(pairs_path))
    corpus = resolve_corpus(dataset_key, args.recipes)
    recipes = load_recipes(Path(corpus["path"]))
    logger.info("read %d split rows, %d answers and %d recipes from %s",
                len(split), len(answers), len(recipes), corpus["path"])

    template = context_template()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    block = check_tokenizer(tokenizer, template)
    end = [tokenizer.eos_token_id]

    records: list[dict] = []
    truncated = []
    for start in range(0, len(split), TOKENISE_BATCH):
        chunk = split[start:start + TOKENISE_BATCH]
        contexts, answers_text = [], []
        for row in chunk:
            answer_row = answers[row["row_id"]]
            if (answer_row["primary_id"], answer_row["partner_id"]) != (row["primary_id"], row["partner_id"]):
                raise ValueError(f"row {row['row_id']} does not name the pair the answers file holds")
            user = user_message(recipes[row["primary_id"]], recipes[row["partner_id"]],
                                answer_row["prompt"])
            context = template.format(system=STUDENT_SYSTEM, user=user)
            if render_student_prompt(STUDENT_SYSTEM, user) != context + EMPTY_REASONING:
                raise AssertionError("the row's context and the student's prompt have drifted apart")
            contexts.append(context)
            answers_text.append(answer_row["answer"])

        context_ids = tokenizer(contexts, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answers_text, add_special_tokens=False)["input_ids"]
        for row, context, answer in zip(chunk, context_ids, answer_ids):
            # The context is what the service hands in and is masked. The assistant opening is the
            # student's own first tokens, so it sits with the answer on the supervised side and the
            # student learns to write a complete turn whether or not its server opens one for it.
            handed_in = list(context)
            supervised = list(block) + list(answer) + end
            if len(handed_in) + len(supervised) > MAX_SEQ_LEN:
                truncated.append(row["row_id"])
            records.append({
                "row_id": row["row_id"],
                "split": row["split"],
                "primary_id": row["primary_id"],
                "partner_id": row["partner_id"],
                "primary_category": row["primary_category"],
                "pool": row["pool"],
                "n_masked_tokens": len(handed_in),
                "n_supervised_tokens": len(supervised),
                "input_ids": torch.tensor(handed_in + supervised, dtype=torch.int32),
                "labels": torch.tensor([-100] * len(handed_in) + supervised, dtype=torch.int32),
            })

    if truncated:
        raise ValueError(f"{len(truncated)} rows exceed the {MAX_SEQ_LEN} token cap: {truncated[:10]}")

    totals = [len(record["input_ids"]) for record in records]
    supervised = [record["n_supervised_tokens"] for record in records]
    by_split: dict[str, dict] = {}
    for name in ("train", "dev", "eval"):
        lengths = [len(record["input_ids"]) for record in records if record["split"] == name]
        by_split[name] = {
            "rows": len(lengths),
            "median_tokens": int(statistics.median(lengths)),
            "max_tokens": max(lengths),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The payload is plain containers and tensors, so a trainer reads it back with
    # torch.load(..., weights_only=True). It lands by rename, so a killed pass leaves no partial file.
    temp_path = out_path.with_name(out_path.name + ".tmp")
    torch.save({
        "meta": {
            "tokenizer": args.model,
            "vocab_size": tokenizer.vocab_size,
            "max_seq_len": MAX_SEQ_LEN,
            "template_sha256": "sha256:" + hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest(),
            "system_prompt": STUDENT_SYSTEM,
            "masking": (
                "labels are -100 over the context the service hands in, and carry the assistant "
                "opening, the teacher's answer and the end of turn token"
            ),
            "split_file": str(split_path),
        },
        "rows": records,
    }, temp_path)
    os.replace(temp_path, out_path)

    write_manifest(manifest_path, {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": "complete",
        "run": "tokenise",
        "output": str(out_path),
        "split_file": str(split_path),
        "split_checksum": file_checksum(split_path),
        "answers_file": str(args.answers),
        "answers_checksum": file_checksum(Path(args.answers)),
        "dataset": {"key": dataset_key, **corpus},
        "tokenizer": {
            "model_id": args.model,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "template_sha256": "sha256:" + hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest(),
            "padding": "none; the rows are written at their own length and padded by the batch",
        },
        "masking": {
            "rule": (
                "the context the service hands in is masked and the assistant turn carries the loss, "
                "the empty reasoning block included, so the student writes a complete turn whether "
                "or not its server opens one for it"
            ),
            "masked": "the system turn, the user turn and the assistant turn's first line",
            "supervised": "the empty reasoning block, the answer, and the end of turn token",
        },
        "counts": {
            "rows": len(records),
            "by_split": {name: sum(1 for record in records if record["split"] == name)
                         for name in ("train", "dev", "eval")},
            "over_cap": 0,
        },
        "tokens": {
            "cap": MAX_SEQ_LEN,
            "median_total": int(statistics.median(totals)),
            "max_total": max(totals),
            "median_supervised": int(statistics.median(supervised)),
            "by_split": by_split,
        },
    })

    logger.info("wrote %d tokenised rows to %s", len(records), out_path)
    print(f"rows {len(records)}  median {int(statistics.median(totals))} tokens  max {max(totals)}"
          f"  cap {MAX_SEQ_LEN}")
    print("by split: " + "  ".join(
        f"{name} rows={body['rows']} max={body['max_tokens']}" for name, body in by_split.items()))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", choices=("split", "tokenise"), default="split",
                        help="draw the split, or tokenise the rows a split names")
    parser.add_argument("--answers", default=str(ANSWERS_JSONL),
                        help="the teacher's answers, whose line order is the row address")
    parser.add_argument("--categories", default=str(CATEGORIES_JSONL),
                        help="the category sidecar the pools are drawn from")
    parser.add_argument("--pairs", default=str(PAIRS_JSONL),
                        help="the pair set the dataset key is read from")
    parser.add_argument("--split-file", default=None,
                        help="tokenise this split; defaults to the newest one written")
    parser.add_argument("--seed", type=int, default=SEED, help="the draw's seed")
    parser.add_argument("--model", default=STUDENT_MODEL_ID, help="the tokenizer's checkpoint")
    parser.add_argument("--recipes", default=None,
                        help="read the corpus from this file instead of the registered dataset")
    parser.add_argument("--out", default=None, help="write the output here instead")
    parser.add_argument("--stamp", default=None, help="date the output with this stamp instead of today")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    return run_split(args) if args.run == "split" else run_tokenise(args)


if __name__ == "__main__":
    raise SystemExit(main())
