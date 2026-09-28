"""Tokenise the rows a split names into the file a trainer reads.

Tokenising pins the student's prompt through the parity assertions in
src/rag/training/student.py, renders each row as the service would hand it in, and appends the
teacher's answer. The prompt is masked and the answer carries the loss, the empty reasoning block
included, so a student served without that opening still writes a complete assistant turn. Rows are
written unpadded, since padding belongs to the batch a trainer builds.
"""

import hashlib
import logging
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ...core.checksum import file_checksum
from ...training.student import (
    CHAT_TEMPLATE,
    EMPTY_REASONING,
    STUDENT_MODEL_ID,
    STUDENT_SYSTEM,
    assert_template_matches,
    render_student_prompt,
)
from ..pairs import write_manifest
from ..pairs.selection import (
    dataset_key_of,
    load_pairs,
    load_recipes,
    resolve_corpus,
)
from ..source_block import user_message
from .selection import check_provenance, load_answers, load_split, newest_split
from .types import (
    ANSWERS_JSONL,
    MAX_SEQ_LEN,
    PAIRS_JSONL,
    TOKENISE_BATCH,
    dated_paths,
)

logger = logging.getLogger(__name__)


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

    The checkpoint's own template has to be the string in src/rag/training/student.py, the end of an
    answer has to be the sequence the model stops on, and the join between the assistant opening and
    the reasoning block has to be clean. That last one is a property of two constants rather than of
    any row, so it is checked once, on the same probe the template parity uses.
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


def tokenise_rows(
    split_file: str | None = None,
    answers_path: Path = ANSWERS_JSONL,
    pairs_path: Path = PAIRS_JSONL,
    recipes: str | None = None,
    model_id: str = STUDENT_MODEL_ID,
    output_path: Path | None = None,
    stamp: str | None = None,
) -> int:
    """Run the pass: render every split row, tokenise it, and write the tensors and the manifest.

    The landing spot follows the dated default unless `output_path` names one, and the stamp dates
    the default in place of today. The rows land by rename, so a killed pass leaves no partial file.
    """
    answers_path = Path(answers_path)
    pairs_path = Path(pairs_path)
    split_path = Path(split_file) if split_file is not None else newest_split()
    out_path, manifest_path = dated_paths("tokenised_rows", ".pt", stamp)
    if output_path is not None:
        out_path = Path(output_path)
        manifest_path = out_path.with_name(f"{out_path.stem}_manifest.json")

    split = load_split(split_path)
    answers = load_answers(answers_path)
    check_provenance(split_path, answers_path)

    # The prompts are rendered from the corpus the pairs were chosen against, so a recipe that has
    # moved or changed since is a hard stop rather than a quietly different training row.
    dataset_key = dataset_key_of(load_pairs(pairs_path))
    corpus = resolve_corpus(dataset_key, recipes)
    recipes = load_recipes(Path(corpus["path"]))
    logger.info("read %d split rows, %d answers and %d recipes from %s",
                len(split), len(answers), len(recipes), corpus["path"])

    template = context_template()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
            "tokenizer": model_id,
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
        "answers_file": str(answers_path),
        "answers_checksum": file_checksum(answers_path),
        "dataset": {"key": dataset_key, **corpus},
        "tokenizer": {
            "model_id": model_id,
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
