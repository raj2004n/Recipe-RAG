"""Distil the teacher's recipe answers into the student generator.

The teacher was handed two source recipes and a request, and wrote one new dish out of them. The
student is trained to answer the same way from the same pair: a one line system prompt, the user
turn the teacher read, and the teacher's answer as the target. This file holds that prompt and the
turn it renders into, which is the whole of the student's side of a training row.

The system prompt and the user turn come from src/rag/data/student_prompt.py rather than being held
here. The service renders a request's prompt through that same definition, so the row a student is
trained on and the request it answers cannot drift apart, and the reasoning behind the one line
system prompt is written down in that one place.

The chat template is written out here rather than read from the checkpoint. The serving path
renders prompts with the tokenizer built for the snapshot's generator model, which is gemma, and
gemma folds the system text into the first user turn while Qwen opens a system turn of its own. The
two shapes are not interchangeable, so the student's is spelled out. The parity assertion is what
keeps the copy honest: a hand-written fork of a checkpoint's template is correct only while it is
the same string that checkpoint carries.

Which rows are trained on, and the tokenised rows themselves, are written by the distillation split
script, which pins the template here through that same assertion rather than keeping a copy of its
own. This file reads those rows back and fits a rank 16 adapter over the attention and MLP
projections, one row to a micro-batch, sixteen of them to a step, with the learning rate climbing
to the settled peak over the first few steps and decaying from there along a cosine.

A step is shaped by the card it has to fit on. A row can run to the 4,096 token cap, and the logits
of a whole row would be a 248,320 wide projection per token, far past what a laptop card holds, so
the loss is taken a slice at a time and each slice is recomputed in the backward pass rather than
kept. The backbone's own activations are recomputed the same way, which is what leaves room for a
row of any length on a card of this size.
"""

import contextlib
import hashlib
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from ..core.checksum import file_checksum
from ..core.config import PROMPTS_CONFIG
from ..data.pairs import add_to_dvc, write_manifest
from ..data.preprocessor import Recipe
from ..data.source_block import labelled_blocks
from ..data.student_prompt import student_prompt, student_system
from ..evaluation.harness import begin_mlflow_run

logger = logging.getLogger(__name__)

# The checkpoint the student starts from, before any adapter is attached.
STUDENT_MODEL_ID = "Qwen/Qwen3.5-0.8B"

# The line the student opens with, read from the live prompts config the service also reads.
STUDENT_SYSTEM = student_system(PROMPTS_CONFIG["template"])

# Qwen3.5's generation prompt closes an empty reasoning block before the answer begins, so the
# answer starts straight after it. The angle brackets are escaped so the tag cannot be read as
# markup on its way into the file.
EMPTY_REASONING = "\x3cthink\x3e\n\n\x3c/think\x3e\n\n"

# The two-message case in full, which is every row of this distillation: a system turn, a user
# turn, and an assistant turn opened for the answer the student is trained to produce.
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
) + EMPTY_REASONING

# The tokenised rows a run reads and the directory its adapters land in. A run's folder is named for
# the day it started, the same way the split and its rows are, so a later run lands beside it.
TOKENISED_DIR = Path("data/pairs")
ARTIFACTS_DIR = Path("artifacts") / "student"

# The adapter: rank and width enough to move the voice of an answer without rewriting the
# checkpoint under it, spread over the attention and MLP projections of every layer.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# The loop. A row is all that fits on the card at once, so sixteen of them make a step; the peak
# rate is the settled one, reached over the first 3% of the steps and decayed away from there.
LEARNING_RATE = 2e-4
WARMUP_FRACTION = 0.03
ADAM_BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
ACCUMULATION = 16
EPOCHS = 3
SEED = 42

# How many tokens of a row are projected to the vocabulary at once, and how small a slice a row is
# retried at when the card runs out. Slices are recomputed in the backward pass, so this trades time
# against memory without changing the loss.
LOSS_CHUNK = 256
MIN_LOSS_CHUNK = 32
LOG_EVERY_STEPS = 10

MLFLOW_EXPERIMENT = "rag-student"


def student_messages(primary: Recipe, partner: Recipe, request: str) -> list[dict[str, str]]:
    """The messages a training row's prompt is made of, without the target."""
    system, user = student_prompt(STUDENT_SYSTEM, labelled_blocks([primary, partner]), request)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_student_prompt(system: str, user: str) -> str:
    """Two messages written the way the student's own chat template writes them."""
    return CHAT_TEMPLATE.format(system=system, user=user)


def assert_template_matches(tokenizer) -> None:
    """Fail here rather than train on a prompt shape the checkpoint does not carry.

    The installed checkpoint's own template is the authority. Once the two disagree, every rendered
    row is a format the student will never be served with, and a wrong transcription of the
    reasoning block does not survive a byte comparison, so the comparison is made on the strings.
    """
    probe = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    from_checkpoint = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    from_here = render_student_prompt("SYS", "USER")
    if from_here != from_checkpoint:
        raise AssertionError(
            "the hardcoded chat template no longer matches the checkpoint: "
            f"{from_here!r} != {from_checkpoint!r}"
        )


def lora_config() -> LoraConfig:
    """The adapter the student trains, at the rank and width this file declares."""
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def newest_tokenised() -> Path:
    """The tokenised rows a run reads when it is not told which file to read."""
    candidates = sorted(TOKENISED_DIR.glob("tokenised_rows_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"no tokenised rows in {TOKENISED_DIR}; run the split script's tokenise pass first"
        )
    return candidates[-1]


def load_tokenised(path: Path) -> dict:
    """The tokenised rows, checked against the prompt and masking this file trains them with.

    A row was rendered from the template above and masked at the boundary between the context the
    service hands in and the turn the student writes, so both are what a trainer has to agree with.
    A template that has since moved leaves every row in a shape the student is never served, and a
    boundary that has moved trains it to write its own instructions back out, so neither is taken on
    trust from the file's name.
    """
    payload = torch.load(path, weights_only=True)
    meta = payload["meta"]
    live = "sha256:" + hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest()
    if meta["template_sha256"] != live:
        raise ValueError(f"{path} was rendered from a different chat template than this file holds")
    if meta["system_prompt"] != STUDENT_SYSTEM:
        raise ValueError(f"{path} was rendered with a different system prompt")
    splits = {row["split"] for row in payload["rows"]}
    if splits != {"train", "dev", "eval"}:
        raise ValueError(f"{path} holds splits {sorted(splits)}")
    for row in payload["rows"]:
        masked, supervised = row["n_masked_tokens"], row["n_supervised_tokens"]
        labels, input_ids = row["labels"], row["input_ids"]
        if len(input_ids) != masked + supervised or len(labels) != len(input_ids):
            raise ValueError(f"row {row['row_id']} is a length its own masking does not explain")
        if bool((labels[:masked] != -100).any()) or not torch.equal(labels[masked:], input_ids[masked:]):
            raise ValueError(f"row {row['row_id']} does not carry the settled masking")
    return payload


def chunk_loss(hidden_slice: torch.Tensor, label_slice: torch.Tensor, lm_head) -> torch.Tensor:
    """Summed cross entropy over one slice of a row, recomputed rather than retained.

    The projection to the vocabulary is the widest tensor a step makes, 248,320 values to a token,
    so it is taken a slice at a time and cast to fp32 for the loss. Wrapping this in a checkpoint is
    what keeps the slice's logits out of the backward pass's memory, at the cost of projecting the
    slice a second time as it goes back through.
    """
    logits = lm_head(hidden_slice).float()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        label_slice.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def row_loss(base, input_ids: torch.Tensor, labels: torch.Tensor, chunk: int, checkpointed: bool):
    """A row's summed loss over its supervised tokens, and how many of those there were.

    The last hidden state is taken without the position the row has no target for, since the state
    at a token is what predicts the token after it, and the model's own projection does the rest.
    """
    hidden = base.model(input_ids=input_ids).last_hidden_state[:, :-1].contiguous()
    targets = labels[:, 1:].contiguous()
    total, supervised = torch.zeros((), device=hidden.device), 0
    for start in range(0, hidden.shape[1], chunk):
        hidden_slice = hidden[:, start:start + chunk]
        target_slice = targets[:, start:start + chunk]
        if checkpointed:
            term = checkpoint(
                chunk_loss, hidden_slice, target_slice, base.lm_head, use_reentrant=False
            )
        else:
            term = chunk_loss(hidden_slice, target_slice, base.lm_head)
        total = total + term
        supervised += int((target_slice != -100).sum())
    return total, supervised


def train_row_loss(base, input_ids: torch.Tensor, labels: torch.Tensor):
    """A row's loss, retried on smaller slices if the card fills up.

    The length of a row is not a choice, and a slice is the only part of a step whose size is, so a
    card that runs out is answered by cutting the slices rather than by dropping the row.
    """
    chunk = LOSS_CHUNK
    while True:
        try:
            return row_loss(base, input_ids, labels, chunk, checkpointed=True)
        except torch.cuda.OutOfMemoryError:
            if chunk <= MIN_LOSS_CHUNK:
                raise
            torch.cuda.empty_cache()
            chunk = max(MIN_LOSS_CHUNK, chunk // 2)
            logger.warning(
                "out of memory on a %d token row; retrying it at a loss slice of %d",
                input_ids.shape[1], chunk,
            )


def mean_loss(base, rows: list[dict], device: str) -> float:
    """The token weighted mean loss over a set of rows, with no graph kept.

    Rows differ in how much of them is supervised, so the per row means are weighted by the tokens
    behind them; an unweighted mean would let a row that ends early pull the figure around.
    """
    total, supervised = 0.0, 0
    with torch.no_grad():
        for row in rows:
            input_ids = row["input_ids"].unsqueeze(0).to(device).long()
            labels = row["labels"].unsqueeze(0).to(device).long()
            row_total, row_supervised = row_loss(base, input_ids, labels, LOSS_CHUNK, False)
            total += float(row_total)
            supervised += row_supervised
            del input_ids, labels
    return total / supervised


def build_student(device: str = "cuda"):
    """The checkpoint with the adapter attached, ready to take one row at a time.

    Gradient checkpointing and a disabled cache are what let a row of the full 4,096 token cap
    through a card this size, since the activations the backward pass needs are recomputed instead
    of held. The adapter's own weights stay in fp32, so an optimiser step at this learning rate is
    not rounded away by a bf16 parameter.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("training the student needs a CUDA device")
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, dtype=torch.bfloat16)
    model = get_peft_model(base_model, lora_config())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model.to(device), model.get_base_model()


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """The fraction of the peak learning rate that a step runs at.

    The rate climbs from near nothing to the peak across the warmup, so the opening steps do not
    throw a randomly initialised adapter at full sized updates, then follows a cosine down to
    nothing, which spends the middle of the run where the loss is still moving.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(run_dir: Path, model, document: dict) -> Path:
    """Write one epoch's adapter, the LoRA config peft keeps beside it, and the rows it was fit on.

    The directory is what the `.dvc` pointer beside it stands for, so this is also
    where the run records it.
    """
    checkpoint_dir = run_dir / f"epoch_{document['epoch']:02d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    write_manifest(checkpoint_dir / "checkpoint.json", document)
    # The directory is whole, record included, so the run that wrote the epoch moves its pointer.
    add_to_dvc(checkpoint_dir)
    return checkpoint_dir


def train_student(
    rows_path: Path | None = None,
    out_dir: Path = ARTIFACTS_DIR,
    epochs: int = EPOCHS,
    limit: int | None = None,
    seed: int = SEED,
    no_mlflow: bool = False,
) -> int:
    """Fit an adapter over the tokenised rows, one epoch at a time, and write it out.

    The rows default to the newest ones the split script wrote, since a run follows the
    tokenisation that pins its chat template. Each epoch writes its adapter, the manifest that
    records what it was fit on, and the dev loss it scored, so a finished run can be read back from
    the directory it landed in. MLflow logging is on unless `no_mlflow` turns it off; the adapters,
    the manifests and the printed summary are written either way.
    """
    device = "cuda"
    rows_path = Path(rows_path) if rows_path is not None else newest_tokenised()
    payload = load_tokenised(rows_path)
    train_rows = [row for row in payload["rows"] if row["split"] == "train"]
    dev_rows = [row for row in payload["rows"] if row["split"] == "dev"]

    per_epoch = min(len(train_rows), limit) if limit is not None else len(train_rows)
    steps_per_epoch = math.ceil(per_epoch / ACCUMULATION)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, round(WARMUP_FRACTION * total_steps))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    run_name = f"qwen3.5-0.8b_lora_r{LORA_RANK}_{stamp}"
    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "tokenised_rows": str(rows_path),
        "rows_checksum": file_checksum(rows_path),
        "template_sha256": payload["meta"]["template_sha256"],
        "system_prompt": STUDENT_SYSTEM,
        "student_model_id": STUDENT_MODEL_ID,
        "device": torch.cuda.get_device_name(0),
    }
    params = {
        "model_id": STUDENT_MODEL_ID,
        "lora.rank": LORA_RANK,
        "lora.alpha": LORA_ALPHA,
        "lora.dropout": LORA_DROPOUT,
        "lora.target_modules": ",".join(LORA_TARGET_MODULES),
        "lora.bias": "none",
        "lr.peak": LEARNING_RATE,
        "lr.schedule": "cosine to zero",
        "lr.warmup_steps": warmup_steps,
        "batch.micro_rows": 1,
        "batch.rows_per_step": ACCUMULATION,
        "optimizer": f"adamw betas={ADAM_BETAS} weight_decay=0.0",
        "grad_clip": GRAD_CLIP,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "loss_chunk": LOSS_CHUNK,
        "seed": seed,
        "epochs": epochs,
        "rows.train": len(train_rows),
        "rows.per_epoch": per_epoch,
        "rows.dev": len(dev_rows),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    manifest_path = run_dir / "run_manifest.json"
    write_manifest(manifest_path, {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": "running",
        "run": run_name,
        "params": params,
        "provenance": provenance,
        "epochs": [],
    })

    torch.manual_seed(seed)
    model, base = build_student(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    logger.info(
        "%s: %d adapter parameters over %d modules; %d rows an epoch in %d steps of %d",
        run_name, sum(p.numel() for p in trainable),
        sum(1 for name, _ in model.named_parameters() if name.endswith("lora_A.default.weight")),
        per_epoch, steps_per_epoch, ACCUMULATION,
    )

    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps)
    )
    order_rng = random.Random(seed)
    mlflow_on = not no_mlflow

    def log_metric(name: str, value: float, step: int | None = None) -> None:
        if mlflow_on:
            mlflow.log_metric(name, value, step=step)

    history: list[dict] = []
    global_step = 0
    with contextlib.nullcontext() if not mlflow_on else begin_mlflow_run(
        MLFLOW_EXPERIMENT, run_name
    ):
        if mlflow_on:
            mlflow.log_params(params)
        for epoch in range(1, epochs + 1):
            model.train()
            torch.cuda.reset_peak_memory_stats()
            order = list(range(len(train_rows)))
            order_rng.shuffle(order)
            order = order[:per_epoch]
            group_loss, group_rows = 0.0, 0
            epoch_loss, epoch_rows = 0.0, 0
            grad_norm, started = 0.0, time.perf_counter()
            step_lr = scheduler.get_last_lr()[0]
            for position, index in enumerate(order, start=1):
                row = train_rows[index]
                input_ids = row["input_ids"].unsqueeze(0).to(device).long()
                labels = row["labels"].unsqueeze(0).to(device).long()
                total, supervised = train_row_loss(base, input_ids, labels)
                (total / supervised / ACCUMULATION).backward()
                row_mean = float(total) / supervised
                group_loss, group_rows = group_loss + row_mean, group_rows + 1
                epoch_loss, epoch_rows = epoch_loss + row_mean, epoch_rows + 1
                del input_ids, labels
                if position % ACCUMULATION and position != len(order):
                    continue
                step_lr = scheduler.get_last_lr()[0]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % LOG_EVERY_STEPS and position != len(order):
                    continue
                logger.info(
                    "epoch %d step %d/%d | loss %.4f | grad_norm %.3f | lr %.2e | %.0fs",
                    epoch, global_step, total_steps, group_loss / group_rows, grad_norm,
                    step_lr, time.perf_counter() - started,
                )
                log_metric("train.loss", group_loss / group_rows, step=global_step)
                log_metric("lr", step_lr, step=global_step)
                group_loss, group_rows = 0.0, 0

            train_loss = epoch_loss / epoch_rows
            model.eval()
            dev_loss = mean_loss(base, dev_rows, device)
            elapsed = time.perf_counter() - started
            peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            record = {
                "epoch": epoch,
                "optimizer_steps": global_step,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "grad_norm": grad_norm,
                "learning_rate": step_lr,
                "peak_memory_gib": peak_gib,
                "seconds": round(elapsed, 1),
            }
            checkpoint_dir = save_checkpoint(run_dir, model, {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows_trained": per_epoch,
                "trained_on": provenance,
                **record,
            })
            history.append(record)
            logger.info(
                "epoch %d done | train %.4f | dev %.4f | peak %.2f GiB | %.0fs | %s",
                epoch, train_loss, dev_loss, peak_gib, elapsed, checkpoint_dir,
            )
            log_metric("train.epoch_loss", train_loss, step=global_step)
            log_metric("dev.loss", dev_loss, step=global_step)
            log_metric("epoch.seconds", elapsed, step=global_step)
            log_metric("peak_memory_gib", peak_gib, step=global_step)
            if mlflow_on:
                mlflow.log_artifact(str(checkpoint_dir))

        write_manifest(manifest_path, {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state": "complete",
            "run": run_name,
            "params": params,
            "provenance": provenance,
            "epochs": history,
        })
        if mlflow_on:
            mlflow.log_artifact(str(manifest_path))

    print(f"run {run_name} -> {run_dir}")
    for record in history:
        print(
            f"  epoch {record['epoch']}: train {record['train_loss']:.4f}"
            f"  dev {record['dev_loss']:.4f}  peak {record['peak_memory_gib']} GiB"
            f"  {record['seconds']:.0f}s"
        )
    return 0
