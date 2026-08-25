"""Golden-subset drafter.

Drafts (query, reference_answer) candidates for the eval-data golden subset by
calling a local Ollama model on each recipe. The query-drafting rules mirror the
existing eval_dataset_generator (which already produces three queries per
recipe). The reference-answer drafting rules are deliberately strict: the model
is told to extract, not invent, so the human verifier's correction burden is
bounded.

Output rows follow the eval-data schema agreed in PR 1:
  {
    "query": ...,
    "recipe_ids": [recipe_id, ...],   # always a list
    "reference_answer": ...,          # None if drafting failed
    "query_type": "ingredients" | "name" | "technique" | "dietary" | "unknown",
    "source": "llm-draft",
    "verified": False,
  }
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pydantic import BaseModel, ValidationError
from ollama import chat
from tqdm import tqdm
import json
import time


class DraftRecord(BaseModel):
    """One candidate (query, reference_answer) pair as drafted by the model."""

    query: str
    reference_answer: str
    query_type: str = "unknown"


class DraftList(BaseModel):
    """Wrapper the model is asked to emit so we can parse a multi-query batch."""

    candidates: list[DraftRecord]


def _ingredient_list(recipe: dict) -> list[str]:
    """Flatten a recipe's ingredient_groups into a flat list of names."""
    out: list[str] = []
    for group in recipe.get("ingredient_groups", []):
        for ing in group.get("ingredients", []):
            if ing.get("name"):
                out.append(ing["name"])
    return out


def _instruction_text(recipe: dict) -> str:
    """Flatten a recipe's instruction_groups into a single string of steps."""
    parts: list[str] = []
    for group in recipe.get("instruction_groups", []):
        for step in group.get("instructions", []):
            if step:
                parts.append(step)
    return "\n".join(parts)


def build_prompt(recipe: dict) -> str:
    """Build the drafter prompt.

    Two candidate (query, reference_answer) records per recipe. The reference
    answer is asked to draw only from the recipe's own quantities, steps, and
    times; the model is explicitly told to omit anything it does not see.
    """
    ingredients = ", ".join(_ingredient_list(recipe))
    instructions = _instruction_text(recipe)
    name = recipe.get("recipe_name", "")
    return f"""You are drafting evaluation data for a recipe retrieval system.

For the recipe below, produce two (query, reference_answer) candidates that a
real user might type. Each query is what the user types; the reference answer
is the gold answer the system is expected to produce.

Rules:
- Each query must be under 20 words.
- One query should focus on ingredients; the other on technique or name. Do not
  repeat the recipe name verbatim in both queries.
- The reference answer must use ONLY information that appears in the recipe
  text below (quantities, ingredients, technique, times, temperatures). If a
  fact is not in the recipe, omit it. Do not invent.
- The reference answer should be 2 to 4 sentences and should mention the
  recipe's key ingredients and one technique or timing fact.
- Ignore adjectives and people names in the recipe name when drafting the
  queries.

Respond only with JSON in this exact shape:
{{"candidates": [
  {{"query": "...", "reference_answer": "...", "query_type": "ingredients"}},
  {{"query": "...", "reference_answer": "...", "query_type": "technique"}}
]}}

Recipe name:
{name}

Ingredients:
{ingredients}

Instructions:
{instructions}
"""


def _validate_query_type(value: str) -> str:
    """Clamp query_type to one of the four known buckets, else 'unknown'."""
    allowed = {"ingredients", "name", "technique", "dietary"}
    return value if value in allowed else "unknown"


def draft_one(
    recipe_id: int,
    recipe: dict,
    model_name: str,
    max_retries: int,
    backoff_base: int,
) -> list[DraftRecord]:
    """Draft candidate records for one recipe. Returns [] on failure."""
    prompt = build_prompt(recipe)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            parsed = DraftList.model_validate_json(response.message.content)
            for cand in parsed.candidates:
                cand.query_type = _validate_query_type(cand.query_type)
            return list(parsed.candidates)
        except (ValidationError, Exception) as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)
    if last_error is not None:
        print(f"draft_one failed for recipe_id={recipe_id}: {last_error}")
    return []


def load_processed_ids(output_path: Path) -> set[int]:
    """Recipe ids already covered by the output file, so we can resume runs."""
    if not output_path.exists():
        return set()
    processed: set[int] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ids = row.get("recipe_ids") or [row.get("recipe_id")]
                processed.update(int(i) for i in ids if i is not None)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return processed


def _process_recipe(
    recipe_id: int,
    recipe: dict,
    model_name: str,
    max_retries: int,
    backoff_base: int,
) -> tuple[int, list[DraftRecord]]:
    return recipe_id, draft_one(recipe_id, recipe, model_name, max_retries, backoff_base)


def draft_candidates(
    recipes_path: Path,
    output_path: Path,
    model_name: str,
    max_workers: int,
    max_retries: int,
    backoff_base: int,
    limit: int | None = None,
    offset: int = 0,
) -> None:
    """Draft candidate records for recipes in the corpus, in parallel.

    Resumes from anything already in `output_path`. `limit` and `offset` are
    convenience knobs so a long draft can be run in chunks without re-doing
    work; they apply before the resume check.
    """
    with recipes_path.open() as f:
        recipes = json.load(f)

    end = offset + limit if limit is not None else len(recipes)
    todo = list(enumerate(recipes[offset:end], start=offset))

    already = load_processed_ids(output_path)
    todo = [(i, r) for i, r in todo if i not in already]
    if not todo:
        return

    failed: list[int] = []
    with output_path.open("a") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_recipe,
                i,
                r,
                model_name,
                max_retries,
                backoff_base,
            ): i
            for i, r in todo
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Drafting golden candidates"):
            recipe_id = futures[fut]
            try:
                rid, candidates = fut.result()
            except Exception:  # noqa: BLE001
                failed.append(recipe_id)
                continue
            if not candidates:
                failed.append(recipe_id)
                continue
            for cand in candidates:
                out_f.write(json.dumps({
                    "query": cand.query,
                    "recipe_ids": [rid],
                    "reference_answer": cand.reference_answer,
                    "query_type": cand.query_type,
                    "source": "llm-draft",
                    "verified": False,
                }) + "\n")
            out_f.flush()

    if failed:
        preview = failed[:10]
        print(f"Failed to draft for {len(failed)} recipes: {preview}")
    print(f"Drafted for {len(todo) - len(failed)} recipes; saved to {output_path}")
