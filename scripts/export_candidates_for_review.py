"""CLI: read candidate rows + the recipe corpus, emit a Lavish batch-review HTML.

One HTML file per batch; each file shows at most BATCH_SIZE candidates (default
10, per the PR 3 scope decision) so a single lavish session doesn't accumulate
a long decision queue. The page mutates a sibling `.decisions_batch_<n>.jsonl`
through the existing `window.lavish.queuePrompt` mechanism.

Decisions are not applied to the candidate file by this CLI - the apply CLI
does that. This CLI is read-only on the candidates file and write-only on the
decision journal.
"""

import argparse
import json
import re
from html import escape
from pathlib import Path

from src.rag.core.config import DATA_PATHS

BATCH_SIZE = 10
HTML_TEMPLATE_HEAD = """<!doctype html>
<html lang="en" data-theme="luxury">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/daisyui.css" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/themes.css" />
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.2.4/dist/index.global.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  :where(.grid, .flex) > * {{ min-width: 0; }}
  :where(p, h1, h2, h3, h4, h5, h6, li, dd, blockquote, figcaption, td, th, .badge, .label) {{
    overflow-wrap: anywhere;
  }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  pre {{ white-space: pre-wrap; }}
</style>
</head>
<body class="bg-base-100 text-base-content">
<header class="max-w-5xl mx-auto px-4 pt-10 pb-4">
  <div class="flex flex-wrap items-center gap-2 mb-3">
    <span class="badge badge-soft badge-info">Golden candidates · review</span>
    <span class="badge badge-ghost">batch {batch_index} / {batch_total}</span>
    <span class="badge badge-ghost">{batch_size} candidates</span>
  </div>
  <h1 class="text-3xl font-extrabold">Golden candidates · batch {batch_index}</h1>
  <p class="mt-2 text-base-content/80">
    Mark each candidate as correct, fix, or drop. "Fix" lets you correct the
    query, ground truth, or reference answer in place. Decisions land in
    <code>{decisions_path}</code>. The apply CLI reads that file and writes the
    cleaned golden.jsonl.
  </p>
</header>
"""


HTML_TEMPLATE_FOOT = """<footer class="max-w-5xl mx-auto px-4 py-8 text-sm text-base-content/60 border-t border-base-content/10">
  Golden candidates review surface. Decisions are journaled in
  <code>{decisions_path}</code>.
</footer>
</body>
</html>
"""


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Skips blank and malformed lines."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_corpus(path: Path) -> dict[int, dict]:
    """Read the cleaned-recipes corpus into an id-keyed dict."""
    with path.open() as f:
        recipes = json.load(f)
    return {r["recipe_id"]: r for r in recipes}


def _recipe_snippet(recipe: dict, max_chars: int = 600) -> str:
    """Render a recipe's name + first few ingredients as a short preview."""
    name = recipe.get("recipe_name", "(unnamed)")
    ingredients: list[str] = []
    for group in recipe.get("ingredient_groups", [])[:1]:
        for ing in group.get("ingredients", [])[:6]:
            if ing.get("name"):
                ingredients.append(ing["name"])
    head = f"{name}\n  ingredients: {', '.join(ingredients) if ingredients else '(none)'}"
    return head[:max_chars]


def _render_candidate_row(idx: int, row: dict, recipes: dict[int, dict]) -> str:
    """Render one candidate as a review card with decision controls."""
    recipe_ids = row.get("recipe_ids") or [row.get("recipe_id")]
    recipe_id = recipe_ids[0] if recipe_ids else None
    recipe = recipes.get(recipe_id, {}) if recipe_id is not None else {}
    snippet = _recipe_snippet(recipe)
    query = escape(str(row.get("query", "")))
    reference = escape(str(row.get("reference_answer", "")))
    query_type = escape(str(row.get("query_type", "unknown")))
    form_name = f"cand-{idx}"
    original_json = escape(json.dumps({
        "candidate_index": idx,
        "row": row,
        "decision": None,
        "corrections": {},
    }))
    return f"""
<section class="max-w-5xl mx-auto px-4 py-4">
  <div class="card card-border bg-base-100">
    <div class="card-body">
      <div class="flex flex-wrap items-center gap-2 mb-2">
        <span class="badge badge-soft badge-info">cand #{idx}</span>
        <span class="badge badge-ghost">recipe_id={escape(str(recipe_id))}</span>
        <span class="badge badge-ghost">query_type={query_type}</span>
      </div>
      <h2 class="card-title text-lg">Query</h2>
      <pre class="bg-base-200 rounded-box p-3 text-sm">{query}</pre>
      <h2 class="card-title text-lg mt-3">Reference answer (draft)</h2>
      <pre class="bg-base-200 rounded-box p-3 text-sm">{reference}</pre>
      <h2 class="card-title text-lg mt-3">Recipe snippet</h2>
      <pre class="bg-base-200 rounded-box p-3 text-sm">{escape(snippet)}</pre>

      <form data-lavish-question="{form_name}"
        onsubmit="event.preventDefault(); const f = event.currentTarget; const decision = (new FormData(f).get('{form_name}-decision') || '').toString(); if (!decision) {{ return; }} const corrections = {{ query: f.querySelector(&quot;[name={form_name}-query]&quot;).value || null, recipe_ids: f.querySelector(&quot;[name={form_name}-recipe_ids]&quot;).value || null, reference_answer: f.querySelector(&quot;[name={form_name}-reference]&quot;).value || null, query_type: f.querySelector(&quot;[name={form_name}-query_type]&quot;).value || null }}; const __original = {original_json}; window.lavish.queuePrompt('cand {idx}: ' + decision, {{ tag: 'decision', text: JSON.stringify({{ decision, corrections }}), element: f, data: {{ candidate_index: {idx}, decision, corrections, original: __original }} }}); f.querySelector('[data-lavish-submit]').classList.add('btn-success'); f.querySelector('[data-lavish-submit]').textContent = 'Queued &#x2713;';">
        <h2 class="card-title text-lg mt-4">Decision</h2>
        <div class="grid md:grid-cols-3 gap-3 mt-2">
          <label class="card card-border bg-base-200 cursor-pointer hover:bg-base-300/40 transition">
            <div class="card-body p-3"><div class="flex items-center gap-2">
              <input type="radio" name="{form_name}-decision" value="correct" class="radio radio-success" />
              <span class="font-semibold">Correct</span>
            </div></div>
          </label>
          <label class="card card-border bg-base-200 cursor-pointer hover:bg-base-300/40 transition">
            <div class="card-body p-3"><div class="flex items-center gap-2">
              <input type="radio" name="{form_name}-decision" value="fix" class="radio radio-warning" />
              <span class="font-semibold">Fix</span>
            </div></div>
          </label>
          <label class="card card-border bg-base-200 cursor-pointer hover:bg-base-300/40 transition">
            <div class="card-body p-3"><div class="flex items-center gap-2">
              <input type="radio" name="{form_name}-decision" value="drop" class="radio radio-error" />
              <span class="font-semibold">Drop</span>
            </div></div>
          </label>
        </div>
        <details class="mt-3">
          <summary class="cursor-pointer text-sm text-base-content/70">Edit fields (only used if Fix is selected)</summary>
          <div class="grid md:grid-cols-2 gap-3 mt-3">
            <label class="form-control">
              <span class="label-text">Query</span>
              <textarea name="{form_name}-query" class="textarea textarea-bordered" rows="2">{query}</textarea>
            </label>
            <label class="form-control">
              <span class="label-text">Recipe IDs (comma-separated)</span>
              <input type="text" name="{form_name}-recipe_ids" class="input input-bordered" value="{escape(','.join(str(i) for i in recipe_ids if i is not None))}" />
            </label>
            <label class="form-control md:col-span-2">
              <span class="label-text">Reference answer</span>
              <textarea name="{form_name}-reference" class="textarea textarea-bordered" rows="4">{reference}</textarea>
            </label>
            <label class="form-control">
              <span class="label-text">Query type</span>
              <input type="text" name="{form_name}-query_type" class="input input-bordered" value="{query_type}" />
            </label>
          </div>
        </details>
        <div class="mt-3 flex items-center gap-3">
          <button type="submit" class="btn btn-primary btn-sm" data-lavish-submit>Queue decision</button>
          <span class="text-xs text-base-content/60">Submits one prompt to the agent.</span>
        </div>
      </form>
    </div>
  </div>
</section>
"""


def _next_decisions_journal(eval_dir: Path) -> Path:
    """Pick a fresh decisions journal filename so multiple batches can coexist."""
    existing = sorted(eval_dir.glob(".decisions_batch_*.jsonl"))
    if not existing:
        return eval_dir / ".decisions_batch_001.jsonl"
    last = existing[-1].stem  # ".decisions_batch_001"
    m = re.search(r"(\d+)$", last)
    n = int(m.group(1)) + 1 if m else 1
    return eval_dir / f".decisions_batch_{n:03d}.jsonl"


def _next_review_html(eval_dir: Path) -> Path:
    """Pick a fresh review-html filename so multiple batches can coexist."""
    existing = sorted(eval_dir.glob(".candidates_review_batch_*.html"))
    if not existing:
        return eval_dir / ".candidates_review_batch_001.html"
    last = existing[-1].stem
    m = re.search(r"(\d+)$", last)
    n = int(m.group(1)) + 1 if m else 1
    return eval_dir / f".candidates_review_batch_{n:03d}.html"


def run(args: argparse.Namespace) -> int:
    candidates_path = Path(args.candidates or DATA_PATHS["eval_golden_candidates"])
    corpus_path = Path(DATA_PATHS["cleaned_recipes"])
    eval_dir = candidates_path.parent

    rows = _read_jsonl(candidates_path)
    recipes = _load_corpus(corpus_path)
    if not rows:
        print(f"No candidates found at {candidates_path}; nothing to export.")
        return 0

    batch_index = args.batch_index
    if batch_index is None:
        existing = sorted(eval_dir.glob(".candidates_review_batch_*.html"))
        batch_index = len(existing) + 1

    start = (batch_index - 1) * BATCH_SIZE
    end = start + BATCH_SIZE
    batch = rows[start:end]
    if not batch:
        print(f"No candidates in batch {batch_index} (start={start}); nothing to export.")
        return 0

    batch_total = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    html_path = eval_dir / f".candidates_review_batch_{batch_index:03d}.html"
    decisions_path = _next_decisions_journal(eval_dir)

    cards = "\n".join(_render_candidate_row(i, row, recipes) for i, row in enumerate(batch, start=start))
    html = (
        HTML_TEMPLATE_HEAD.format(
            title=f"Golden candidates · batch {batch_index}",
            batch_index=batch_index,
            batch_total=batch_total,
            batch_size=len(batch),
            decisions_path=escape(str(decisions_path)),
        )
        + cards
        + HTML_TEMPLATE_FOOT.format(decisions_path=escape(str(decisions_path)))
    )

    html_path.write_text(html)
    print(f"Wrote {len(batch)} candidates to {html_path}")
    print(f"Decisions will land in {decisions_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=str, default=None, help="Override the candidates jsonl path.")
    parser.add_argument("--batch-index", type=int, default=None, help="Force a specific batch index; defaults to the next unused one.")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
