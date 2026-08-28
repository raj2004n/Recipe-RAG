"""CLI: read prompts.jsonl rows + the recipe corpus, emit a Lavish review HTML.

One HTML file per batch; each file shows at most BATCH_SIZE prompts (default 10)
so a single lavish session doesn't accumulate a long decision queue. The page
mutates a sibling `.decisions_prompts_batch_<n>.jsonl` through the existing
``window.lavish.queuePrompt`` mechanism.

Per the eval-data schema for the prompts stage, the row fields are
``query``, ``recipe_ids``, ``query_type``, ``context_chunks``, ``prompt_template``,
and ``prompt_verified``. The reviewer can mark each row accept / fix / drop.
Decisions land in the journal; the apply CLI reads the journal and emits
verified prompt rows for the answer-drafting step.
"""

import argparse
import json
import re
from html import escape
from pathlib import Path

from src.rag.core.config import DATA_PATHS

BATCH_SIZE = 10
HTML_TEMPLATE_HEAD = """<!doctype html>
<html lang="en" data-theme="emerald">
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
    <span class="badge badge-soft badge-info">Prompts · review</span>
    <span class="badge badge-ghost">batch {batch_index} / {batch_total}</span>
    <span class="badge badge-ghost">{batch_size} prompts</span>
  </div>
  <h1 class="text-3xl font-extrabold">Prompts · batch {batch_index}</h1>
  <p class="mt-2 text-base-content/80">
    Mark each prompt as accept, fix, or drop. "Fix" lets you correct the query,
    recipe_ids, or query_type in place. Decisions land in
    <code>{decisions_path}</code>. The apply CLI reads that file and writes
    verified prompt rows for the answer-drafting step.
  </p>
</header>
"""


HTML_TEMPLATE_FOOT = """<footer class="max-w-5xl mx-auto px-4 py-8 text-sm text-base-content/60 border-t border-base-content/10">
  Prompts review surface. Decisions are journaled in
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


def _recipe_name(recipes: dict[int, dict], recipe_id: int | None) -> str:
    """Look up a recipe name from the corpus; '?' if not found."""
    if recipe_id is None:
        return "?"
    recipe = recipes.get(recipe_id, {})
    return recipe.get("recipe_name", "?")


def _render_context_chunks(context_chunks: dict) -> str:
    """Render the context_chunks block as labelled text blocks per key."""
    if not context_chunks:
        return "<p class=\"text-sm opacity-70\">No context chunks.</p>"
    parts: list[str] = []
    label_map = {
        "primary": "Primary chunk",
        "info_chunk": "Info section",
        "swap_targets": "Swap target",
        "siblings": "Sibling chunks",
    }
    for key, value in context_chunks.items():
        label = label_map.get(key, key)
        if isinstance(value, list):
            inner = "\n\n---\n\n".join(str(v) for v in value)
        else:
            inner = str(value)
        parts.append(
            f'<h3 class="text-sm font-semibold mt-2">{escape(label)}</h3>'
            f'<pre class="bg-base-200 rounded-box p-3 text-xs">{escape(inner)}</pre>'
        )
    return "\n".join(parts)


def _render_prompt_row(idx: int, row: dict, recipes: dict[int, dict]) -> str:
    """Render one prompt as a review card with decision controls."""
    recipe_ids = row.get("recipe_ids") or []
    primary_id = recipe_ids[0] if recipe_ids else None
    primary_name = _recipe_name(recipes, primary_id)
    sibling_names = [_recipe_name(recipes, rid) for rid in recipe_ids[1:]]
    query = escape(str(row.get("query", "")))
    query_type = escape(str(row.get("query_type", "unknown")))
    prompt_template = escape(str(row.get("prompt_template", "")))
    context_chunks = row.get("context_chunks", {})

    form_name = f"prompt-{idx}"
    original_json = escape(json.dumps({
        "prompt_index": idx,
        "row": row,
        "decision": None,
        "corrections": {},
    }))
    return f"""
<section class="max-w-5xl mx-auto px-4 py-4">
  <div class="card card-border bg-base-100">
    <div class="card-body">
      <div class="flex flex-wrap items-center gap-2 mb-2">
        <span class="badge badge-soft badge-info">prompt #{idx}</span>
        <span class="badge badge-ghost">type={query_type}</span>
        <span class="badge badge-ghost">template={prompt_template}</span>
        <span class="badge badge-ghost">primary={escape(str(primary_id))} ({escape(primary_name)})</span>
        {''.join(f'<span class="badge badge-ghost">sibling={escape(str(rid))} ({escape(name)})</span>' for rid, name in zip(recipe_ids[1:], sibling_names))}
      </div>
      <h2 class="card-title text-lg">Prompt</h2>
      <pre class="bg-base-200 rounded-box p-3 text-sm">{query}</pre>

      <h2 class="card-title text-lg mt-3">Context chunks</h2>
      {_render_context_chunks(context_chunks)}

      <form data-lavish-question="{form_name}"
        onsubmit="event.preventDefault(); const f = event.currentTarget; const decision = (new FormData(f).get('{form_name}-decision') || '').toString(); if (!decision) {{ return; }} const corrections = {{ query: f.querySelector(&quot;[name={form_name}-query]&quot;).value || null, recipe_ids: f.querySelector(&quot;[name={form_name}-recipe_ids]&quot;).value || null, query_type: f.querySelector(&quot;[name={form_name}-query_type]&quot;).value || null }}; const __original = {original_json}; window.lavish.queuePrompt('prompt {idx}: ' + decision, {{ tag: 'decision', text: JSON.stringify({{ decision, corrections }}), element: f, data: {{ prompt_index: {idx}, decision, corrections, original: __original }} }}); f.querySelector('[data-lavish-submit]').classList.add('btn-success'); f.querySelector('[data-lavish-submit]').textContent = 'Queued &#x2713;';">
        <h2 class="card-title text-lg mt-4">Decision</h2>
        <div class="grid md:grid-cols-3 gap-3 mt-2">
          <label class="card card-border bg-base-200 cursor-pointer hover:bg-base-300/40 transition">
            <div class="card-body p-3"><div class="flex items-center gap-2">
              <input type="radio" name="{form_name}-decision" value="accept" class="radio radio-success" />
              <span class="font-semibold">Accept</span>
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
            <label class="form-control md:col-span-2">
              <span class="label-text">Query</span>
              <textarea name="{form_name}-query" class="textarea textarea-bordered" rows="2">{query}</textarea>
            </label>
            <label class="form-control">
              <span class="label-text">Recipe IDs (comma-separated)</span>
              <input type="text" name="{form_name}-recipe_ids" class="input input-bordered" value="{escape(','.join(str(i) for i in recipe_ids if i is not None))}" />
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


def _default_batch_index(eval_dir: Path) -> int:
    """Pick the next unused batch index for review-html filenames."""
    existing = sorted(eval_dir.glob(".prompts_review_batch_*.html"))
    return len(existing) + 1


def run(args: argparse.Namespace) -> int:
    prompts_path = Path(args.prompts or DATA_PATHS["eval_prompts"])
    corpus_path = Path(DATA_PATHS["cleaned_recipes"])
    eval_dir = prompts_path.parent

    rows = _read_jsonl(prompts_path)
    recipes = _load_corpus(corpus_path)
    if not rows:
        print(f"No prompts found at {prompts_path}; nothing to export.")
        return 0

    batch_index = args.batch_index
    if batch_index is None:
        batch_index = _default_batch_index(eval_dir)

    start = args.start if args.start is not None else (batch_index - 1) * BATCH_SIZE
    count = args.count if args.count is not None else BATCH_SIZE
    end = start + count
    batch = rows[start:end]
    if not batch:
        print(f"No prompts in batch {batch_index} (start={start}); nothing to export.")
        return 0

    batch_total = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    html_path = eval_dir / f".prompts_review_batch_{batch_index:03d}.html"
    # Each batch has its own decisions journal so reviews don't collide.
    decisions_path = eval_dir / f".decisions_prompts_batch_{batch_index:03d}.jsonl"

    cards = "\n".join(_render_prompt_row(i, row, recipes) for i, row in enumerate(batch, start=start))
    html = (
        HTML_TEMPLATE_HEAD.format(
            title=f"Prompts review · batch {batch_index}",
            batch_index=batch_index,
            batch_total=batch_total,
            batch_size=len(batch),
            decisions_path=escape(str(decisions_path)),
        )
        + cards
        + HTML_TEMPLATE_FOOT.format(decisions_path=escape(str(decisions_path)))
    )

    html_path.write_text(html)
    print(f"Wrote {len(batch)} prompts to {html_path}")
    print(f"Decisions will land in {decisions_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=str, default=None, help="Override the prompts jsonl path.")
    parser.add_argument("--batch-index", type=int, default=None, help="Force a specific batch index; defaults to the next unused one.")
    parser.add_argument("--start", type=int, default=None, help="Override the row offset. Defaults to (batch_index-1)*BATCH_SIZE so each batch gets its own slice.")
    parser.add_argument("--count", type=int, default=None, help="Number of rows to render. Defaults to BATCH_SIZE.")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
