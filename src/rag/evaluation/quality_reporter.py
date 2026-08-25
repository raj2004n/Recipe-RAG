"""Dataset-quality reporter for the eval-data workstream.

Computes summary statistics over the two eval datasets (golden and LLM-generated)
and renders them as both a human-readable table and a machine-readable JSON file.
The two datasets are always reported separately, never blended — golden is small
and human-verified, LLM-generated is large and unverified, and any aggregate
metric would obscure that distinction.

The reporter tolerates two shapes on the read side:
- New:  {"recipe_ids": [int, ...], "query": ..., ...}
- Old:  {"recipe_id": int, "query": ..., ...}
A row that uses the legacy recipe_id field is counted in legacy_rows and the
warning string notes the count so schema drift can't slip by unnoticed.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class DatasetQuality:
    """Summary statistics for a single eval dataset."""

    name: str
    path: str
    rows: int
    distinct_queries: int
    query_diversity: float
    distinct_recipes: int
    coverage: float
    avg_queries_per_recipe: float
    ambiguity_rate: float
    verified_rate: float
    reference_answer_rate: float
    source_breakdown: dict[str, int] = field(default_factory=dict)
    legacy_rows: int = 0
    warning: str | None = None


@dataclass(frozen=True)
class QualityReport:
    """Top-level report: the corpus we evaluate against and both datasets' quality."""

    corpus_size: int
    corpus_path: str
    golden: DatasetQuality
    llm_generated: DatasetQuality


def _read_rows(path: str) -> tuple[list[Mapping], str | None]:
    """Read a JSONL file into a list of dicts.

    Missing file or empty file is not an error — it returns zero rows and a
    descriptive warning so the caller can render the empty state.
    """
    p = Path(path)
    if not p.exists():
        return [], f"file not found: {path}"
    rows: list[Mapping] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return [], f"file empty: {path}"
    return rows, None


def _read_with_legacy(path: str) -> tuple[list[Mapping], int, str | None]:
    """Like _read_rows, but also count rows that fell through to legacy recipe_id.

    Returns (rows, legacy_rows, warning). legacy_rows is the count of rows
    lacking `recipe_ids` but carrying a legacy `recipe_id` field; such rows
    still resolve correctly via _recipe_ids, but the count is exposed so the
    report can warn about schema drift.
    """
    rows, warning = _read_rows(path)
    legacy_rows = 0
    for row in rows:
        if "recipe_ids" not in row and "recipe_id" in row:
            legacy_rows += 1
    return rows, legacy_rows, warning


def _recipe_ids(row: Mapping) -> list[int]:
    """Return the ground-truth recipe ids for a row, accepting either schema."""
    ids = row.get("recipe_ids")
    if ids is not None:
        return [int(i) for i in ids if i is not None]
    legacy = row.get("recipe_id")
    return [int(legacy)] if legacy is not None else []


def _ambiguity_rate_per_query(rows: list[Mapping]) -> float:
    """A query is ambiguous iff it appears in rows whose union of ground-truth
    sets spans more than one distinct recipe id (definition B, per-query)."""
    query_to_recipes: dict[str, set[int]] = {}
    for row in rows:
        ids = _recipe_ids(row)
        query = row.get("query")
        if not ids or not isinstance(query, str) or not query:
            continue
        query_to_recipes.setdefault(query, set()).update(ids)
    if not query_to_recipes:
        return 0.0
    ambiguous = sum(1 for ids in query_to_recipes.values() if len(ids) > 1)
    return ambiguous / len(query_to_recipes)


def _query_diversity(rows: list[Mapping]) -> float:
    """Distinct queries divided by total rows. 1.0 means every row is unique."""
    if not rows:
        return 0.0
    distinct = len({row["query"] for row in rows if isinstance(row.get("query"), str) and row["query"]})
    return distinct / len(rows)


def _coverage(rows: list[Mapping], corpus_size: int) -> float:
    """Distinct recipes referenced by any row, divided by corpus size."""
    if corpus_size == 0:
        return 0.0
    referenced: set[int] = set()
    for row in rows:
        referenced.update(_recipe_ids(row))
    return len(referenced) / corpus_size


def _avg_queries_per_recipe(rows: list[Mapping]) -> float:
    """Total rows divided by distinct recipes referenced."""
    if not rows:
        return 0.0
    referenced: set[int] = set()
    for row in rows:
        referenced.update(_recipe_ids(row))
    if not referenced:
        return 0.0
    return len(rows) / len(referenced)


def _rate(rows: list[Mapping], predicate) -> float:
    """Fraction of rows for which predicate(row) is truthy."""
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _source_breakdown(rows: list[Mapping]) -> dict[str, int]:
    """Count rows by their `source` field; missing field is bucketed as 'unknown'."""
    counts: dict[str, int] = {}
    for row in rows:
        src = row.get("source", "unknown")
        counts[str(src)] = counts.get(str(src), 0) + 1
    return counts


def _corpus_size(corpus_path: str) -> int:
    """Count the number of recipes in the corpus file.

    Raises FileNotFoundError rather than silently returning 0, because a 0
    denominator would corrupt the coverage metric for both datasets.
    """
    p = Path(corpus_path)
    if not p.exists():
        raise FileNotFoundError(
            f"corpus file not found: {corpus_path} — coverage denominator would be wrong"
        )
    with p.open() as f:
        recipes = json.load(f)
    return len(recipes)


def _summarize(
    name: str,
    path: str,
    rows: list[Mapping],
    legacy_rows: int,
    corpus_size: int,
    warning: str | None,
) -> DatasetQuality:
    """Build a DatasetQuality from a row list."""
    return DatasetQuality(
        name=name,
        path=path,
        rows=len(rows),
        distinct_queries=len(
            {row["query"] for row in rows if isinstance(row.get("query"), str) and row["query"]}
        ),
        query_diversity=_query_diversity(rows),
        distinct_recipes=len({i for row in rows for i in _recipe_ids(row)}),
        coverage=_coverage(rows, corpus_size),
        avg_queries_per_recipe=_avg_queries_per_recipe(rows),
        ambiguity_rate=_ambiguity_rate_per_query(rows),
        verified_rate=_rate(rows, lambda r: bool(r.get("verified"))),
        reference_answer_rate=_rate(rows, lambda r: r.get("reference_answer") not in (None, "")),
        source_breakdown=_source_breakdown(rows),
        legacy_rows=legacy_rows,
        warning=_compose_warning(warning, legacy_rows),
    )


def _compose_warning(file_warning: str | None, legacy_rows: int) -> str | None:
    """Combine a file-level warning with a schema-drift note if applicable."""
    parts = [w for w in (file_warning,) if w]
    if legacy_rows > 0:
        parts.append(f"{legacy_rows} row(s) used legacy recipe_id field")
    return "; ".join(parts) if parts else None


def report_quality(
    golden_path: str,
    llm_generated_path: str,
    corpus_path: str,
) -> QualityReport:
    """Build a QualityReport from the two eval datasets and a corpus file.

    The corpus file must exist — see _corpus_size. The two eval datasets
    are tolerated as missing or empty, with a warning recorded per dataset.
    """
    corpus_size = _corpus_size(corpus_path)
    golden_rows, golden_legacy, golden_warn = _read_with_legacy(golden_path)
    llm_rows, llm_legacy, llm_warn = _read_with_legacy(llm_generated_path)
    return QualityReport(
        corpus_size=corpus_size,
        corpus_path=corpus_path,
        golden=_summarize("golden", golden_path, golden_rows, golden_legacy, corpus_size, golden_warn),
        llm_generated=_summarize(
            "llm_generated", llm_generated_path, llm_rows, llm_legacy, corpus_size, llm_warn
        ),
    )


def format_table(report: QualityReport) -> str:
    """Render the report as a fixed-width text table for terminal output."""
    g, l = report.golden, report.llm_generated
    label_w = 28
    lines = [
        "Eval dataset quality report",
        f"  Source corpus: {report.corpus_size} recipes ({report.corpus_path})",
        "",
        f"  {'metric'.ljust(label_w)}{'golden'.rjust(12)}{'llm_generated'.rjust(18)}",
        f"  {'-' * (label_w + 30)}",
    ]

    def row(label: str, gv, lv, fmt=str) -> str:
        return f"  {label.ljust(label_w)}{fmt(gv).rjust(12)}{fmt(lv).rjust(18)}"

    lines.append(row("rows", g.rows, l.rows))
    lines.append(row("distinct queries", g.distinct_queries, l.distinct_queries))
    lines.append(row("query diversity", g.query_diversity, l.query_diversity, lambda v: f"{v:.3f}"))
    lines.append(row("distinct recipes", g.distinct_recipes, l.distinct_recipes))
    lines.append(
        row(
            "coverage (% of corpus)",
            g.coverage * 100,
            l.coverage * 100,
            lambda v: f"{v:.1f}%",
        )
    )
    lines.append(
        row(
            "avg queries per recipe",
            g.avg_queries_per_recipe,
            l.avg_queries_per_recipe,
            lambda v: f"{v:.2f}",
        )
    )
    lines.append(
        row(
            "ambiguity rate (per-query)",
            g.ambiguity_rate * 100,
            l.ambiguity_rate * 100,
            lambda v: f"{v:.1f}%",
        )
    )
    lines.append(
        row(
            "verified rate",
            g.verified_rate * 100,
            l.verified_rate * 100,
            lambda v: f"{v:.1f}%",
        )
    )
    lines.append(
        row(
            "reference_answer rate",
            g.reference_answer_rate * 100,
            l.reference_answer_rate * 100,
            lambda v: f"{v:.1f}%",
        )
    )
    lines.append(
        row(
            "legacy rows (recipe_id)",
            g.legacy_rows,
            l.legacy_rows,
        )
    )
    lines.append("")
    lines.append("  source breakdown")
    sources = sorted(set(g.source_breakdown) | set(l.source_breakdown))
    for src in sources:
        lines.append(row(f"    {src}", g.source_breakdown.get(src, 0), l.source_breakdown.get(src, 0)))
    if g.warning or l.warning:
        lines.append("")
        lines.append("  warnings")
        if g.warning:
            lines.append(f"    golden: {g.warning}")
        if l.warning:
            lines.append(f"    llm_generated: {l.warning}")
    return "\n".join(lines) + "\n"


def write_json(report: QualityReport, output_path: str) -> None:
    """Serialise the report to JSON. Dataclasses are dumped via asdict."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)
