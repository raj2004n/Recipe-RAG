"""Shared judgment aggregation for the generation and end-to-end evaluators.

Both scoring stages judge generated answers with the same four dimensions of
the LLM judge and aggregate over a slice and per query type. The helpers here
are the single definition of what a dimension mean, a disagreement rate, and
a slice aggregate are, so the two MLflow experiment families never drift on
the shape of their scores. The per-row judgment builder turns the repeated
judge calls for one answer into the per-dimension scores and disagreement
used by both evaluators, and the budget helper mirrors the prompt builder's
token loop so the judge always sees the context the generator saw.
"""

from collections.abc import Iterable, Sequence

JUDGED_DIMENSIONS = (
    "constraint_satisfaction",
    "query_relevance",
    "faithfulness",
    "expectedness",
)

_CORE_DIMENSIONS = ("constraint_satisfaction", "query_relevance", "faithfulness")


def mean(values: Sequence[float]) -> float | None:
    """Mean of the non-None values, or None when every value is missing."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def disagrees(values: Sequence[int]) -> bool | None:
    """Whether the judge calls for one dimension gave different scores.

    None when no call succeeded, so an all-failed row is not counted as
    agreement or disagreement.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return len(set(present)) > 1


def judgment(core_verdicts: Sequence, expectedness_verdicts: Sequence) -> tuple[dict, dict]:
    """Mean score and disagreement per dimension from one row's judge calls.

    core_verdicts are the repeated core-judge results for one answer and
    expectedness_verdicts the repeated expectedness results, each entry a
    pydantic verdict model or None when the call failed. Failed entries are
    skipped so a row whose calls all fail is recorded with no score rather
    than a false zero. The expectedness dimension is present but None when no
    expectedness call succeeded, which keeps the two evaluators' records
    uniform in shape.
    """
    values: dict[str, list[int]] = {dim: [] for dim in _CORE_DIMENSIONS}
    for verdict in core_verdicts:
        if verdict is None:
            continue
        for dim in _CORE_DIMENSIONS:
            values[dim].append(getattr(verdict, dim))
    expectedness = [v.expectedness for v in expectedness_verdicts if v]

    scores = {dim: mean(values[dim]) for dim in _CORE_DIMENSIONS}
    scores["expectedness"] = mean(expectedness)
    disagreement = {dim: disagrees(values[dim]) for dim in _CORE_DIMENSIONS}
    disagreement["expectedness"] = disagrees(expectedness)
    return scores, disagreement


def budget_texts(texts: Sequence[str], tokenizer, max_tokens: int) -> list[str]:
    """The texts the prompt builder would keep under its token budget.

    Mirrors the builder's accumulation loop so an evaluator can hand the
    judge exactly the context the generator saw and flag truncation. The
    texts keep their order and the first text that would overflow the budget
    ends the list.
    """
    included = []
    used = 0
    for text in texts:
        tokens = len(tokenizer.encode(text))
        if used + tokens > max_tokens:
            break
        used += tokens
        included.append(text)
    return included


def aggregate(rows: Sequence[dict]) -> dict:
    """Slice-level means, disagreement rates, and structural flag rates.

    Takes per-row records as produced by either evaluator, each carrying
    scores, disagreement, and structural dicts. A dimension contributes its
    mean only when at least one row scored it and a disagreement rate only
    when any row reported a disagreement signal, so an all-unscored slice
    stays empty rather than reporting false zeroes.
    """
    result = {}
    for dimension in JUDGED_DIMENSIONS:
        dim_mean = mean([row["scores"][dimension] for row in rows])
        if dim_mean is not None:
            result[f"mean.{dimension}"] = dim_mean
        disagreements = [
            row["disagreement"][dimension]
            for row in rows
            if row["disagreement"][dimension] is not None
        ]
        if disagreements:
            result[f"disagreement.{dimension}"] = (
                sum(disagreements) / len(disagreements)
            )
    if rows:
        result["structure_pass_rate"] = (
            sum(1 for row in rows if row["structural"]["structure_pass"])
            / len(rows)
        )
        flag_names = [
            name for name in rows[0]["structural"] if name != "structure_pass"
        ]
        for name in flag_names:
            result[f"flag.{name}"] = (
                sum(1 for row in rows if row["structural"][name]) / len(rows)
            )
    return result
