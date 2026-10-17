from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


class PromotionConfigError(Exception):
    """A configs/promotion.yaml that cannot feed the gate."""


@dataclasses.dataclass(frozen=True)
class Check:
    """One bar applied to one run metric.

    scope names where the metric lives: "slice" for a slice aggregate or
    "query_type:<name>" for a single query type. value is None when the run
    never logged the metric, which is a failed check so an incomplete run
    fails closed.
    """

    metric: str
    scope: str
    operator: str  # ">=" for at_least bars, "<=" for at_most bars
    bar: float
    value: float | None

    @property
    def passed(self) -> bool:
        if self.value is None:
            return False
        if self.operator == ">=":
            return self.value >= self.bar
        if self.operator == "<=":
            return self.value <= self.bar
        return False

    def as_dict(self) -> dict[str, Any]:
        """A JSON-friendly record for the promotion log in the registry.

        `passed` is resolved here rather than left to a reader of the record, so
        the registry says on its own which bars a promotion went ahead against.
        """
        return {
            "metric": self.metric,
            "scope": self.scope,
            "operator": self.operator,
            "bar": self.bar,
            "value": self.value,
            "passed": self.passed,
        }


def _load_bar_spec(
    section: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Split one `at_least` / `at_most` spec into the two maps, validated."""
    at_least = section.get("at_least") or {}
    at_most = section.get("at_most") or {}
    if not isinstance(at_least, dict) or not isinstance(at_most, dict):
        raise PromotionConfigError(
            "bar specs must hold 'at_least' and 'at_most' mapping sections"
        )
    for metric, value in {**at_least, **at_most}.items():
        if value is None:
            raise PromotionConfigError(f"bar {metric!r} has no threshold")
    return dict(at_least), dict(at_most)


def load_promotion_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the gate criteria from a promotion YAML."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    gate = data.get("gate")
    if not isinstance(gate, dict):
        raise PromotionConfigError(f"no 'gate' section in {path}")
    for section in ("bars", "per_query_type_bars"):
        if section not in gate:
            raise PromotionConfigError(f"missing '{section}' in {path}")
    for section in ("bars", "per_query_type_bars"):
        _load_bar_spec(gate[section])
    for key in ("experiment", "run_marker_param", "sample_size_caveat"):
        if key not in gate:
            raise PromotionConfigError(f"missing gate '{key}' in {path}")
    return data


def _slice_prefixes() -> set[str]:
    """The metric prefixes the e2e run logs for the slice aggregate.

    The end-to-end script logs the slice aggregate bare
    (mean.constraint_satisfaction) and every per-query-type aggregate under a
    query-type prefix (recipe_variation.mean.constraint_satisfaction). Any
    other first segment of a dotted metric is a query type present in the run.
    """
    return {"mean", "disagreement", "flag", "retrieval"}


def query_type_names(metrics: dict[str, float]) -> list[str]:
    """The query types present in a run, derived from its metric keys."""
    prefixes = {key.split(".", 1)[0] for key in metrics if "." in key}
    return sorted(prefixes - _slice_prefixes())


def apply_bars(
    metrics: dict[str, float],
    at_least: dict[str, float],
    at_most: dict[str, float],
    scope: str,
) -> list[Check]:
    """Build the checks for one scope from its metrics and the bars."""
    checks = []
    for metric, bar in sorted(at_least.items()):
        checks.append(Check(metric, scope, ">=", bar, metrics.get(metric)))
    for metric, bar in sorted(at_most.items()):
        checks.append(Check(metric, scope, "<=", bar, metrics.get(metric)))
    return checks


def evaluate(metrics: dict[str, float], config: dict[str, Any]) -> list[Check]:
    """The full gate verdict from one run's metrics.

    Slice bars run against the bare metrics; per-query-type bars run against
    every query type the run contains. A query type's metric is looked up by
    its prefix, so recipe_variation.mean.constraint_satisfaction is the value
    checked against the mean.constraint_satisfaction floor.
    """
    gate = config["gate"]
    slice_at_least, slice_at_most = _load_bar_spec(gate["bars"])
    type_at_least, type_at_most = _load_bar_spec(gate["per_query_type_bars"])

    checks = apply_bars(metrics, slice_at_least, slice_at_most, "slice")
    for query_type in query_type_names(metrics):
        prefix = f"{query_type}."
        scoped = {
            key[len(prefix):]: value
            for key, value in metrics.items()
            if key.startswith(prefix)
        }
        checks.extend(
            apply_bars(scoped, type_at_least, type_at_most, f"query_type:{query_type}")
        )
    return checks
