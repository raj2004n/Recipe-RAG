"""Tests for the promotion gate verdict and the registry promotion record.

The gate logic is exercised against a small synthetic criteria config so the
tests pin the behavior (per-type floors, the judge-disagreement rule, fail
closed on missing metrics) without coupling to whatever the bars are calibrated
to; the repo's real config gets a structural validation check only.
"""

import json

import pytest

from src.rag.core.snapshot import registry
from src.rag.evaluation.promotion import (
    Check,
    PromotionConfigError,
    apply_bars,
    evaluate,
    load_promotion_config,
    query_type_names,
)


def _config() -> dict:
    """A minimal but well-formed gate criteria config for logic tests."""
    return {
        "gate": {
            "experiment": "test-exp",
            "run_marker_param": "snapshot_name",
            "sample_size_caveat": "caveat",
            "bars": {
                "at_least": {
                    "mean.constraint_satisfaction": 1.75,
                    "structure_pass_rate": 0.95,
                },
                "at_most": {
                    "disagreement.constraint_satisfaction": 0.20,
                },
            },
            "per_query_type_bars": {
                "at_least": {
                    "mean.constraint_satisfaction": 1.50,
                    "structure_pass_rate": 0.75,
                },
                "at_most": {
                    "disagreement.constraint_satisfaction": 0.30,
                },
            },
        }
    }


def _base_metrics() -> dict:
    """A run whose slice and one query type clear every bar in _config."""
    return {
        "structure_pass_rate": 0.97,
        "mean.constraint_satisfaction": 1.85,
        "disagreement.constraint_satisfaction": 0.10,
        "recipe_variation.structure_pass_rate": 0.80,
        "recipe_variation.mean.constraint_satisfaction": 1.60,
        "recipe_variation.disagreement.constraint_satisfaction": 0.10,
    }


# --- verdict logic --------------------------------------------------------


def test_query_type_names_derived_from_metric_keys():
    metrics = {
        "structure_pass_rate": 1.0,
        "mean.constraint_satisfaction": 1.8,
        "flag.has_name": 1.0,
        "retrieval.golden_recall": 0.9,
        "recipe_variation.structure_pass_rate": 0.9,
        "recipe_variation.mean.constraint_satisfaction": 1.6,
        "recipe_combination.structure_pass_rate": 0.8,
    }
    assert query_type_names(metrics) == ["recipe_combination", "recipe_variation"]


def test_apply_bars_respects_direction():
    checks = apply_bars(
        {"structure_pass_rate": 0.96, "disagreement.constraint_satisfaction": 0.2},
        {"structure_pass_rate": 0.95},
        {"disagreement.constraint_satisfaction": 0.2},
        "slice",
    )
    by_metric = {c.metric: c for c in checks}
    assert by_metric["structure_pass_rate"].operator == ">="
    assert by_metric["structure_pass_rate"].passed is True
    assert by_metric["disagreement.constraint_satisfaction"].operator == "<="
    assert by_metric["disagreement.constraint_satisfaction"].passed is True


def test_apply_bars_fails_a_bar_with_no_value():
    checks = apply_bars({}, {"mean.constraint_satisfaction": 1.75}, {}, "slice")
    assert len(checks) == 1
    assert checks[0].value is None
    assert checks[0].passed is False


def test_evaluate_passes_when_slice_and_types_clear():
    checks = evaluate(_base_metrics(), _config())
    assert checks
    assert all(check.passed for check in checks)


def test_evaluate_fails_when_slice_mean_under_the_bar():
    metrics = _base_metrics()
    metrics["mean.constraint_satisfaction"] = 1.40  # under the 1.75 slice bar
    checks = evaluate(metrics, _config())
    failed = [c for c in checks if not c.passed]
    assert failed
    assert {c.scope for c in failed} == {"slice"}
    assert "mean.constraint_satisfaction" in {c.metric for c in failed}


def test_evaluate_fails_on_type_floor_even_when_slice_passes():
    # The one weak query type must sink the gate even though the slice clears.
    metrics = _base_metrics()
    metrics["recipe_variation.mean.constraint_satisfaction"] = 1.40  # under 1.50
    checks = evaluate(metrics, _config())
    failed = [c for c in checks if not c.passed]
    assert failed
    assert {c.scope for c in failed} == {"query_type:recipe_variation"}


def test_evaluate_fails_high_mean_with_high_disagreement():
    # The judge-noise rule: the mean is not trusted when the disagreement is
    # over the cap, so the dimension fails even though its mean clears.
    metrics = _base_metrics()
    metrics["disagreement.constraint_satisfaction"] = 0.40  # over the 0.20 cap
    checks = evaluate(metrics, _config())
    failed = [c for c in checks if not c.passed]
    assert any(
        c.metric == "disagreement.constraint_satisfaction" and c.scope == "slice"
        for c in failed
    )
    assert not all(c.passed for c in checks)


def test_evaluate_looks_up_type_bars_under_the_type_prefix():
    checks = evaluate(_base_metrics(), _config())
    type_checks = [c for c in checks if c.scope == "query_type:recipe_variation"]
    by_metric = {c.metric: c for c in type_checks}
    # The 1.60 value lives under recipe_variation.mean.constraint_satisfaction
    # and is what clears the 1.50 type floor.
    assert by_metric["mean.constraint_satisfaction"].value == 1.60
    assert by_metric["mean.constraint_satisfaction"].passed is True


def test_check_as_dict_is_json_friendly():
    check = Check("structure_pass_rate", "slice", ">=", 0.95, 0.97)
    assert json.dumps(check.as_dict())


# --- config loading -------------------------------------------------------


def test_repo_promotion_config_loads_and_is_well_formed():
    config = load_promotion_config("configs/promotion.yaml")
    gate = config["gate"]
    assert gate["experiment"] == "rag-end-to-end-eval"
    assert gate["run_marker_param"] == "snapshot_name"
    assert gate["sample_size_caveat"]
    for section in ("bars", "per_query_type_bars"):
        at_least, at_most = gate[section]["at_least"], gate[section]["at_most"]
        assert at_least and at_most
        for value in {**at_least, **at_most}.values():
            assert isinstance(value, (int, float))


def test_missing_gate_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("gate:\n  bars:\n    at_least: {}\n    at_most: {}\n")
    with pytest.raises(PromotionConfigError):
        load_promotion_config(bad)


# --- registry promotion record -------------------------------------------


def test_record_promotion_sets_targets_baseline_and_record(tmp_path):
    registry_file = tmp_path / "registry.json"
    checks = [Check("structure_pass_rate", "slice", ">=", 0.95, 0.97).as_dict()]
    registry.record_promotion(
        registry_file, "hybrid_1",
        run_id="r1", run_name="eval-e2e-hybrid_1", checks=checks,
    )
    assert registry.get_folder_for_target(registry_file, "baseline") == "hybrid_1"
    record = registry.get_promotion_record(registry_file, "hybrid_1")
    assert record["run_id"] == "r1"
    assert record["run_name"] == "eval-e2e-hybrid_1"
    assert record["promoted_at"]
    assert record["checks"] == checks


def test_promoted_survives_unrelated_registry_write(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry.record_promotion(
        registry_file, "hybrid_1", run_id="r1", run_name="n", checks=[]
    )
    registry.update_status(registry_file, "hybrid_2", "complete")
    assert registry.get_promotion_record(registry_file, "hybrid_1")["run_id"] == "r1"
    assert registry.get_folder_for_target(registry_file, "baseline") == "hybrid_1"


def test_repromote_keeps_earlier_promoted_record(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry.record_promotion(
        registry_file, "hybrid_1", run_id="r1", run_name="n1", checks=[]
    )
    registry.record_promotion(
        registry_file, "hybrid_2", run_id="r2", run_name="n2", checks=[]
    )
    data = json.loads(registry_file.read_text())
    assert registry.get_folder_for_target(registry_file, "baseline") == "hybrid_2"
    assert set(data["promoted"]) == {"hybrid_1", "hybrid_2"}
    assert data["promoted"]["hybrid_1"]["run_id"] == "r1"


def test_promoted_section_survives_legacy_migration(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({
        "targets": {"current": "a", "latest": "a", "baseline": "a"},
        "status": {"a": "complete"},
        "datasets": {},
        "promoted": {"a": {"run_id": "r9", "run_name": "n", "promoted_at": "t", "checks": []}},
    }))
    registry.update_status(registry_file, "b", "building")
    assert registry.get_promotion_record(registry_file, "a")["run_id"] == "r9"
    assert registry.get_folder_for_target(registry_file, "baseline") == "a"


def test_missing_registry_has_empty_promoted(tmp_path):
    registry_file = tmp_path / "missing.json"
    assert registry.get_folder_for_target(registry_file, "baseline") is None
    assert registry.get_promotion_record(registry_file, "hybrid_1") is None
