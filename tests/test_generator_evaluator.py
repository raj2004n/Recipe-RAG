"""Tests for the golden-context generator evaluator and its judge client."""

import json
from types import SimpleNamespace

import pytest

import src.rag.evaluation.generator_evaluator as gen_eval
from src.rag.evaluation.generator_evaluator import (
    CoreVerdict,
    ExpectednessVerdict,
    GeneratorEvaluator,
    JudgeClient,
    structural_checks,
)
from src.rag.evaluation.judge_prompts import judge_messages, judge_prompt_hash


WELL_FORMED = (
    "Beef Curry Sausage Patties\n"
    "Servings: 4\n"
    "\n"
    "Ingredients\n"
    "- 500 g beef mince\n"
    "- 1 egg\n"
    "- 2 garlic cloves, minced\n"
    "- 1 tbsp curry powder\n"
    "\n"
    "Instructions\n"
    "1. Mix the mince, egg, garlic, and curry powder in a bowl.\n"
    "2. Shape into eight patties with wet hands.\n"
    "3. Pan-fry for 3 minutes per side until browned and cooked through.\n"
    "\n"
    "Note: the curry powder replaces the sage; keep the thyme so the\n"
    "patties still taste like the original.\n"
)

# Shares a couple of ingredient lines with WELL_FORMED, like a real adapted
# recipe does, but not long enough word runs to look like a copy.
REFERENCE = (
    "Recipe Name: Sage Sausage Patties\n"
    "Ingredients:\n"
    "- 500 g beef mince\n"
    "- 1 egg\n"
    "- 1 tbsp dried sage\n"
    "- 1 tsp thyme leaves\n"
    "Instructions:\n"
    "- Mix everything in a bowl\n"
    "- Pan-fry the patties until browned\n"
)


# ---------------------------------------------------------------------------
# structural checks


def test_structural_checks_pass_on_wellformed_answer():
    flags = structural_checks(WELL_FORMED, [REFERENCE])
    assert flags["structure_pass"] is True
    assert all(value for value in flags.values())


def test_missing_name_line_flagged():
    anonymous = "Ingredients\n" + WELL_FORMED.split("Ingredients\n", 1)[1]
    assert structural_checks(anonymous, [REFERENCE])["has_name"] is False


def test_chatty_preamble_flagged():
    chatty = (
        "Okay, here is a recipe for you, adapted from the references.\n"
        + WELL_FORMED
    )
    assert structural_checks(chatty, [REFERENCE])["has_name"] is False


def test_markdown_decorated_sections_tolerated():
    decorated = WELL_FORMED.replace("Ingredients\n", "**Ingredients:**\n")
    decorated = decorated.replace("Instructions\n", "**Instructions:**\n")
    decorated = decorated.replace("- 500 g beef mince", "*   500g beef mince")
    assert structural_checks(decorated, [REFERENCE])["ingredients_with_amounts"] is True
    assert structural_checks(decorated, [REFERENCE])["structure_pass"] is True


def test_ingredients_without_quantities_flagged():
    vague = WELL_FORMED.replace("500 g beef mince", "some beef mince")
    vague = vague.replace("1 egg", "an egg")
    vague = vague.replace("2 garlic cloves, minced", "garlic")
    vague = vague.replace("1 tbsp curry powder", "curry powder")
    assert structural_checks(vague, [REFERENCE])["ingredients_with_amounts"] is False


def test_unnumbered_steps_flagged():
    bulleted = WELL_FORMED.replace("1. Mix", "- Mix")
    bulleted = bulleted.replace("2. Shape", "- Shape")
    bulleted = bulleted.replace("3. Pan-fry", "- Pan-fry")
    assert structural_checks(bulleted, [REFERENCE])["numbered_steps"] is False


def test_markdown_fence_flagged():
    fenced = f"```json\n{WELL_FORMED}\n```"
    assert structural_checks(fenced, [REFERENCE])["no_markdown_fence"] is False


def test_short_answer_flagged():
    short = "Beef patties\nIngredients\n- beef\nInstructions\n1. Fry it."
    assert structural_checks(short, [REFERENCE])["length_in_bounds"] is False


def test_verbatim_copy_flagged():
    # The response is the reference itself, the loudest possible echo.
    flags = structural_checks(REFERENCE, [REFERENCE])
    assert flags["not_verbatim_copy"] is False
    assert flags["structure_pass"] is False


def test_adapted_answer_reusing_reference_lines_not_flagged():
    flags = structural_checks(WELL_FORMED, [REFERENCE])
    assert flags["not_verbatim_copy"] is True


# ---------------------------------------------------------------------------
# judge prompts


def test_judge_messages_have_both_personas():
    messages = judge_messages()
    assert set(messages) == {"core", "expectedness"}
    assert all(message.strip() for message in messages.values())


def test_judge_prompt_hash_is_stable():
    first = judge_prompt_hash()
    assert first == judge_prompt_hash()
    assert len(first) == 16


# ---------------------------------------------------------------------------
# judge client


def _fake_chat(content, calls):
    def chat(model, messages, format, options):
        calls.append(
            {"model": model, "messages": messages, "format": format,
             "options": options}
        )
        return SimpleNamespace(message=SimpleNamespace(content=content))
    return chat


def _verdict_json(**scores):
    payload = {
        "constraint_satisfaction": scores.get("constraint_satisfaction", 2),
        "query_relevance": scores.get("query_relevance", 2),
        "faithfulness": scores.get("faithfulness", 2),
        "reasons": {
            "constraint_satisfaction": "ok",
            "query_relevance": "ok",
            "faithfulness": "ok",
        },
    }
    return json.dumps(payload)


def test_judge_client_parses_and_passes_options(monkeypatch):
    monkeypatch.setattr(gen_eval.time, "sleep", lambda *_: None)
    calls = []
    monkeypatch.setattr(
        gen_eval, "chat", _fake_chat(_verdict_json(), calls)
    )
    client = JudgeClient(
        model_id="minimax-m3:cloud", temperature=0.2, seed=42
    )
    verdict = client.score_core("query", "context", "response")

    assert isinstance(verdict, CoreVerdict)
    assert verdict.constraint_satisfaction == 2
    assert len(calls) == 1
    assert calls[0]["model"] == "minimax-m3:cloud"
    assert calls[0]["format"] == "json"
    assert calls[0]["options"]["temperature"] == 0.2
    assert calls[0]["options"]["seed"] == 42
    assert calls[0]["messages"][0]["role"] == "system"
    assert "constraint_satisfaction" in calls[0]["messages"][0]["content"]


def test_judge_client_strips_markdown_fence(monkeypatch):
    monkeypatch.setattr(gen_eval.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        gen_eval, "chat", _fake_chat(f"```json\n{_verdict_json()}\n```", [])
    )
    verdict = JudgeClient(model_id="m").score_core("q", "c", "r")
    assert verdict is not None and verdict.faithfulness == 2


def test_judge_client_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr(gen_eval.time, "sleep", lambda *_: None)
    attempts = []
    monkeypatch.setattr(
        gen_eval, "chat", _fake_chat("not json at all", attempts)
    )
    client = JudgeClient(model_id="m", max_retries=3)
    assert client.score_core("q", "c", "r") is None
    assert len(attempts) == 3


def test_judge_client_expectedness_parse(monkeypatch):
    monkeypatch.setattr(gen_eval.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        gen_eval, "chat",
        _fake_chat(json.dumps({"expectedness": 1, "reason": "close"}), [])
    )
    verdict = JudgeClient(model_id="m").score_expectedness("q", "golden", "r")
    assert isinstance(verdict, ExpectednessVerdict)
    assert verdict.expectedness == 1


def test_judge_client_rejects_out_of_range_scores(monkeypatch):
    monkeypatch.setattr(gen_eval.time, "sleep", lambda *_: None)
    attempts = []
    monkeypatch.setattr(
        gen_eval, "chat", _fake_chat(_verdict_json(constraint_satisfaction=7), attempts)
    )
    client = JudgeClient(model_id="m", max_retries=2)
    # 7 is out of the 0-2 rubric, so validation fails and the call retries.
    assert client.score_core("q", "c", "r") is None
    assert len(attempts) == 2


# ---------------------------------------------------------------------------
# evaluator aggregation


class _Tokenizer:
    def encode(self, text):
        return text.split()


class _PromptBuilder:
    def __init__(self, max_context_tokens=4000):
        self.tokenizer = _Tokenizer()
        self.max_context_tokens = max_context_tokens

    def build_prompt_from_recipes(self, query, recipes):
        return "system", "user: " + query


class _Generator:
    def generate(self, system, user):
        return WELL_FORMED


class _StubJudge:
    """Returns queued verdicts, one per call, cycling per row."""

    def __init__(self, core, expectedness):
        self._core = list(core)
        self._expectedness = list(expectedness)

    def score_core(self, query, context, response):
        return self._core.pop(0)

    def score_expectedness(self, query, reference_answer, response):
        return self._expectedness.pop(0)


def _recipe(recipe_id):
    return {
        "recipe_id": recipe_id,
        "recipe_name": f"Recipe {recipe_id}",
        "ingredient_groups": [
            {"ingredients": [{"name": "beef", "amount": "500", "unit": "g"}]}
        ],
        "instruction_groups": [
            {"instructions": ["Mix everything", "Fry the patties"]}
        ],
    }


def _row(recipe_id, query_type):
    return {
        "query": f"adapt recipe {recipe_id}",
        "query_type": query_type,
        "recipe_ids": [recipe_id, recipe_id + 100],
        "answer": "a reference answer",
    }


def test_evaluator_aggregates_means_disagreement_and_query_types():
    judge = _StubJudge(
        core=[
            CoreVerdict(constraint_satisfaction=2, query_relevance=2, faithfulness=2),
            CoreVerdict(constraint_satisfaction=2, query_relevance=1, faithfulness=2),
            CoreVerdict(constraint_satisfaction=1, query_relevance=1, faithfulness=1),
            CoreVerdict(constraint_satisfaction=1, query_relevance=1, faithfulness=1),
        ],
        expectedness=[
            ExpectednessVerdict(expectedness=2),
            ExpectednessVerdict(expectedness=2),
            ExpectednessVerdict(expectedness=0),
            ExpectednessVerdict(expectedness=2),
        ],
    )
    recipes_by_id = {i: _recipe(i) for i in (1, 101, 2, 102)}
    evaluator = GeneratorEvaluator(
        prompt_builder=_PromptBuilder(),
        generator=_Generator(),
        judge=judge,
        recipes_by_id=recipes_by_id,
        judge_calls=2,
    )

    result = evaluator.evaluate([_row(1, "recipe_variation"), _row(2, "recipe_combination")])

    slice_scores = result["slice"]
    assert slice_scores["mean.constraint_satisfaction"] == pytest.approx(1.5)
    assert slice_scores["mean.query_relevance"] == pytest.approx(1.25)
    # Row 1's two calls disagreed on relevance; row 2's agreed.
    assert slice_scores["disagreement.query_relevance"] == pytest.approx(0.5)
    assert slice_scores["disagreement.expectedness"] == pytest.approx(0.5)
    assert slice_scores["structure_pass_rate"] == pytest.approx(1.0)

    variation = result["query_type"]["recipe_variation"]
    assert variation["mean.constraint_satisfaction"] == pytest.approx(2.0)
    assert variation["disagreement.query_relevance"] == pytest.approx(1.0)

    combination = result["query_type"]["recipe_combination"]
    assert combination["mean.constraint_satisfaction"] == pytest.approx(1.0)
    assert combination["disagreement.query_relevance"] == pytest.approx(0.0)

    row1 = result["per_row"][0]
    assert row1["scores"]["query_relevance"] == pytest.approx(1.5)
    assert row1["context_truncated"] is False
    assert row1["missing_recipe_ids"] == []
    assert row1["structural"]["structure_pass"] is True


def test_evaluator_flags_context_truncation():
    judge = _StubJudge(
        core=[CoreVerdict(constraint_satisfaction=2, query_relevance=2, faithfulness=2)] * 2,
        expectedness=[ExpectednessVerdict(expectedness=2)] * 2,
    )
    recipes_by_id = {i: _recipe(i) for i in (1, 101)}
    # A budget of five words cannot hold either formatted recipe.
    evaluator = GeneratorEvaluator(
        prompt_builder=_PromptBuilder(max_context_tokens=5),
        generator=_Generator(),
        judge=judge,
        recipes_by_id=recipes_by_id,
        judge_calls=2,
    )
    row = evaluator.evaluate([_row(1, "recipe_variation")])["per_row"][0]
    assert row["context_truncated"] is True


def test_evaluator_records_missing_recipe_ids():
    judge = _StubJudge(
        core=[CoreVerdict(constraint_satisfaction=2, query_relevance=2, faithfulness=2)] * 2,
        expectedness=[ExpectednessVerdict(expectedness=2)] * 2,
    )
    # Only recipe 1 resolves; 101 is absent from the corpus.
    evaluator = GeneratorEvaluator(
        prompt_builder=_PromptBuilder(),
        generator=_Generator(),
        judge=judge,
        recipes_by_id={1: _recipe(1)},
        judge_calls=2,
    )
    row = evaluator.evaluate([_row(1, "recipe_variation")])["per_row"][0]
    assert row["missing_recipe_ids"] == [101]
    assert row["recipe_ids"] == [1, 101]


def test_evaluator_scores_none_when_all_judge_calls_fail():
    judge = _StubJudge(core=[None, None], expectedness=[None, None])
    evaluator = GeneratorEvaluator(
        prompt_builder=_PromptBuilder(),
        generator=_Generator(),
        judge=judge,
        recipes_by_id={1: _recipe(1), 101: _recipe(101)},
        judge_calls=2,
    )
    result = evaluator.evaluate([_row(1, "recipe_variation")])
    row = result["per_row"][0]
    assert row["scores"]["constraint_satisfaction"] is None
    assert row["disagreement"]["constraint_satisfaction"] is None
    # No successful judge calls, so no dimension means at all.
    assert "mean.constraint_satisfaction" not in result["slice"]