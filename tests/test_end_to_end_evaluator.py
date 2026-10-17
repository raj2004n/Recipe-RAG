import pytest

from src.evaluation.end_to_end_evaluator import EndToEndEvaluator
from src.evaluation.generator_evaluator import CoreVerdict, ExpectednessVerdict
from src.evaluation.scoring import JUDGED_DIMENSIONS


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


class _Tokenizer:
    def encode(self, text):
        return text.split()


class _PromptBuilder:
    def __init__(self, max_context_tokens=2000):
        self.tokenizer = _Tokenizer()
        self.max_context_tokens = max_context_tokens

    def build_prompt_from_recipes(self, query, recipes):
        return "system", "user: " + query


class _Generator:
    def generate(self, system, user):
        return WELL_FORMED


class _StubJudge:
    """Returns queued verdicts, one per call, cycling per row."""

    def __init__(self, core, expectedness=()):
        self._core = list(core)
        self._expectedness = list(expectedness)
        self.core_calls = []
        self.expectedness_calls = []

    def score_core(self, query, context, response):
        self.core_calls.append((query, context, response))
        return self._core.pop(0)

    def score_expectedness(self, query, reference_answer, response):
        self.expectedness_calls.append(query)
        if not self._expectedness:
            return ExpectednessVerdict(expectedness=2)
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


def _row(recipe_id, query_type="recipe_variation", with_answer=True):
    row = {
        "query": f"adapt recipe {recipe_id}",
        "query_type": query_type,
        "recipe_ids": [recipe_id, recipe_id + 100],
    }
    if with_answer:
        row["answer"] = "a reference answer"
    return row


def _evaluator(retrieve, judge, max_context_tokens=2000):
    return EndToEndEvaluator(
        prompt_builder=_PromptBuilder(max_context_tokens),
        generator=_Generator(),
        judge=judge,
        retrieve=retrieve,
        k=3,
        judge_calls=2,
    )


def _retrieve(*recipe_lists):
    """A retrieve callable serving the given recipe lists in rank order."""
    queue = list(recipe_lists)
    return lambda query, k: queue.pop(0)


def _both_core_verdicts():
    return [
        CoreVerdict(constraint_satisfaction=2, query_relevance=2, faithfulness=2),
        CoreVerdict(constraint_satisfaction=2, query_relevance=2, faithfulness=2),
    ]


# ---------------------------------------------------------------------------
# retrieval diagnostics


def test_evaluator_records_both_golden_ids_retrieved():
    judge = _StubJudge(core=_both_core_verdicts())
    row = _evaluator(
        _retrieve([_recipe(1), _recipe(101)]), judge
    ).evaluate([_row(1)])["per_row"][0]

    assert row["retrieved_recipe_ids"] == [1, 101]
    assert row["golden_recall"] == 1.0
    assert row["retrieval_success"] is True


def test_evaluator_records_partial_retrieval():
    judge = _StubJudge(core=_both_core_verdicts())
    row = _evaluator(
        _retrieve([_recipe(1), _recipe(999)]), judge
    ).evaluate([_row(1)])["per_row"][0]

    assert row["retrieved_recipe_ids"] == [1, 999]
    assert row["golden_recall"] == 0.5
    assert row["retrieval_success"] is False


def test_evaluator_scores_when_no_golden_id_retrieved():
    judge = _StubJudge(core=_both_core_verdicts())
    result = _evaluator(
        _retrieve([_recipe(200), _recipe(300)]), judge
    ).evaluate([_row(1)])
    row = result["per_row"][0]

    assert row["golden_recall"] == 0.0
    assert row["retrieval_success"] is False
    # The answer is still judged on the served context, which is not the query's
    # ground truth.
    assert row["scores"]["constraint_satisfaction"] == 2.0
    assert result["slice"]["retrieval.golden_recall"] == 0.0


def test_evaluator_preserves_retrieval_rank_order():
    judge = _StubJudge(core=_both_core_verdicts())
    row = _evaluator(
        _retrieve([_recipe(3), _recipe(1)]), judge
    ).evaluate([_row(1)])["per_row"][0]

    assert row["retrieved_recipe_ids"] == [3, 1]
    assert row["golden_recall"] == 0.5


def test_judge_context_is_the_retrieved_recipes_not_the_golden_pair():
    judge = _StubJudge(core=_both_core_verdicts())
    # The golden pair is [1, 101], but retrieval serves 3 and 7.
    _evaluator(
        _retrieve([_recipe(3), _recipe(7)]), judge
    ).evaluate([_row(1)])

    context = judge.core_calls[0][1]
    assert "Recipe Name: Recipe 3" in context
    assert "Recipe Name: Recipe 7" in context
    assert "Recipe Name: Recipe 1" not in context


# ---------------------------------------------------------------------------
# judged-dimension aggregation


def test_evaluator_aggregates_judged_dimensions_and_retrieval():
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
    retrieve = _retrieve(
        [_recipe(1), _recipe(101)],
        [_recipe(2), _recipe(102)],
    )
    result = _evaluator(retrieve, judge).evaluate(
        [_row(1, "recipe_variation"), _row(2, "recipe_combination")]
    )

    slice_scores = result["slice"]
    assert slice_scores["mean.constraint_satisfaction"] == pytest.approx(1.5)
    assert slice_scores["mean.query_relevance"] == pytest.approx(1.25)
    assert slice_scores["mean.expectedness"] == pytest.approx(1.5)
    assert slice_scores["disagreement.query_relevance"] == pytest.approx(0.5)
    assert slice_scores["disagreement.expectedness"] == pytest.approx(0.5)
    assert slice_scores["structure_pass_rate"] == pytest.approx(1.0)
    assert slice_scores["retrieval.golden_recall"] == pytest.approx(1.0)
    assert slice_scores["retrieval.both_ground_truth_rate"] == pytest.approx(1.0)

    variation = result["query_type"]["recipe_variation"]
    assert variation["mean.constraint_satisfaction"] == pytest.approx(2.0)
    assert variation["disagreement.query_relevance"] == pytest.approx(1.0)

    combination = result["query_type"]["recipe_combination"]
    assert combination["mean.constraint_satisfaction"] == pytest.approx(1.0)
    assert combination["disagreement.query_relevance"] == pytest.approx(0.0)

    row1 = result["per_row"][0]
    assert row1["scores"]["query_relevance"] == pytest.approx(1.5)
    assert row1["context_truncated"] is False


def test_evaluator_flags_context_truncation():
    judge = _StubJudge(core=_both_core_verdicts())
    # A budget of eight tokens holds the first recipe's render but drops the
    # second, exactly as the prompt builder would.
    row = _evaluator(
        _retrieve([_recipe(1), _recipe(101)]),
        judge,
        max_context_tokens=8,
    ).evaluate([_row(1)])["per_row"][0]

    assert row["context_truncated"] is True
    assert row["scores"]["constraint_satisfaction"] == 2.0


def test_evaluator_scores_no_expectedness_without_reference_answer():
    judge = _StubJudge(core=_both_core_verdicts())
    row = _evaluator(
        _retrieve([_recipe(1), _recipe(101)]), judge
    ).evaluate([_row(1, with_answer=False)])["per_row"][0]

    assert row["scores"]["expectedness"] is None
    assert row["disagreement"]["expectedness"] is None
    assert judge.expectedness_calls == []
    # The three core dimensions are still judged.
    assert row["scores"]["faithfulness"] == 2.0


def test_slice_has_no_dimension_means_when_all_judge_calls_fail():
    judge = _StubJudge(core=[None, None], expectedness=[None, None])
    result = _evaluator(
        _retrieve([_recipe(1), _recipe(101)]), judge
    ).evaluate([_row(1)])

    assert "mean.constraint_satisfaction" not in result["slice"]
    assert "mean.expectedness" not in result["slice"]
    # Retrieval diagnostics do not depend on the judge.
    assert result["slice"]["retrieval.golden_recall"] == pytest.approx(1.0)


def test_judged_dimensions_complete():
    assert set(JUDGED_DIMENSIONS) == {
        "constraint_satisfaction",
        "query_relevance",
        "faithfulness",
        "expectedness",
    }
