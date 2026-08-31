"""End-to-end evaluation: answer golden rows through the full pipeline path.

The golden-context generator evaluator feeds each row's ground-truth recipes
straight to the prompt builder, isolating generation from retrieval. This
evaluator closes that gap. For every golden row the injected retrieve
callable runs the pipeline's own retrieval window and returns the recipe
dicts that would go into the serving prompt, deduped and capped at the
serving top-k. The prompt is built from those retrieved recipes rather than
the golden ones, the answer is generated greedily, and the same structural
checks and double-called LLM judge as the generation stage score it. Because
the judge context and the novelty reference are the retrieved recipes, every
score carries the retrieval outcome inside it, and the record keeps the
retrieved ids next to the scores so a bad answer is traceable to a retrieval
miss.
"""

import logging
from collections.abc import Sequence

from ..data.recipe_format import recipe_to_string
from .generator_evaluator import structural_checks
from .scoring import aggregate, budget_texts, judgment, mean


logger = logging.getLogger(__name__)


class EndToEndEvaluator:
    """Scores the full pipeline on golden rows, retrieval included.

    prompt_builder, generator, and judge are injected exactly as in the
    generator evaluator, and retrieve(query, k) returns recipe dicts in rank
    order. k is the serving top-k the context is built from, so it should
    match the snapshot's prompting configuration. The judge context and the
    novelty reference are the retrieved recipes, mirroring what the generator
    actually saw.
    """

    def __init__(
        self,
        prompt_builder,
        generator,
        judge,
        retrieve,
        k: int = 3,
        judge_calls: int = 2,
    ):
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.judge = judge
        self.retrieve = retrieve
        self.k = k
        self.judge_calls = judge_calls

    def evaluate(self, rows: Sequence[dict]) -> dict:
        """Score every row and aggregate over the slice and per query type.

        A row without a verified reference answer is scored on the three core
        dimensions only, since the expectedness judge needs the reference
        answer to compare against. The slice aggregate adds the retrieval
        diagnostics (golden recall and the both-retrieved rate) to the judged
        dimensions and structural rates.
        """
        per_row = [self._evaluate_row(row) for row in rows]
        query_types = sorted({row.get("query_type") for row in per_row})
        return {
            "per_row": per_row,
            "slice": self._aggregate(per_row),
            "query_type": {
                query_type: self._aggregate(
                    [r for r in per_row if r["query_type"] == query_type]
                )
                for query_type in query_types
            },
        }

    def _budgeted_texts(self, texts: Sequence[str]) -> list[str]:
        """The texts the prompt builder will keep under its token budget."""
        return budget_texts(
            texts,
            self.prompt_builder.tokenizer,
            self.prompt_builder.max_context_tokens,
        )

    def _evaluate_row(self, row: dict) -> dict:
        query = row["query"]
        golden_ids = list(row["recipe_ids"])
        reference_answer = row.get("answer")

        recipes = list(self.retrieve(query, self.k))
        texts = [recipe_to_string(recipe) for recipe in recipes]
        included = self._budgeted_texts(texts)
        context = "\n\n".join(included)

        system, user = self.prompt_builder.build_prompt_from_recipes(
            query=query,
            recipes=recipes,
            recipe_to_string=recipe_to_string,
        )
        response = self.generator.generate(system, user)

        flags = structural_checks(response, texts)

        core_verdicts = [
            self.judge.score_core(query, context, response)
            for _ in range(self.judge_calls)
        ]
        expectedness_verdicts = (
            [
                self.judge.score_expectedness(query, reference_answer, response)
                for _ in range(self.judge_calls)
            ]
            if reference_answer
            else []
        )
        scores, disagreement = judgment(core_verdicts, expectedness_verdicts)

        retrieved_ids = [recipe["recipe_id"] for recipe in recipes]
        golden = set(golden_ids)
        retrieved = set(retrieved_ids)
        golden_recall = (len(retrieved & golden) / len(golden)) if golden else 0.0
        retrieval_success = bool(golden and golden <= retrieved)

        return {
            "query": query,
            "query_type": row.get("query_type"),
            "recipe_ids": golden_ids,
            "retrieved_recipe_ids": retrieved_ids,
            "golden_recall": golden_recall,
            "retrieval_success": retrieval_success,
            "context_truncated": len(included) < len(texts),
            "generated": response,
            "structural": flags,
            "scores": scores,
            "disagreement": disagreement,
            "core_judge": [
                v.model_dump() if v else None for v in core_verdicts
            ],
            "expectedness_judge": [
                v.model_dump() if v else None for v in expectedness_verdicts
            ],
        }

    def _aggregate(self, rows: Sequence[dict]) -> dict:
        """Judged-dimension aggregates plus the retrieval diagnostics."""
        result = aggregate(rows)
        if rows:
            result["retrieval.golden_recall"] = mean(
                [row["golden_recall"] for row in rows]
            )
            result["retrieval.both_ground_truth_rate"] = mean(
                [1.0 if row["retrieval_success"] else 0.0 for row in rows]
            )
        return result
