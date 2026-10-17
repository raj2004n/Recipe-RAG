import json

from ..retrieval.dense.encoder import Encoder
from ..retrieval.dense.indexer import Indexer
from ..retrieval.dense.retriever import Retriever
from ..retrieval.sparse.bm25retriever import BM25Retriever
from ..retrieval.sparse.tokenizer import Tokenizer
from ..retrieval.hybrid.fuser import ChunkFuser, FusedChunk
from ..retrieval.hybrid.reranker import CrossEncoderReranker
from ..generation.prompt_builder import PromptBuilder
from ..generation.generator import Generator
from transformers import AutoTokenizer
from ..core.snapshot import Snapshot


class HybridPipeline:
    ALLOWED_STRATEGIES = ("hybrid",)

    def __init__(
        self,
        snapshot: Snapshot,
        recipes: list[dict] | None = None,
        recipes_path: str | None = None,
    ):
        if snapshot.retrieval_strategy not in self.ALLOWED_STRATEGIES:
            raise ValueError(
                f"HybridPipeline requires retrieval_strategy in "
                f"{self.ALLOWED_STRATEGIES}, got "
                f"{snapshot.retrieval_strategy!r}"
            )
        if recipes is None and recipes_path is None:
            raise ValueError(
                "HybridPipeline needs either `recipes` or `recipes_path`."
            )

        self._snapshot = snapshot

        cfg = snapshot.base_config
        hybrid_cfg = cfg["hybrid"]

        # dense retriever
        encoder = Encoder(
            model_id=cfg["encoder"]["model_id"],
            batch_size=cfg["encoder"]["batch_size"],
            normalize_embeddings=cfg["encoder"]["normalize_embeddings"],
        )
        indexer = Indexer(
            index_type=cfg["indexer"]["type"],
            dimension=cfg["indexer"]["dimension"],
            score_threshold=cfg["retriever"]["score_threshold"],
        )
        indexer.set_indexed_sets(snapshot.indexed_chunk_sets)
        self.dense_retriever = Retriever(
            encoder=encoder,
            indexer=indexer,
            top_k=hybrid_cfg["retrieval"]["top_k_dense"],
        )

        # sparse retriever
        if not snapshot.bm25_indexed_chunk_sets:
            raise RuntimeError(
                "HybridPipeline expected bm25_indexed_chunk_sets on the "
                "snapshot; none found. Did you build with strategy='hybrid'?"
            )
        self.bm25_retriever = BM25Retriever(
            indexed_chunk_set=snapshot.bm25_indexed_chunk_sets[0],
            tokenizer=Tokenizer(),
        )

        # fuse and rerank
        self.fuser = ChunkFuser(
            strategy=hybrid_cfg["fusion"]["strategy"],
            rrf_k=hybrid_cfg["fusion"]["rrf_k"],
            weight_dense=hybrid_cfg["fusion"]["weight_dense"],
            weight_sparse=hybrid_cfg["fusion"]["weight_sparse"],
        )
        self.reranker = CrossEncoderReranker(
            model_id=hybrid_cfg["reranker"]["model_id"],
            max_length=hybrid_cfg["reranker"]["max_length"],
        )
        self._rerank_top_n = hybrid_cfg["reranker"]["top_n"]
        self._final_top_k = hybrid_cfg["prompt"]["final_top_k"]

        # create recipes index
        self.recipes_by_id: dict[int, dict] = {}

        if recipes is not None:
            self.recipes_by_id = {r["recipe_id"]: r for r in recipes}
        elif recipes_path is not None:
            with open(recipes_path) as f:
                loaded = json.load(f)
            self.recipes_by_id = {r["recipe_id"]: r for r in loaded}

        # tokenizer, shared with prompt builder
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["generator"]["model_id"],
        )

        self.prompt_builder = PromptBuilder(
            tokenizer=self.tokenizer,
            max_context_tokens=snapshot.prompts_config["settings"]["max_context_tokens"],
            prompt_templates=snapshot.prompts_config["template"],
        )

        self.generator: Generator | None = None


    def load_generator(self) -> Generator:
        """Lazy Generator, cached after first call."""
        if self.generator is None:
            gen_cfg = self._snapshot.base_config["generator"]
            self.generator = Generator(
                model_id=gen_cfg["model_id"],
                max_input_tokens=gen_cfg["max_input_tokens"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                truncation=gen_cfg["truncation"],
                return_tensors=gen_cfg["return_tensors"],
                adapter_path=gen_cfg["adapter_path"],
                tokenizer=self.tokenizer,
            )
        return self.generator


    def query(self, query: str, k: int = 3) -> dict:
        """Full RAG query: fuse, rerank, fetch recipe, prompt, generate."""
        fused = self.fuser.fuse(
            dense=self.dense_retriever.retrieve(query),
            sparse=self.bm25_retriever.retrieve(query),
        )
        reranked = self.reranker.rerank(query, fused, top_n=self._rerank_top_n)

        recipes = self._fetch_recipes(reranked, k=k)
        system, user = self.prompt_builder.build_prompt_from_recipes(
            query=query,
            recipes=recipes,
        )
        answer = self.load_generator().generate(system, user)

        return {
            "query": query,
            "answer": answer,
            "prompt_used": (system, user),
            "fused_top": [
                {
                    "chunk_id": fc.chunk.chunk_id,
                    "recipe_id": fc.chunk.recipe_id,
                    "fused_score": fc.fused_score,
                    "sources": list(fc.sources),
                }
                for fc in reranked[:k]
            ],
        }


    def debug_query(self, query: str, k: int = 3) -> tuple[list[dict], str, tuple[str, str]]:
        """Retrieval-only split: fuse, rerank, and resolve chunks to whole recipes.

        Mirrors DensePipeline.debug_query: it runs the full retrieval path but
        stops before generation. Returns the deduped recipes the generator
        would see, in rank order, plus the prompt that would be built from
        them. The window is capped by the reranker's top_n, so a k larger
        than that returns at most that many recipes.
        """
        fused = self.fuser.fuse(
            dense=self.dense_retriever.retrieve(query),
            sparse=self.bm25_retriever.retrieve(query),
        )
        reranked = self.reranker.rerank(query, fused, top_n=self._rerank_top_n)

        recipes = self._fetch_recipes(reranked, k=k)
        prompts = self.prompt_builder.build_prompt_from_recipes(
            query=query,
            recipes=recipes,
        )

        return recipes, "", prompts


    def _fetch_recipes(
        self,
        reranked: list[FusedChunk],
        k: int,
    ) -> list[dict]:
        """Helper method to resolve chunks back to whole recipes.

        Dedupe by `recipe_id`,a recipe can have multiple chunks (ing, it, info)
        Keep first-seen order
        """
        seen: set[int] = set()
        out: list[dict] = []

        for fc in reranked:
            recipe_id = fc.chunk.recipe_id
            if recipe_id in seen:
                continue
            recipe = self.recipes_by_id.get(recipe_id)
            if recipe is None:
                continue
            seen.add(recipe_id)
            out.append(recipe)
            if len(out) >= k:
                break

        return out
