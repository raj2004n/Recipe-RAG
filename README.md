# Recipe-RAG

A recipe question-answering system built around a versioned retrieval pipeline.

It crawls recipes from [recipetin eats](https://www.recipetineats.com), cleans them into a
normalised form, and serves a query interface that pulls relevant chunks out of the
corpus and writes a short answer in the voice of the original site. Two retrieval
strategies - dense and hybrid - are implemented and selectable per build.

## What this is

The interesting part isn't the recipe corpus; it's the pipeline underneath it. Each
build produces a **sealed snapshot** - an immutable, content-hashed folder containing
the chunk sets, the FAISS and BM25 indices, the model config, and a copy of the configs
that produced it. A central registry tracks which snapshot is `current` and which is
`latest`. Nothing in the snapshot folder ever changes after it's sealed; new builds
write new folders, and the registry is the single source of truth for what's live.

Two retrieval strategies ride on top of this:

- **Dense** embeds whole recipes with sentence-transformers and indexes them with a
  flat inner-product FAISS index. Simple, fast, predictable.
- **Hybrid** runs BM25 over an ingredients+instructions chunk set, fuses it with a
  dense retriever over a separate info chunk set using reciprocal rank fusion,
  reranks the merged list with a cross-encoder, and resolves back to whole recipes
  before prompting the generator.

Both feed the same generator. The build side picks the strategy; the inference side
just loads `latest` and answers.

## Why these choices

A snapshot subsystem rather than ad-hoc files: every build is reproducible from the
registry alone. If a new build ships bad output, the previous one is a registry
edit away.

A registry rather than a `snapshots/latest/` symlink: symlinks don't survive being
copied across machines or DVC remotes. The registry is a tiny JSON file that any
process can read.

Hybrid with a cross-encoder reranker: dense alone misses exact-ingredient matches
("I have chicken thighs and soy sauce"); BM25 alone misses semantic matches
("something comforting for a cold evening"). The cross-encoder is what lets the
hybrid path beat either single retriever on realistic recipe queries.

## Quickstart

```bash
# Clone and set up
git clone <repo-url>
cd Recipe-RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt

# Pull data and build the index end-to-end
dvc repro

# Or step-by-step if you want to watch each stage
python -m scripts.build_rag --run set_strategy
python -m scripts.build_rag --run chunk
python -m scripts.build_rag --run transform
python -m scripts.build_rag --run index
python -m scripts.build_rag --run finalise_build

# Serve queries over HTTP
export RECIPE_RAG_API_KEYS='{"dev-admin-key":"admin","dev-public-key":"public"}'
uvicorn src.rag.serve.app:create_app --factory
# then: curl -H 'X-API-Key: dev-public-key' -H 'content-type: application/json' \
#   -d '{"query":"quick chicken dinner"}' http://localhost:8000/query
```

You'll need a local Ollama instance running `gemma3:4b` for the eval-dataset
generator; the inference-time generator is loaded in-process.

## Architecture

```
sources (Scrapy spider)
    │  raw jsonl
    ▼
preprocess → cleaned Recipe objects
    │
    ▼
[chunk → transform → index → seal_snapshot]   ← the 4-stage build pipeline
    │
    ▼
sealed snapshot under snapshots/<name>/
    │
    ▼
[DensePipeline | HybridPipeline]   ← loaded from snapshots/registry.json
    │
    ▼
answer
```

The build side is orchestrator-only - chunk-set assembly happens here. The
inference side is naive about chunk-set filtering - it loads whatever the
registry points at. The cross-encoder reranker is a per-query inference
concern; it doesn't live in the build.

MLflow tracks the parameters and artifacts of every build run locally. The
snapshot registry is the deployment-side contract; the build is the audit
trail.

## What's interesting if you're hiring

- A real **snapshot subsystem with a registry**, not a `models/v3.pt` file.
- **Two retrieval strategies** that share a generator, with a build-time
  decision about which chunk sets to produce.
- **Eval-gated promotion**: a held-out eval pass is what decides whether a
  new snapshot gets the `latest` label. The build alone doesn't ship.
- **Sealed, content-hashed snapshots**: every build is reproducible from the
  registry alone.
- A **4-stage build pipeline** orchestrated through DVC, with MLflow tracking
  every run.

## Shipping the baseline to this machine

Promotion pins a snapshot as the deployment baseline in the registry, and the
serve image with that baseline baked in runs on this machine. To build and start
it:

```bash
export RECIPE_RAG_API_KEYS='{"dev-admin-key":"admin","dev-public-key":"public"}'
python -m scripts.ship_baseline
```

The script resolves the promoted baseline from the registry, builds the serve
image with it baked in (the Dockerfile's `ARG SNAPSHOT`), starts a
`docker run --gpus all` container mapping port 8000 to the host, and waits for
`/health` to report ok before returning. Re-running it replaces the container,
so shipping a new baseline is the same command. It fails closed: no baseline
target in the registry, a baseline that is not sealed or not promoted, missing
`RECIPE_RAG_API_KEYS`, or a service that never reports healthy each stop the
ship and exit non-zero.

The script calls plain `docker`. If your user is not in the `docker` group,
either run each docker call via `sudo` or add yourself to the group once with
`sudo usermod -aG docker $USER` and log back in (docker group membership is
root-equivalent access to the host).

## Status

Working end-to-end on a developer machine. The build, retrieval, and generation
loops all run; snapshots seal and load cleanly.

Not done yet:

- Dockerfile and containerised deploy.
- CI / CD and a held-out eval stage for promotion gating.

The HTTP service (`/health`, `/info`, `POST /query`) and the two-role API key
auth are already wired up; the gap between "runs on a developer machine" and
"deployed" is now the eval stage, the container, and the workflows.

A deployment plan for the above lives in `.lavish/deployment-readiness.html`.
The short version: README → split requirements → tests → FastAPI service
with auth → eval stage → Dockerfile → CI → build image → release → gated
promotion.

## Author

Raj Negi

[contact]
