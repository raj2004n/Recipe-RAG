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
serve image with that baseline baked in runs on this machine. To pull the
promoted baseline and start it:

```bash
export RECIPE_RAG_API_KEYS='{"dev-admin-key":"admin","dev-public-key":"public"}'
python -m scripts.ship_baseline
```

The script resolves the promoted baseline from the registry, pulls the image
tagged `<baseline folder>` from ghcr, starts a `docker run --gpus all`
container mapping port 8000 to the host, and waits for `/health` to report ok
before returning. Re-running it replaces the container, so shipping a new
baseline is the same command. It fails closed: no baseline target in the
registry, a baseline that is not sealed or not promoted, an image that cannot
be pulled, missing `RECIPE_RAG_API_KEYS`, or a service that never reports
healthy each stop the ship and exit non-zero.

If the image is not in ghcr yet, run the build workflow to push it first, or
point the script at a locally built image with
`python -m scripts.ship_baseline --image-repo recipe-rag`.

The script calls plain `docker`. If your user is not in the `docker` group,
either run each docker call via `sudo` or add yourself to the group once with
`sudo usermod -aG docker $USER` and log back in (docker group membership is
root-equivalent access to the host).

## CI / CD

Three GitHub Actions workflows drive the pipeline:

- `ci.yml` runs ruff and the full test suite on GitHub-hosted runners for
  every push to main and every pull request. The tests use fixtures, so they
  pass on a clean runner with no snapshot, dataset, or model weights present.
- `build.yml` runs on a self-hosted runner on this machine and fires only after
  `ci.yml` passes on a push to main. It stages the model weights, resolves the
  promoted baseline from the registry, builds the serve image with
  `ARG SNAPSHOT=<baseline>`, and pushes it to `ghcr.io/raj2004n/recipe-rag` as
  `<git sha>`, `latest`, and `<baseline folder>`.
- `deploy.yml` is a manual `workflow_dispatch`. It pulls
  `recipe-rag:<baseline folder>` and runs it on this machine, replacing any
  running container.

The build and deploy jobs run in the machine's checkout rather than a fresh
clone, because the sealed snapshot, the dataset, the registry, and the staged
weights all live only on this machine. The image's payload is split: git
carries code, configs, and requirements, while a local DVC remote at
`/home/raj/dvc_storage` carries the snapshot contents and the model weights are
staged by `scripts/stage_weights.sh` from the HF cache. GitHub-hosted runners
therefore cannot build the serve image at all.

Setup for the self-hosted path:

1. Register a self-hosted runner at repository scope with the labels
   `self-hosted, gpu, linux`. The runner user needs the `docker` group and
   read/write access to the checkout at `/home/raj/projects/Recipe-RAG`.
2. Add a repository secret `RECIPE_RAG_API_KEYS` holding the API-key JSON the
   deployed service must accept.
3. The image package on ghcr is public, so the host can pull it without a
   token after the workflow's short-lived checkout token expires.

## Status

Working end-to-end on a developer machine. The build, retrieval, and generation
loops all run; snapshots seal and load cleanly. The serve image builds and has
been smoke-validated on this machine, the promotion gate records a baseline in
the registry, and the CI / CD workflows are in place.

Not done yet:

- Registering the self-hosted runner and adding the `RECIPE_RAG_API_KEYS`
  secret (the two GitHub-side setup steps).
- Validating the workflows on a real push to main.

The HTTP service (`/health`, `/info`, `POST /query`) and the two-role API key
auth are wired up; the remaining gap is the GitHub-side runner registration and
the first end-to-end workflow run.

## Author

Raj Negi

[contact]
