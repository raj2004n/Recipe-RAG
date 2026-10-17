# Recipe-RAG
## Let him cook.
-- will insert picture here --
## What it is

I like cooking, but after a full day of work, I don't like thinking. So I made this retrieval-augmented-generation (RAG) system to do this for me! I give it a prompt, it retrieves the relevant recipes, and a small local LLM writes a recipe. The ``How it Works`` section covers the project in detail and explains why each decision was made.


## Quickstart

You need Python 3.12 and a CUDA GPU to serve. Ollama is only needed for the eval and drafting
passes, and Docker only for the image path at the end.

```bash
git clone <repo-url>
cd recipe-rag
python -m venv .venv && source .venv/bin/activate
```

The torch wheels are cu124 builds that PyPI does not carry, and `faiss-gpu`'s metadata asks for a
newer CUDA runtime than those wheels provide even though the pair runs fine together. The
Dockerfile and CI both resolve faiss separately with `--no-deps`, and a local venv needs the same:

```bash
export PIP_INDEX_URL=https://download.pytorch.org/whl/cu124
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
grep -v '^faiss-gpu' requirements/base.txt > /tmp/base.no-faiss.txt
grep -v '^-r base.txt' requirements/serve.txt > /tmp/serve.extras.txt
grep -vE '^(-r serve\.txt|-r base\.txt)' requirements/dev.txt > /tmp/dev.extras.txt
python -m pip install -r /tmp/base.no-faiss.txt -r /tmp/serve.extras.txt -r /tmp/dev.extras.txt
python -m pip install --no-deps faiss-gpu==1.14.3
```

`requirements/dev.txt` is the set for local work. The served image installs only
`requirements/serve.txt`, which layers on `base.txt`.


## How it Works

```
recipetineats.com
    │  crawl, then clean into one validated corpus
    ▼
the corpus  (1,494 recipes)
    │  build: chunk → transform → index → seal
    ▼
a sealed snapshot  (immutable, versioned, pinned to the corpus)
    │  loaded once, at boot
    ▼
retrieve → fuse → rerank → prompt → answer
```

--Will write soon...--

### Build a snapshot

```bash
dvc repro
```

That crawls the site, preprocesses the crawl, runs the four build stages, and seals the result.
It is the whole pipeline in one command, and DVC skips any stage whose declared outputs are
unchanged. To watch the stages individually, or to iterate on one of them:

```bash
python -m scripts.build_rag --run set_strategy   # register the dataset, fix the strategy
python -m scripts.build_rag --run chunk          # chunk sets for the chosen strategy
python -m scripts.build_rag --run transform      # encode the dense sets, tokenise the sparse ones
python -m scripts.build_rag --run index          # build the FAISS and BM25 indices
python -m scripts.build_rag --run finalise_build # seal the staging tree into a versioned folder
```

`python -m scripts.build_rag --run build` wraps the same stages in a single MLflow run, and
`--run test_build` runs all five without MLflow.

### Serve it

```bash
export RECIPE_RAG_API_KEYS='{"dev-admin-key":"admin","dev-public-key":"public"}'
uvicorn src.serve.app:create_app --factory
```

```bash
curl -H 'X-API-Key: dev-public-key' -H 'content-type: application/json' \
  -d '{"query":"quick chicken dinner"}' http://localhost:8000/query
```

Open `http://localhost:8000/` for the browser interface. The page carries no key of its own and
forwards through keyless routes, so anything that can reach the port can spend GPU time; the
deployed container therefore publishes on loopback only.

### Tests and evaluation

```bash
pytest -q
python -m scripts.evaluate_retrieval --snapshot latest_build
python -m scripts.evaluate_generation --snapshot latest_build
python -m scripts.evaluate_end_to_end --snapshot latest_build
```

Both eval scripts log to the same local store; `--no-mlflow` prints results instead and
`--limit N` caps the rows for a smoke run. The generation eval builds each golden row's context
from its ground-truth recipe ids through the real prompt path, so it never exercises retrieval.
The end-to-end run is the one the promotion gate reads.

### Ship it

The image carries the app, one sealed snapshot, the dataset that snapshot pins, the LoRA
adapter, and the model weights pre-baked as an HF cache, so it boots with no network and no HF
token. Weight staging has to happen before a local build, and needs the weights in the local HF
cache:

```bash
bash scripts/stage_weights.sh
docker build --build-arg SNAPSHOT=<folder> -t recipe-rag:<sha> .
```

On this machine the same thing is one command, which pulls the image the registry's `serve`
target names and replaces the running container:

```bash
export RECIPE_RAG_API_KEYS='...'
python -m scripts.ship_baseline
```

`--image-repo recipe-rag` uses a local image instead of pulling, `--no-pull` skips the pull, and
`--port` moves the host port off 8000.
