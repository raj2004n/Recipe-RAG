# syntax=docker/dockerfile:1
# Serve-only image for the RAG service. A reproducible two-stage build: the
# builder assembles a venv holding just the serve requirements, and the runtime
# copies it in with the app, the sealed snapshot, the dataset it pins, and the
# inference model weights, so the image boots and answers offline. The torch
# cu124 wheels are not on PyPI, so installs use the PyTorch index first.
#
# Build with:  docker build -t recipe-rag:$(git rev-parse --short=12 HEAD) .
# Run with:    docker run --gpus all -p 8000:8000 \
#                -e RECIPE_RAG_API_KEYS='{"dev-admin-key":"admin","dev-public-key":"public"}' \
#                recipe-rag:<sha>

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements/base.txt requirements/serve.txt requirements/

# faiss-gpu 1.14.3's metadata asks for nvidia-cuda-runtime-cu12 >=12.6, which
# clashes with torch 2.6.0+cu124 shipping the 12.4 wheels. The pair runs fine
# together (faiss uses torch's 12.4 libraries), so faiss is dropped out of the
# main resolve and added back with --no-deps, mirroring the working host venv.
# The filters are anchored: the faiss==... line in base.txt and the "-r base.txt"
# include line in serve.txt.
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_NO_CACHE_DIR=1

RUN python3 -m venv /opt/venv
RUN grep -v '^faiss-gpu' requirements/base.txt > /build/base.no-faiss.txt \
 && grep -v '^-r base.txt' requirements/serve.txt > /build/serve.extras.txt
RUN . /opt/venv/bin/activate && pip install -r /build/base.no-faiss.txt -r /build/serve.extras.txt
RUN . /opt/venv/bin/activate && pip install --no-deps faiss-gpu==1.14.3

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV PATH=/opt/venv/bin:$PATH
ENV HF_HOME=/models
# every model the service loads is already in the cache above, so the hub reads
# it and never dials out. Without this the loaders revalidate each file against
# huggingface.co on every start, which is a boot-time network dependency the
# image is meant not to have: the request still fails closed on a missing file,
# it just fails from the cache rather than the network.
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# the venv's bin/python symlinks to the system python, so the runtime needs it.
# gcc is a run-time need rather than a build one: the Qwen3.5 modeling file
# imports flash-linear-attention, whose triton imports compile a small C shim on
# first use. Without a compiler that import fails, and transformers reports the
# failure as a missing model class, which buries the real cause. The shim includes
# stdlib.h and dlfcn.h, which come from libc6-dev, and Python.h, which comes from
# python3-dev. Triton carries cuda.h itself and the toolkit supplies libcuda, so
# the CUDA development packages are not needed.
RUN apt-get update && apt-get install -y --no-install-recommends python3 gcc libc6-dev python3-dev && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# app code and config
COPY src ./src
COPY configs ./configs

# the sealed snapshot, its registry, and the dataset it pins. The lifespan
# fails closed if the dataset checksum drifts, so these three must stay in step.
# The workflow passes the folder the registry's serve target names. The default
# is for a local build and has to be a real name: an empty value would make the
# COPY below take the whole snapshots/ tree.
ARG SNAPSHOT=hybrid_20260927_1431_0
COPY snapshots/$SNAPSHOT ./snapshots/$SNAPSHOT
COPY snapshots/registry.json ./snapshots/registry.json
COPY data/processed/cleaned_recipes.json ./data/processed/cleaned_recipes.json

# pre-baked HF hub cache for Qwen3.5-0.8B, bge-reranker-v2-m3, all-MiniLM-L6-v2
COPY .docker/weights /models/hub

# the LoRA adapter the served snapshot's generator config names. It goes to the
# same relative path the config carries, so adapter_path resolves inside the
# image without the sealed config differing from the one the host runs.
COPY artifacts/student/qwen3.5-0.8b_lora_r16_20260919_1757/epoch_02 \
     ./artifacts/student/qwen3.5-0.8b_lora_r16_20260919_1757/epoch_02

EXPOSE 8000
CMD ["uvicorn", "src.rag.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
