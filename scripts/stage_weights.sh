#!/usr/bin/env bash
# Stage the three inference-model weights from the local HF cache into the
# build context so the image bundles them and boots offline.
#
# This mirrors the model ids in configs/base_config.yaml. The copied dirs land
# under .docker/weights/ in the same models--{org}--{name} layout transformers
# expects inside HF_HOME, and the Dockerfile places them at /models/hub so the
# image needs no network and no HF token at boot.
set -euo pipefail

mkdir -p .docker/weights

for model in \
  google/gemma-3-1b-it \
  BAAI/bge-reranker-v2-m3 \
  sentence-transformers/all-MiniLM-L6-v2; do
  # the hub cache maps each slash in the model id to a double dash
  src="${HOME}/.cache/huggingface/hub/models--${model//\//--}"
  if [ -d "$src" ]; then
    echo "staging $model"
    cp -r "$src" .docker/weights/
  else
    echo "missing HF cache for $model at $src" >&2
    exit 1
  fi
done

echo "staged weights into .docker/weights/"
