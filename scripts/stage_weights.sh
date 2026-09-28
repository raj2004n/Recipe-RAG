#!/usr/bin/env bash
# Stage the three inference-model weights from the local HF cache into the
# build context so the image bundles them and boots offline.
#
# This mirrors the model ids the sealed snapshot's configs name, so a snapshot
# that names a different generator has to be reflected in the list below. The
# copied dirs land under .docker/weights/ in the same models--{org}--{name}
# layout transformers expects inside HF_HOME, and the Dockerfile places them at
# /models/hub so the image needs no network and no HF token at boot. Dirs
# already staged are left in place, so re-running after a build does not
# re-copy the weights, and a staged dir the list no longer names is removed so
# the image carries the models this list names and nothing else.
set -euo pipefail

mkdir -p .docker/weights

models=(
  Qwen/Qwen3.5-0.8B
  BAAI/bge-reranker-v2-m3
  sentence-transformers/all-MiniLM-L6-v2
)

for model in "${models[@]}"; do
  # the hub cache maps each slash in the model id to a double dash
  name="models--${model//\//--}"
  src="${HOME}/.cache/huggingface/hub/${name}"
  dest=".docker/weights/${name}"
  if [ -d "$dest" ]; then
    echo "already staged $model"
    continue
  fi
  if [ -d "$src" ]; then
    echo "staging $model"
    cp -r "$src" "$dest"
  else
    echo "missing HF cache for $model at $src" >&2
    exit 1
  fi
done

for staged in .docker/weights/models--*; do
  [ -d "$staged" ] || continue
  keep=""
  for model in "${models[@]}"; do
    if [ "$staged" = ".docker/weights/models--${model//\//--}" ]; then
      keep=1
      break
    fi
  done
  if [ -z "$keep" ]; then
    echo "removing staged weights the list no longer names: ${staged#.docker/weights/}"
    rm -rf "$staged"
  fi
done

echo "staged weights into .docker/weights/"
