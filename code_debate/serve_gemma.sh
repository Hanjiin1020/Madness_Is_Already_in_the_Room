#!/usr/bin/env bash


source "$(dirname "$0")/_common.sh"
launch_vllm "${GPU:-1}" "${PORT:-8003}" \
  google/gemma-4-12B-it \
  --served-model-name gemma-4-12B-it
