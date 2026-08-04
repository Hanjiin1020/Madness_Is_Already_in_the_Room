#!/usr/bin/env bash



source "$(dirname "$0")/_common.sh"
launch_vllm "${GPU:-3}" "${PORT:-8011}" \
  mistralai/Pixtral-12B-2409 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-num-seqs 64
