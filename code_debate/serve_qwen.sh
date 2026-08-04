#!/usr/bin/env bash




#   (src/mllm_backend.py: chat_template_kwargs={"enable_thinking": false}).
source "$(dirname "$0")/_common.sh"
launch_vllm "${GPU:-2}" "${PORT:-8004}" \
  Qwen/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B
