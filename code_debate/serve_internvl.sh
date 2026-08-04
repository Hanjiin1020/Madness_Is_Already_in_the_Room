#!/usr/bin/env bash


source "$(dirname "$0")/_common.sh"
launch_vllm "${GPU:-0}" "${PORT:-8001}" \
  OpenGVLab/InternVL3_5-14B-Instruct \
  --trust-remote-code \
  --served-model-name InternVL3_5-14B
