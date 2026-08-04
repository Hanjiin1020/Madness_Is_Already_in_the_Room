#!/usr/bin/env bash



#     VLLM_BIN=/path/to/mllama-venv/bin/vllm GPU=0 PORT=8002 bash serve_llama.sh


#     python main.py ... --min-tokens 128 --min-tokens-models llama32vision
source "$(dirname "$0")/_common.sh"
launch_vllm "${GPU:-0}" "${PORT:-8002}" \
  meta-llama/Llama-3.2-11B-Vision-Instruct \
  --enforce-eager --max-num-seqs 16 \
  --served-model-name Llama-3.2-11B-Vision
