#!/usr/bin/env bash

#

#   GPUS=0,1 PORT=8006 bash serve_judge.sh

#






set -u

BIN="${VLLM_BIN:-$(command -v vllm || true)}"
if [ -z "$BIN" ]; then
  echo "[중단] vllm 실행 파일을 찾을 수 없습니다." >&2
  echo "       가상환경을 activate 하거나 VLLM_BIN=/path/to/vllm 을 지정하세요." >&2
  exit 1
fi
export PATH="$(dirname "$BIN"):$PATH"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

echo "[$(date +%T)] serving Qwen/Qwen3-32B  GPUS=${GPUS:-0,1} port=${PORT:-8006}"
CUDA_VISIBLE_DEVICES="${GPUS:-0,1}" exec "$BIN" serve Qwen/Qwen3-32B \
  --port "${PORT:-8006}" --tensor-parallel-size 2 \
  --max-model-len "${MAXLEN:-16384}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --disable-custom-all-reduce
