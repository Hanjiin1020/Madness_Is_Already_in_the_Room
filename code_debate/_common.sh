#!/usr/bin/env bash

#

#









set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-$HERE/logs}"
mkdir -p "$LOGDIR"

MAXLEN="${MAXLEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# launch_vllm <gpu> <port> <model> [extra args...]
launch_vllm() {
  local gpu="$1" port="$2" model="$3"; shift 3

  local bin="${VLLM_BIN:-$(command -v vllm || true)}"
  if [ -z "$bin" ]; then
    echo "[중단] vllm 실행 파일을 찾을 수 없습니다." >&2
    echo "       가상환경을 activate 하거나 VLLM_BIN=/path/to/vllm 을 지정하세요." >&2
    exit 1
  fi

  export PATH="$(dirname "$bin"):$PATH"

  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

  echo "[$(date +%T)] serving $model  GPU=$gpu port=$port maxlen=$MAXLEN  extra=$*"
  CUDA_VISIBLE_DEVICES="$gpu" exec "$bin" serve "$model" \
    --port "$port" --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAXLEN" "$@"
}
