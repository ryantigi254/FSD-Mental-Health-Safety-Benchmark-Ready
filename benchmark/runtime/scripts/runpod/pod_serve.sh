#!/usr/bin/env bash
# pod_serve.sh — Run on the RunPod H100 pod via SSH.
#
# Usage:
#   ./pod_serve.sh gpt-oss-120b    # start vLLM for GPT-OSS-120B
#   ./pod_serve.sh glm-4.7-flash   # start vLLM for GLM-4.7-Flash
#
# The script kills any running vLLM process first, then starts a new one.
# Logs are written to /workspace/logs/<model>.log
#
# Prerequisites on pod:
#   pip install vllm  (or use pre-installed vllm image)
#   HF_TOKEN env var set if models are gated
#   /runpod-volume mounted as network volume for weight cache

set -euo pipefail

MODEL=${1:-}
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <gpt-oss-120b|glm-4.7-flash>"
    exit 1
fi

CACHE_DIR="/runpod-volume/huggingface"
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

# Kill any running vLLM server
pkill -f "vllm serve" 2>/dev/null && echo "Stopped previous vLLM process." || echo "No vLLM process running."
sleep 3

case "$MODEL" in
  gpt-oss-120b)
    HF_MODEL="openai/gpt-oss-120b"
    SERVED_NAME="gpt-oss-120b"
    EXTRA_FLAGS=""
    # MXFP4 MoE + BF16 — fits on single H100, no quantization flag needed
    ;;
  glm-4.7-flash)
    HF_MODEL="zai-org/GLM-4.7-Flash"
    SERVED_NAME="glm-4.7-flash"
    EXTRA_FLAGS="--trust-remote-code"
    # 30B/3B active MoE, BF16 ~60 GB
    ;;
  *)
    echo "Unknown model: $MODEL. Supported: gpt-oss-120b, glm-4.7-flash"
    exit 1
    ;;
esac

LOG_FILE="$LOG_DIR/${MODEL}.log"
echo "Starting vLLM for $HF_MODEL → $LOG_FILE"

nohup vllm serve "$HF_MODEL" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.92 \
    --served-model-name "$SERVED_NAME" \
    --download-dir "$CACHE_DIR" \
    --port 8000 \
    $EXTRA_FLAGS \
    > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready (poll /health)
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s — serving $SERVED_NAME on port 8000."
        exit 0
    fi
    sleep 5
done

echo "ERROR: vLLM did not become ready after 600s. Check $LOG_FILE"
exit 1
