#!/usr/bin/env bash
# run_pod_pipeline.sh — Full two-model RunPod benchmark pipeline.
#
# Runs all 25 studies for GPT-OSS-120B, then restarts vLLM for GLM-4.7-Flash
# and runs all 25 studies again. Total: 50 runs across both models.
#
# Usage (from benchmark/runtime/):
#   POD_HOST=<pod-id>.ssh.runpod.net POD_SSH_KEY=~/.ssh/runpod bash scripts/runpod/run_pod_pipeline.sh
#
# The script:
#   1. SSH into the pod to start vLLM for GPT-OSS-120B
#   2. Runs all 25 local study scripts against the pod endpoint
#   3. SSH into the pod to restart vLLM for GLM-4.7-Flash
#   4. Runs all 25 local study scripts against the pod endpoint
#
# Prerequisites:
#   - .env loaded with endpoint URLs and API key
#   - Pod SSH accessible via POD_HOST / POD_SSH_KEY
#   - Scripts are executable: chmod +x scripts/runpod/*.sh

set -euo pipefail

POD_HOST="${POD_HOST:-}"
POD_SSH_KEY="${POD_SSH_KEY:-$HOME/.ssh/runpod}"
RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "$POD_HOST" ]]; then
    echo "Set POD_HOST to your pod's SSH address, e.g.:"
    echo "  POD_HOST=abc123-22.proxy.runpod.net POD_SSH_KEY=~/.ssh/id_rsa bash scripts/runpod/run_pod_pipeline.sh"
    exit 1
fi

SSH="ssh -i $POD_SSH_KEY -o StrictHostKeyChecking=no root@$POD_HOST"

pod_serve() {
    local model=$1
    echo ""
    echo "▶  Pod: starting vLLM for $model ..."
    $SSH "bash /workspace/pod_serve.sh $model"
}

run_studies() {
    local model_id=$1
    echo ""
    echo "▶  Local: running all 25 studies for $model_id ..."
    cd "$RUNTIME_ROOT"
    MODEL_ID="$model_id" bash scripts/runpod/run_all_studies.sh
}

# Copy pod_serve.sh to pod once
echo "Copying pod_serve.sh to pod..."
scp -i "$POD_SSH_KEY" -o StrictHostKeyChecking=no \
    "$RUNTIME_ROOT/scripts/runpod/pod_serve.sh" \
    "root@$POD_HOST:/workspace/pod_serve.sh"
$SSH "chmod +x /workspace/pod_serve.sh"

# ── MODEL 1: GPT-OSS-120B (runs 1–25) ────────────────────────────────────────
START1=$(date +%s)
pod_serve "gpt-oss-120b"
run_studies "gpt-oss-120b-runpod"
END1=$(date +%s)
echo "GPT-OSS-120B completed in $(( (END1 - START1) / 60 )) min."

# ── MODEL 2: GLM-4.7-Flash (runs 26–50) ──────────────────────────────────────
START2=$(date +%s)
pod_serve "glm-4.7-flash"
run_studies "glm-4.7-flash-runpod"
END2=$(date +%s)
echo "GLM-4.7-Flash completed in $(( (END2 - START2) / 60 )) min."

# ── Done ─────────────────────────────────────────────────────────────────────
TOTAL=$(( (END2 - START1) / 60 ))
echo ""
echo "████  FULL PIPELINE COMPLETE — 50 runs total in ${TOTAL} min  ████"
echo "Results:"
echo "  GPT-OSS-120B  → results*/gpt-oss-120b/"
echo "  GLM-4.7-Flash → results*/glm-4.7-flash/"
