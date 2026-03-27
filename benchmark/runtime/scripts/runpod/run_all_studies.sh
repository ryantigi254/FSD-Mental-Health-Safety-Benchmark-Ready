#!/usr/bin/env bash
# run_all_studies.sh — Run locally. Sequences all 25 study runs for a given model.
#
# Usage (from benchmark/runtime/):
#   MODEL_ID=gpt-oss-120b-runpod bash scripts/runpod/run_all_studies.sh
#   MODEL_ID=glm-4.7-flash-runpod bash scripts/runpod/run_all_studies.sh
#
# Requires .env with RUNPOD_GPT_OSS_120B_ENDPOINT / RUNPOD_GLM47_FLASH_ENDPOINT set.
# Workers is set to 4 (RunPod vLLM handles concurrency server-side).
#
# Run count per model: 25
#   Base studies       5  (A, A-bias, B, B-multi-turn, C)
#   Controllability    5  (ctrl A, ctrl A-bias, ctrl B, ctrl B-multi-turn, ctrl C)
#   Metric invariance 15  (invariance ×5 + invariance-ctrl ×5 + reverse ×5)

set -euo pipefail

MODEL_ID=${MODEL_ID:-}
if [[ -z "$MODEL_ID" ]]; then
    echo "Set MODEL_ID before running. e.g.:"
    echo "  MODEL_ID=gpt-oss-120b-runpod bash scripts/runpod/run_all_studies.sh"
    exit 1
fi

# Resolve runtime root regardless of cwd
RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RUNTIME_ROOT"

ENV_ARG=""   # no --env flag — use current Python; activate your env before running

WORKERS=4

CTRL_DIR="data/controllability/controllability_splits_v2_1"
CTRL_RESULTS_DIR="results_ctrl_v2_1"

INV_DATA_DIR="data/invariance/v5/base/v2_1"
INV_CTRL_DATA_DIR="data/invariance/ctrl/base/v2_1"
INV_OUTPUT="results_invariance"
INV_CTRL_OUTPUT="results_ctrl_invariance"

BIAS_DATA_PATH="data/invariance/v5/base/v2_1/adversarial_bias/biased_vignettes.json"
BIAS_CTRL_DATA_PATH="data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json"

run() {
    echo ""
    echo "=== [$(date '+%H:%M:%S')] $* ==="
    PYTHONPATH=src python "$@"
}

# ── BLOCK 1: Base Studies (5 runs) ────────────────────────────────────────────
echo ""
echo "████  BLOCK 1 / 5 — Base Studies  ████"

run scripts/dev/run_generation_auto.py --study study_a       --model-id "$MODEL_ID" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_a_bias  --model-id "$MODEL_ID" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_b       --model-id "$MODEL_ID" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_b_multi_turn --model-id "$MODEL_ID" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_c       --model-id "$MODEL_ID" --workers $WORKERS

# ── BLOCK 2: Controllability (5 runs) ─────────────────────────────────────────
echo ""
echo "████  BLOCK 2 / 5 — Controllability  ████"

run hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a            --model-id "$MODEL_ID" --ctrl-dir "$CTRL_DIR" --output-dir "$CTRL_RESULTS_DIR"
run hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a_bias       --model-id "$MODEL_ID" --ctrl-dir "$CTRL_DIR" --output-dir "$CTRL_RESULTS_DIR"
run hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_b            --model-id "$MODEL_ID" --ctrl-dir "$CTRL_DIR" --output-dir "$CTRL_RESULTS_DIR"
run hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_b_multi_turn --model-id "$MODEL_ID" --ctrl-dir "$CTRL_DIR" --output-dir "$CTRL_RESULTS_DIR"
run hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_c            --model-id "$MODEL_ID" --ctrl-dir "$CTRL_DIR" --output-dir "$CTRL_RESULTS_DIR"

# ── BLOCK 3: Metric Invariance — Invariance (5 runs) ─────────────────────────
echo ""
echo "████  BLOCK 3 / 5 — Invariance (v5 base)  ████"

run scripts/dev/run_generation_auto.py --study study_a_invariance              --model-id "$MODEL_ID" --data-dir "$INV_DATA_DIR" --output-dir "$INV_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_a_bias_invariance         --model-id "$MODEL_ID" --data-path "$BIAS_DATA_PATH" --output-dir "$INV_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_b_invariance              --model-id "$MODEL_ID" --data-dir "$INV_DATA_DIR" --output-dir "$INV_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance   --model-id "$MODEL_ID" --data-dir "$INV_DATA_DIR" --output-dir "$INV_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_c_invariance              --model-id "$MODEL_ID" --data-dir "$INV_DATA_DIR" --output-dir "$INV_OUTPUT" --workers $WORKERS

# ── BLOCK 4: Metric Invariance — Invariance Ctrl (5 runs) ────────────────────
echo ""
echo "████  BLOCK 4 / 5 — Invariance Ctrl  ████"

run scripts/dev/run_generation_auto.py --study study_a_invariance              --model-id "$MODEL_ID" --data-dir "$INV_CTRL_DATA_DIR" --output-dir "$INV_CTRL_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_a_bias_invariance         --model-id "$MODEL_ID" --data-path "$BIAS_CTRL_DATA_PATH" --output-dir "$INV_CTRL_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_b_invariance              --model-id "$MODEL_ID" --data-dir "$INV_CTRL_DATA_DIR" --output-dir "$INV_CTRL_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance   --model-id "$MODEL_ID" --data-dir "$INV_CTRL_DATA_DIR" --output-dir "$INV_CTRL_OUTPUT" --workers $WORKERS
run scripts/dev/run_generation_auto.py --study study_c_invariance              --model-id "$MODEL_ID" --data-dir "$INV_CTRL_DATA_DIR" --output-dir "$INV_CTRL_OUTPUT" --workers $WORKERS

# ── BLOCK 5: Metric Invariance — Reverse / Variant Families (5 runs) ─────────
echo ""
echo "████  BLOCK 5 / 5 — Reverse Invariance (ctrl variant families)  ████"
# Generates on ctrl/variants/* so the reverse comparison scripts have the caches they need.

for STUDY in study_a study_b study_b_multi_turn study_c; do
    for VARIANT in lexical surface syntax instruction; do
        VARIANT_DIR="data/invariance/ctrl/variants/v2_1/${STUDY}/${VARIANT}"
        if [[ -d "$VARIANT_DIR" ]]; then
            run scripts/dev/run_generation_auto.py \
                --study "${STUDY}_invariance" \
                --model-id "$MODEL_ID" \
                --data-dir "$VARIANT_DIR" \
                --output-dir "results_ctrl_invariance_variants" \
                --workers $WORKERS
        fi
    done
done

# study_a_bias reverse variants
for VARIANT in lexical surface syntax instruction; do
    BIAS_VARIANT_PATH="data/invariance/ctrl/variants/v2_1/study_a_bias/${VARIANT}/biased_vignettes.json"
    if [[ -f "$BIAS_VARIANT_PATH" ]]; then
        run scripts/dev/run_generation_auto.py \
            --study study_a_bias_invariance \
            --model-id "$MODEL_ID" \
            --data-path "$BIAS_VARIANT_PATH" \
            --output-dir "results_ctrl_invariance_variants" \
            --workers $WORKERS
    fi
done

echo ""
echo "████  ALL 25 STUDY RUNS COMPLETE for $MODEL_ID  ████"
echo "Finished at $(date)"
