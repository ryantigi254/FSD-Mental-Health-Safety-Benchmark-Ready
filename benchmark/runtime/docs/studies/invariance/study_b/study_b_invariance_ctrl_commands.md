# Study B Controllability Invariance Commands

Use `--output-dir results_invariance_ctrl` for all commands below.

## Ctrl Base Runs

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl
```

## Ctrl Variant-Family Runs

Use `data/invariance/ctrl/variants/v2_1/study_b` to run every Study B
ctrl variant-family child in one go.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_invariance_ctrl
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_b_invariance --model-id gpt_oss --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --max-samples 5 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --check-only
python hf-local-scripts/run_invariance_generate_only.py --study study_b_invariance --model-id gpt_oss --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 2 --max-samples 5
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_invariance_generate_only.py \
  --study study_b_invariance \
  --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" \
  --data-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_invariance_ctrl \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 8
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_invariance_generate_only.py \
  --study study_b_invariance \
  --model-id medgemma_lmstudio \
  --data-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_invariance_ctrl \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id medgemma_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 4
```
