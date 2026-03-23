# Study B Controllability Invariance Commands

Use `--output-dir results_ctrl_invariance` for all commands below.

## Ctrl Base Runs

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
```

## Ctrl Variant-Family Runs

Use `data/invariance/ctrl/variants/v2_1/study_b` to run every Study B
ctrl variant-family child in one go.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/variants/v2_1/study_b --output-dir results_ctrl_invariance
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_b_invariance --model-id gpt_oss --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --max-samples 5 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --check-only
python hf-local-scripts/run_invariance_generate_only.py --study study_b_invariance --model-id gpt_oss --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2 --max-samples 5
```
