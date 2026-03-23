# Study A Invariance Commands

Use `--output-dir results_invariance` for all commands below.

## Base Runs

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance
```

## Variant-Family Runs

Use `data/invariance/v5_variants_v2_1/study_a` to run every Study A
variant-family child in one go.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5_variants_v2_1/study_a --output-dir results_invariance
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --max-samples 5 --max-tokens 32000 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --check-only
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss --data-dir data/invariance/v5_base_v2_1 --output-dir results_invariance --workers 2 --max-samples 5
```
