# Study A Invariance Commands

## Scope

Study A invariance generation writes to `results/<model-folder>/study_a_invariance_generations.jsonl`.

## Canonical Paths

Use one of these roots directly in the commands:

- Main `v5` variants:
  - `--data-dir data/invariance_variants/v5`
  - `--output-dir results_invariance_v5`
  - default child selected by the runner: `study_a/lexical`
- Controllability-backed invariance:
  - `--data-dir data/invariance_variants/controllability/base`
  - `--output-dir results_invariance_controllability`

## Generation (automatic runner)

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --workers 2
```

Controllability-backed invariance:

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance_variants/controllability/base --output-dir results_invariance_controllability --workers 2
```

Other models (same pattern; adjust `--env` and `--workers` as needed):

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --workers 6
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id qwq --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --workers 6
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --workers 4
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --quantization 4bit
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5
```

## Direct runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss_lmstudio --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --max-samples 5 --max-tokens 32000 --workers 2
```

## Workers

`study_a_invariance` supports `--workers`. If omitted, default is auto (e.g. 4 for LM Studio, 1 for vLLM/local).

## Useful checks

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --check-only
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss_lmstudio --data-dir data/invariance_variants/v5 --output-dir results_invariance_v5 --workers 2 --max-samples 5
```
