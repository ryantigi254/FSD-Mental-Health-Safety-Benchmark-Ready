# Invariance Study Generation Commands

## Prepared Roots

### Base Runs

- `data/invariance_variants/v5`
- `data/invariance_variants/controllability/base`

### Variant-Family Runs

- `data/invariance_variants/controllability/study_a`
- `data/invariance_variants/controllability/study_b`
- `data/invariance_variants/controllability/study_b_multi_turn`
- `data/invariance_variants/controllability/study_c`

Use `--output-dir results_invariance` throughout.

## LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance_variants/v5 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 2
```

## Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance_variants/v5 --output-dir results_invariance --quantization 4bit
```

## vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance_variants/v5 --output-dir results_invariance
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss --data-dir data/invariance_variants/v5 --output-dir results_invariance --max-samples 5 --workers 2
```

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_b \
  --data-root data/invariance_variants/v5 \
  --base-cache results_invariance/qwq/study_b_generations.jsonl \
  --variant-cache results_invariance/qwq/study_b_invariance_generations.jsonl \
  --variant-name controllability_sample \
  --out metric-results/qwq/study_b_invariance_from_controllability.json
```

Per-study notes:

- `docs/studies/invariance/study_a/study_a_invariance_commands.md`
- `docs/studies/invariance/study_a/study_a_bias_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_invariance_commands.md`
- `docs/studies/invariance/study_c/study_c_invariance_commands.md`
