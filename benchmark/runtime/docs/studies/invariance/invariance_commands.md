# Invariance Study Generation Commands

## Prepared Roots

### Base Runs

- `data/invariance/base`
- `data/invariance/variant_family/base`

### Variant-Family Runs

- `data/invariance/variant_family/study_a`
- `data/invariance/variant_family/study_b`
- `data/invariance/variant_family/study_b_multi_turn`
- `data/invariance/variant_family/study_c`

Use `--output-dir results_invariance` throughout.

## LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/base --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/base --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/base --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/base --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/invariance/base/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 2
```

## Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/base --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/base --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/base --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/base --output-dir results_invariance --quantization 4bit
```

## vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/base --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/base --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/base --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/base --output-dir results_invariance
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss --data-dir data/invariance/base --output-dir results_invariance --max-samples 5 --workers 2
```

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_b \
  --data-root data/invariance/base \
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

## Controllability-Backed Invariance

Ctrl invariance commands (using `data/invariance/ctrl_base` and
`data/invariance/ctrl_variant_family`) live in separate files:

- `docs/studies/invariance/ctrl_invariance_commands.md` — top-level ctrl overview
- `docs/studies/invariance/study_a/study_a_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_a/study_a_bias_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_c/study_c_ctrl_invariance_commands.md`
