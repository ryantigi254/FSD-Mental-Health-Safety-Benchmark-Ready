# Invariance Study Generation Commands

## Prepared Roots

### Base Runs

- `data/invariance/v5/base/v2_1`
- `data/invariance/v5/variants/v2_1/base`

### Variant-Family Runs

- `data/invariance/v5/variants/v2_1/study_a`
- `data/invariance/v5/variants/v2_1/study_b`
- `data/invariance/v5/variants/v2_1/study_b_multi_turn`
- `data/invariance/v5/variants/v2_1/study_c`

Use `--output-dir results_invariance` throughout.

## LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/invariance/v5/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 2
```

## Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance --quantization 4bit
```

## vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance --max-samples 5 --workers 2
```

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_b \
  --data-root data/invariance/v5/base/v2_1 \
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

Ctrl invariance commands (using `data/invariance/ctrl/base/v2_1` and
`data/invariance/ctrl/variants/v2_1`) live in separate files:

- `docs/studies/invariance/invariance_ctrl_commands.md` — top-level ctrl overview (forward + reverse)
- `docs/studies/invariance/invariance_reverse_commands.md` — reverse evaluation (Controllability → Invariance)
- `docs/studies/invariance/study_a/study_a_invariance_ctrl_commands.md`
- `docs/studies/invariance/study_a/study_a_bias_invariance_ctrl_commands.md`
- `docs/studies/invariance/study_b/study_b_invariance_ctrl_commands.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_invariance_ctrl_commands.md`
- `docs/studies/invariance/study_c/study_c_invariance_ctrl_commands.md`
