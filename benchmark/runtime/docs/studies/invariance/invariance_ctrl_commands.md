# Controllability Invariance — Generation Commands

> Commands for running invariance generation on controllability-backed samples.
> These use the `ctrl_base` and `ctrl_variant_family` data roots, which are
> built from the controllability splits rather than the v6.1 parent.

---

## Prepared Roots

### Ctrl Base Runs

- `data/invariance/ctrl/base/v2_1`

### Ctrl Variant-Family Runs

- `data/invariance/ctrl/variants/v2_1/study_a`
- `data/invariance/ctrl/variants/v2_1/study_b`
- `data/invariance/ctrl/variants/v2_1/study_b_multi_turn`
- `data/invariance/ctrl/variants/v2_1/study_c`

Use `--output-dir results_ctrl_invariance` throughout.

## LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 2
```

## Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --quantization 4bit
```

## vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_b_multi_turn_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance
```

## Direct Runner

```powershell
python hf-local-scripts/run_invariance_generate_only.py --study study_a_invariance --model-id gpt_oss --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_ctrl_invariance --max-samples 5 --workers 2
```

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_b \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/qwq/study_b_generations.jsonl \
  --variant-cache results_ctrl_invariance/qwq/study_b_invariance_generations.jsonl \
  --variant-name ctrl_invariance \
  --out metric-results/qwq/study_b_ctrl_invariance.json
```

Per-study notes:

- `docs/studies/invariance/study_a/study_a_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_a/study_a_bias_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_ctrl_invariance_commands.md`
- `docs/studies/invariance/study_c/study_c_ctrl_invariance_commands.md`

## Forward Evaluation (Invariance → Controllability)

Run `run_invariance_comparison.py` to measure whether ctrl metrics are stable
across surface perturbations:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_a \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_ctrl_invariance/<MODEL>/study_a_generations.jsonl \
  --variant-cache results_ctrl_invariance/<MODEL>/study_a/<VARIANT>/study_a_generations.jsonl \
  --variant-name <VARIANT> \
  --out metric-results/<MODEL>/study_a_ctrl_forward_invariance.json
```

Repeat per study and per variant family.

## Reverse Evaluation (Controllability → Invariance)

The reverse direction measures which perturbation families change ctrl metrics
most, using `run_controllability_comparison.py` with variant-family caches.

Full commands: [`invariance_reverse_commands.md`](invariance_reverse_commands.md)
