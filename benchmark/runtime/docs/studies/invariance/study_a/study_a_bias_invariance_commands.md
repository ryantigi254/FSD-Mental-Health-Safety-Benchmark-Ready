# Study A Bias Invariance Commands

Use `--data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json`
and `--output-dir results_invariance`.

`study_a_bias_invariance` defaults to `150` cases unless you override
`--max-cases`.

## Base Runs

This target always uses the frozen bias file directly rather than a
`data/invariance_variants/...` root.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwq --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id piaget_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance
```

## Variant-Family Runs

Not applicable for `study_a_bias_invariance`.

This target does not read from `data/invariance_variants/controllability/...`
family folders. It always uses the fixed frozen bias file above.

## Direct Runner

```powershell
python hf-local-scripts/run_study_a_bias_generate_only.py --study-name study_a_bias_invariance --model-id gpt_oss --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --max-cases 5 --max-tokens 32000 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --check-only
python hf-local-scripts/run_study_a_bias_generate_only.py --study-name study_a_bias_invariance --model-id gpt_oss --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance --workers 2 --max-cases 5
```
