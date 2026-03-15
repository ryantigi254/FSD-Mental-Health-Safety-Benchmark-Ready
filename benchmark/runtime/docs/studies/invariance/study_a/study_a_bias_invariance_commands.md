# Study A Bias Invariance Commands

## Scope
Study A bias invariance generation writes to
`results/<model-folder>/study_a_bias_invariance_generations.jsonl`.

The default sampled budget is `150` cases. This path does not use the full
2000-case bias run unless you explicitly raise `--max-cases`.

## Canonical Paths

These commands use the frozen `v5` bias file directly so this target stays
separate from the ordinary Study A bias run:

- `--data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json`
- `--output-dir results_invariance_v5`

## Generation Commands (Automatic Cross-Platform Runner)

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwq --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --workers 4
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --workers 2
```

## Local HF Models (`mh-llm-local-env`)

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id piaget_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --quantization 4bit
```

## vLLM (Local HF Models Only)

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5
```

## Direct Runner

```powershell
python hf-local-scripts/run_study_a_bias_generate_only.py --study-name study_a_bias_invariance --model-id gpt_oss_lmstudio --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --max-cases 5 --max-tokens 32000 --workers 2
```

## Workers

`study_a_bias_invariance` generation supports `--workers`. If not passed,
default is auto:

- `4` for LM Studio models
- `1` for vLLM and local HF models

`--max-cases` defaults to `150` for `study_a_bias_invariance`. Pass a different
value only if you want a larger or smaller sampled slice.

## Useful Checks

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss_lmstudio --env mh-llm-benchmark-env --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --check-only
python hf-local-scripts/run_study_a_bias_generate_only.py --study-name study_a_bias_invariance --model-id gpt_oss_lmstudio --data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json --output-dir results_invariance_v5 --workers 2 --max-cases 5
```
