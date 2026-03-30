# Study A Bias Controllability Invariance Commands

Use `--data-path` pointing to the ctrl invariance bias file
and `--output-dir results_ctrl_invariance`.

`study_a_bias_invariance` defaults to `140` cases (controllability)
unless you override `--max-cases`.

## Ctrl Base Runs

Uses the sampled invariance subset at
`data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json`
(140 cases sampled from the controllability bias file).

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 2
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwen3_lmstudio --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwq --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 6
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id deepseek_r1_lmstudio --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyllm_gml_local --env mh-llm-local-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id piaget_local --env mh-llm-local-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyche_r1_local --env mh-llm-local-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psych_qwen_local --env mh-llm-local-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyllm_gml_vllm --env mh-llm-vllm-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id piaget_vllm --env mh-llm-vllm-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psyche_r1_vllm --env mh-llm-vllm-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id psych_qwen_vllm --env mh-llm-vllm-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance
```

## Variant-Family Runs

Not applicable for `study_a_bias_invariance`.

The bias invariance study does not use variant-family perturbations.

## Direct Runner

```powershell
python hf-local-scripts/run_study_a_bias_generate_only.py --study-name study_a_bias_invariance --model-id gpt_oss --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --max-cases 5 --max-tokens 32000 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id gpt_oss --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --check-only
python hf-local-scripts/run_study_a_bias_generate_only.py --study-name study_a_bias_invariance --model-id gpt_oss --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 2 --max-cases 5
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_invariance_generate_only.py \
  --study study_a_bias_invariance \
  --model-id qwen3.5-distilled \
  --data-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_invariance_ctrl \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwen3.5-distilled --env mh-llm-benchmark-env --data-dir data/invariance/ctrl/base/v2_1 --output-dir results_invariance_ctrl --workers 8
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_invariance_generate_only.py \
  --study study_a_bias_invariance \
  --model-id medgemma_lmstudio \
  --data-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_ctrl_invariance \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id medgemma_lmstudio --env mh-llm-benchmark-env --data-path data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json --output-dir results_ctrl_invariance --workers 4
```
