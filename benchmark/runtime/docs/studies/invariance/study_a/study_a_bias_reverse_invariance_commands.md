# Study A Bias - Reverse Invariance Commands

> **Direction:** Controllability -> Invariance.
> Measures which perturbation families change Study A Bias controllability metrics most.
>
> Study A Bias shares the same variant families as Study A (lexical, surface,
> syntax, instruction) but evaluates against the bias vignettes subset.

Use `--output-dir results_reverse` for all generation commands below.

Generation caches for each reverse study should be saved under `results_reverse/<MODEL>/...`, mirroring the per-model layout used in `results` and `results_invariance`.

---

## Ctrl Base Runs

Uses the sampled invariance subset at
`data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json`
(140 cases sampled from the controllability bias file).

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 2
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id piaget_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
```

## Variant-Family Runs

Not applicable for `ctrl_study_a_bias`.

The bias reverse invariance study does not use variant-family perturbations.

---

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a_bias --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --max-cases 5 --workers 2
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --check-only
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_a_bias --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 2 --max-cases 5
```

---

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `lexical` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/lexical` |
| `surface` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/surface` |
| `syntax` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/syntax` |
| `instruction` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_a/instruction` |

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a_bias \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_reverse/<MODEL>/study_a_bias_generations.jsonl \
  --variant-cache lexical=results_reverse/<MODEL>/study_a/lexical/study_a_bias_generations.jsonl \
  --variant-cache surface=results_reverse/<MODEL>/study_a/surface/study_a_bias_generations.jsonl \
  --variant-cache syntax=results_reverse/<MODEL>/study_a/syntax/study_a_bias_generations.jsonl \
  --variant-cache instruction=results_reverse/<MODEL>/study_a/instruction/study_a_bias_generations.jsonl \
  --variant-type lexical=paraphrase \
  --variant-type surface=paraphrase \
  --variant-type syntax=paraphrase \
  --variant-type instruction=paraphrase \
  --out metric-results/<MODEL>/study_a_bias_reverse_invariance.json
```

## GPT-OSS-20B

LM Studio model. Recommended max 2 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a_bias \
  --model-id gpt_oss \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 2
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 2
```

## Qwen3-8B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a_bias \
  --model-id qwen3_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
```

## QwQ-32B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a_bias \
  --model-id qwq \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
```

## DeepSeek-R1 Distill Qwen 14B

LM Studio model. Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a_bias \
  --model-id deepseek_r1_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 4
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a_bias \
  --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 8
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance_reverse --workers 8
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_a_bias \
  --model-id medgemma_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_a_bias --model-id medgemma_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 4
```



