# Study C - Reverse Invariance Commands

> **Direction:** Controllability -> Invariance.
> Measures which perturbation families change Study C controllability metrics most.

Use `--output-dir results_reverse` for all generation commands below.

Generation caches for each reverse study should be saved under `results_reverse/<MODEL>/...`, mirroring the per-model layout used in `results` and `results_invariance`.

---

## Ctrl Base Runs

### LM Studio

Use `--workers 1` for `gpt_oss` unless you have verified stability: LM Studio often returns HTTP 400 with “model has crashed” when the GPT-OSS backend receives concurrent `/v1/chat/completions` calls. The runtime caps GPT-OSS LM Studio workers to 1 by default; set `LMSTUDIO_GPT_OSS_MAX_WORKERS` to raise that cap if your stack is stable under load.

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 1
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id piaget_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse
```

---

## Ctrl Variant-Family Runs

Use `data/invariance/ctrl/variants/v2_1/study_c` to run every Study C
ctrl variant-family child in one go.

### LM Studio

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id gpt_oss --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse --workers 1
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwq --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse --workers 6
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse --workers 4
```

### Local HF

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyllm_gml_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id piaget_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyche_r1_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psych_qwen_local --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse --quantization 4bit
```

### vLLM

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyllm_gml_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id piaget_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psyche_r1_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id psych_qwen_vllm --ctrl-dir data/invariance/ctrl/variants/v2_1/study_c --output-dir results_reverse
```

---

## Direct Runner

```powershell
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_c --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --max-cases 3 --workers 1
```

## Checks

```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --check-only
python hf-local-scripts/run_ctrl_generate_only.py --study ctrl_study_c --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 1 --max-cases 3
```

---

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `summary_short` | summary | `data/invariance/ctrl/variants/v2_1/study_c/summary_short` |
| `summary_long` | summary | `data/invariance/ctrl/variants/v2_1/study_c/summary_long` |
| `patient_turn_rephrase` | paraphrase | `data/invariance/ctrl/variants/v2_1/study_c/patient_turn_rephrase` |
| `noncritical_reorder` | reorder | `data/invariance/ctrl/variants/v2_1/study_c/noncritical_reorder` |

## Comparison (All Variants)

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_reverse/<MODEL>/study_c_generations.jsonl \
  --variant-cache summary_short=results_reverse/<MODEL>/study_c/summary_short/study_c_generations.jsonl \
  --variant-cache summary_long=results_reverse/<MODEL>/study_c/summary_long/study_c_generations.jsonl \
  --variant-cache patient_turn_rephrase=results_reverse/<MODEL>/study_c/patient_turn_rephrase/study_c_generations.jsonl \
  --variant-cache noncritical_reorder=results_reverse/<MODEL>/study_c/noncritical_reorder/study_c_generations.jsonl \
  --variant-type summary_short=summary \
  --variant-type summary_long=summary \
  --variant-type patient_turn_rephrase=paraphrase \
  --variant-type noncritical_reorder=reorder \
  --out metric-results/<MODEL>/study_c_reverse_invariance.json
```

## Per-Variant (Individual)

```bash
# Summary Short
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_reverse/<MODEL>/study_c_generations.jsonl \
  --variant-cache summary_short=results_reverse/<MODEL>/study_c/summary_short/study_c_generations.jsonl \
  --variant-type summary_short=summary \
  --out metric-results/<MODEL>/study_c_reverse_invariance_summary_short.json

# Summary Long
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_reverse/<MODEL>/study_c_generations.jsonl \
  --variant-cache summary_long=results_reverse/<MODEL>/study_c/summary_long/study_c_generations.jsonl \
  --variant-type summary_long=summary \
  --out metric-results/<MODEL>/study_c_reverse_invariance_summary_long.json

# Patient Turn Rephrase
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_reverse/<MODEL>/study_c_generations.jsonl \
  --variant-cache patient_turn_rephrase=results_reverse/<MODEL>/study_c/patient_turn_rephrase/study_c_generations.jsonl \
  --variant-type patient_turn_rephrase=paraphrase \
  --out metric-results/<MODEL>/study_c_reverse_invariance_patient_turn_rephrase.json

# Noncritical Reorder
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_c \
  --data-root data/invariance/ctrl/base/v2_1 \
  --base-cache results_reverse/<MODEL>/study_c_generations.jsonl \
  --variant-cache noncritical_reorder=results_reverse/<MODEL>/study_c/noncritical_reorder/study_c_generations.jsonl \
  --variant-type noncritical_reorder=reorder \
  --out metric-results/<MODEL>/study_c_reverse_invariance_noncritical_reorder.json
```

## GPT-OSS-20B

LM Studio model. Recommended max 2 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_c \
  --model-id gpt_oss \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 2
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id gpt_oss --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 1
```

## Qwen3-8B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_c \
  --model-id qwen3_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwen3_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
```

## QwQ-32B

LM Studio model. Recommended max 6 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_c \
  --model-id qwq \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 6
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id qwq --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 6
```

## DeepSeek-R1 Distill Qwen 14B

LM Studio model. Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_c \
  --model-id deepseek_r1_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id deepseek_r1_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 4
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_c \
  --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 8
python scripts/dev/run_generation_auto.py --study study_c_invariance --model-id "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0" --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance_reverse --workers 1
```

## MedGemma 27B Text (`google.medgemma-27b-text-it`)

In LM Studio, load the model as `google.medgemma-27b-text-it`. The benchmark runner uses `--model-id medgemma_lmstudio` (aliases include that identifier). Recommended max 4 workers for reverse invariance runs.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_ctrl_generate_only.py \
  --study ctrl_study_c \
  --model-id medgemma_lmstudio \
  --ctrl-dir data/invariance/ctrl/base/v2_1 \
  --output-dir results_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study ctrl_study_c --model-id medgemma_lmstudio --ctrl-dir data/invariance/ctrl/base/v2_1 --output-dir results_reverse --workers 4
```



