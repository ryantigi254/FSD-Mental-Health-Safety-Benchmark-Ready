# Study A Bias — Reverse Invariance Commands

> **Direction:** Controllability → Invariance.
> Measures which perturbation families change Study A Bias controllability metrics most.
>
> Study A Bias shares the same variant families as Study A (lexical, surface,
> syntax, instruction) but evaluates against the bias vignettes subset.

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
  --base-cache results_ctrl_invariance/<MODEL>/study_a_bias_generations.jsonl \
  --variant-cache lexical=results_ctrl_invariance/<MODEL>/study_a/lexical/study_a_bias_generations.jsonl \
  --variant-cache surface=results_ctrl_invariance/<MODEL>/study_a/surface/study_a_bias_generations.jsonl \
  --variant-cache syntax=results_ctrl_invariance/<MODEL>/study_a/syntax/study_a_bias_generations.jsonl \
  --variant-cache instruction=results_ctrl_invariance/<MODEL>/study_a/instruction/study_a_bias_generations.jsonl \
  --variant-type lexical=paraphrase \
  --variant-type surface=paraphrase \
  --variant-type syntax=paraphrase \
  --variant-type instruction=paraphrase \
  --out metric-results/<MODEL>/study_a_bias_reverse_invariance.json
```

## Qwen 3.5 27B Distilled (mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2)

LM Studio model via Apple Silicon MLX. Recommended max 4 workers on 48 GB.

### macOS (Apple Silicon)
```bash
python hf-local-scripts/run_invariance_generate_only.py \
  --study study_a_bias_invariance \
  --model-id qwen3.5-distilled \
  --data-dir data/invariance/v5/base/v2_1 \
  --output-dir results_invariance_reverse \
  --workers 4
```

### Windows (PC)
```powershell
python scripts/dev/run_generation_auto.py --study study_a_bias_invariance --model-id qwen3.5-distilled --env mh-llm-benchmark-env --data-dir data/invariance/v5/base/v2_1 --output-dir results_invariance_reverse --workers 4
```
