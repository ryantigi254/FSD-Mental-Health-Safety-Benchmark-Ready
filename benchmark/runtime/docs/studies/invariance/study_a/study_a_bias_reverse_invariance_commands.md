# Study A Bias — Reverse Invariance Commands

> **Direction:** Controllability → Invariance.
> Measures which perturbation families change Study A Bias controllability metrics most.
>
> Study A Bias shares the same variant families as Study A (lexical, surface,
> syntax, instruction) but evaluates against the bias vignettes subset.

## Variant Families

| Tag | Type | Data Root |
|-----|------|-----------|
| `lexical` | paraphrase | `data/invariance/ctrl_variants_v2_1/study_a/lexical` |
| `surface` | paraphrase | `data/invariance/ctrl_variants_v2_1/study_a/surface` |
| `syntax` | paraphrase | `data/invariance/ctrl_variants_v2_1/study_a/syntax` |
| `instruction` | paraphrase | `data/invariance/ctrl_variants_v2_1/study_a/instruction` |

## Comparison

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_comparison.py \
  --study study_a_bias \
  --data-root data/invariance/ctrl_samples_v2_1 \
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
