# Study B Invariance Commands

## Scope

Study B single-turn invariance generation writes to:

- `results.../<model>/study_b_invariance_generations.jsonl`

## Example

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_invariance \
  --model-id qwq \
  --data-dir data/controllability_splits_large_resolved_invariance_samples \
  --output-dir results_invariance_controllability
```

## Direct runner

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_study_b_invariance_generate_only.py \
  --model-id qwq \
  --data-dir data/controllability_splits_large_resolved_invariance_samples \
  --output-dir results_invariance_controllability \
  --max-samples 5
```
