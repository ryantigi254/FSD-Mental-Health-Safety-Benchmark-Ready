# Study B Multi-Turn Invariance Commands

## Scope

Study B multi-turn invariance generation writes to:

- `results.../<model>/study_b_multi_turn_invariance_generations.jsonl`

## Example

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_multi_turn_invariance \
  --model-id qwq \
  --data-dir data/controllability_splits_large_resolved_invariance_samples \
  --output-dir results_invariance_controllability
```

## Variant matrix

Build the multi-turn family menu from the same sampled root:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability \
  --study study_b_multi_turn
```

Available variants:

- `schedule_earlier`
- `schedule_later`
- `tone_gentle`
- `tone_direct`
- `tone_confrontational`
- `pressure_milder`
- `pressure_stronger`

Run one family without resampling:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_multi_turn_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/controllability/study_b_multi_turn/pressure_stronger \
  --output-dir results_invariance_controllability \
  --variant-tag pressure_stronger
```

## Direct runner

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_study_b_multi_turn_invariance_generate_only.py \
  --model-id qwq \
  --data-dir data/controllability_splits_large_resolved_invariance_samples \
  --output-dir results_invariance_controllability \
  --max-samples 3
```
