# Study C Invariance Commands

## Scope

Study C invariance generation writes to:

- `results.../<model>/study_c_invariance_generations.jsonl`

## Example

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_c_invariance \
  --model-id qwq \
  --data-dir data/controllability_splits_large_resolved_invariance_samples \
  --output-dir results_invariance_controllability
```

## Variant matrix

Build the Study C family menu from the same sampled root:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability \
  --study study_c
```

Available variants:

- `summary_short`
- `summary_long`
- `patient_turn_rephrase`
- `noncritical_reorder`

Run one family without resampling:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_c_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/controllability/study_c/patient_turn_rephrase \
  --output-dir results_invariance_controllability \
  --variant-tag patient_turn_rephrase
```

## Direct runner

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_study_c_invariance_generate_only.py \
  --model-id qwq \
  --data-dir data/controllability_splits_large_resolved_invariance_samples \
  --output-dir results_invariance_controllability \
  --max-cases 3
```
