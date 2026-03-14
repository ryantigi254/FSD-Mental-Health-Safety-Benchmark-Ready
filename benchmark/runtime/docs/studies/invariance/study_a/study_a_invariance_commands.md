# Study A Invariance Commands

## Scope

Study A invariance generation writes to:

- `results.../<model>/study_a_invariance_generations.jsonl`

## `v5` sampled root

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/frozen_splits/v5_invariance_samples
export INVARIANCE_RESULTS_DIR=results_invariance_v5
```

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_a_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

## Controllability-backed sampled root

```bash
cd benchmark/runtime
export INVARIANCE_DATA_DIR=data/controllability_splits_large_resolved_invariance_samples
export INVARIANCE_RESULTS_DIR=results_invariance_controllability
```

```bash
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_a_invariance \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR"
```

## Direct runner

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_study_a_invariance_generate_only.py \
  --model-id qwq \
  --data-dir "$INVARIANCE_DATA_DIR" \
  --output-dir "$INVARIANCE_RESULTS_DIR" \
  --max-samples 5
```
