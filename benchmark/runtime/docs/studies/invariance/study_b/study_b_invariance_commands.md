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

## Variant matrix

Study B single-turn concrete variants on the same sampled root:

- `paraphrase`
- `mild`
- `moderate`
- `strong`
- `question`
- `cultural`

Build them with:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py \
  --base-root data/controllability_splits_large_resolved_invariance_samples \
  --output-root data/invariance_variants/controllability \
  --study study_b
```

Run one family without resampling:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/dev/run_generation_auto.py \
  --study study_b_invariance \
  --model-id qwq \
  --data-dir data/invariance_variants/controllability/study_b/cultural \
  --output-dir results_invariance_controllability \
  --variant-tag cultural
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
