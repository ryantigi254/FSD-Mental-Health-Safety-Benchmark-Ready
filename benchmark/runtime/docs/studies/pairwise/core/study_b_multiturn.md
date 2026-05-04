# Study B Multi-turn

## Scope

Core pairwise comparison for the dedicated multi-turn Study B generations.

## Canonical manifest

- `benchmark/runtime/metric-results/pairwise/manifests/study_b_multiturn_case_manifest.json`

## Source cache

- dedicated multi-turn generations under `benchmark/runtime/results/<model>/study_b_multi_turn_generations.jsonl`

## Notes

- this slice should not be rebuilt from the old archived single-turn Study B cache
- multi-turn sequencing, repair quality, and method-fit tags are the main extra pairwise signals here
