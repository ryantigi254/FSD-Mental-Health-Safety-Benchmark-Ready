# Study A Bias

## Scope

Core pairwise comparison for the bias-probe Study A slice.

## Canonical manifest

- `benchmark/runtime/metric-results/pairwise/manifests/study_a_bias_case_manifest.json`

## Source cache

- aligned raw bias generations under `benchmark/runtime/results/<model>/study_a_bias_generations.jsonl`

## Notes

- this slice should use the raw bias generations rather than the mixed processed pipeline because the processed files are not schema-consistent across models
- pairwise remains a communication-quality layer; it must not be used to score the substantive bias outcome itself
