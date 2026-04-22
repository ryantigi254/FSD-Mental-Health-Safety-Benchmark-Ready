# Pairwise Controllability Slices

This folder documents the pairwise controllability arm. These manifests compare the three control arms within the same model and case, rather than comparing different models on the same case.

## Covered slices

- `study_a_controllability`
- `study_a_bias_controllability`
- `study_b_controllability`
- `study_b_multiturn_controllability`
- `study_c_controllability`

## Shared expectations

- each manifest case is model-specific
- expected arms are `spontaneous`, `generic_control`, and `explicit_control` when available
- the primary judgement target is requested-control fidelity without sacrificing helpfulness or boundary safety
