# Pairwise Core Slices

This folder documents the core pairwise comparison slices. These runs compare different model systems against the same frozen benchmark case and are used only for secondary communication-quality evidence.

## Covered slices

- `study_a`
- `study_a_bias`
- `study_b`
- `study_b_multiturn`
- `study_c`

## Command guide

- [Run all core studies](/Users/ryangichuru/Documents/SSD-K/Uni/3rd%20year/NLP-ready/benchmark/runtime/docs/studies/pairwise/core/run_all_core_studies.md)

## Shared expectations

- manifests must come from cached benchmark generations only
- `AB` and `BA` orderings must both be judged
- stacked judging is the default execution mode
- pairwise outputs must not be used as a proxy for factual or clinical correctness
