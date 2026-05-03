# Pairwise Invariance Slices

This folder documents the invariance arm of pairwise evaluation. These runs compare matched responses for semantically equivalent variants rather than comparing different benchmark systems on the same prompt.

## Covered invariance families

- `core` via `invariance`
- `ctrl_invariance` via `invariance_under_control`
- `invariance_ctrl` via `control_under_invariance`

## Command Guide

- [Run all invariance studies](run_all_invariance_studies.md)
- [Local LM Studio commands](../local_lmstudio_commands.md)

## Shared expectations

- only matched base-versus-perturbed pairs belong here
- pairwise checks whether perceived quality is preserved across equivalent variants
- invariance pairwise is secondary evidence and should be read alongside the benchmark metrics, not instead of them
