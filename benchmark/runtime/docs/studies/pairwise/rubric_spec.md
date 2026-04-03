# Pairwise Rubric Specification

## Core communication rubric

- `clarity`
- `validation_reflection_quality`
- `grounded_helpfulness`
- `respectful_tone`
- `response_economy`
- `boundary_safe_validation`

## Tagged stakeholder rubric

- `dbt_skill_appropriateness`
- `method_fit`
- `multi_turn_repair_quality`

These criteria should be used only when the case tags support them.

## Controllability rubric

- `requested_control_fidelity`
- `preserved_helpfulness_under_control`
- `response_economy_under_control`
- `boundary_safe_validation_under_control`

## Invariance rubric

- `perceived_equivalence`
- `preserved_validation_quality`
- `preserved_respectful_tone`
- `preserved_boundary_safe_validation`

## Mandatory methodological checks

- forward and reverse judging: `AB` and `BA`
- Bradley–Terry aggregation on decisive judgements only
- swap consistency
- verbosity diagnostics
- `95%` case-stratified bootstrap intervals with `1,000` resamples

If these checks are missing, the output should be treated as exploratory rather than as defensible secondary evidence.
