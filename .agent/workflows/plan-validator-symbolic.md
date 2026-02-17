---
description: Use to verify step-wise plans against explicit preconditions, effects, and invariants before or during execution.
---

# Plan Validator Symbolic

## When to Use
- When a plan exists and must be checked for logical correctness.
- Before executing high-impact or tightly coupled task sequences.
- When in Plan mode or preparing a proposed plan (hard rule from user rules).

## Validation Workflow
1. Parse plan steps into symbolic tuples:
   - `state_before`
   - `action`
   - `state_after`
2. Check each step for:
   - Preconditions satisfied by `state_before`.
   - Effects consistent with `state_after`.
   - Invariants preserved.
3. Mark invalid transitions and identify the earliest failing step.
4. Propose minimal corrections to restore plan validity.

## Output Format
- `Validation result`: `Valid` or `Invalid`.
- If invalid, include:
  - Failing step index.
  - Broken precondition/invariant.
  - Minimal correction.

## Related Workflows
- Input usually comes from `/logic-chain-task-planner`.
- Use `/self-correction-loop` after applying corrections.
