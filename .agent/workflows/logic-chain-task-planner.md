---
description: Use for multi-step implementation or debugging tasks that benefit from explicit preconditions, ordered actions, expected state transitions, and validation checks.
---

# Logic Chain Task Planner

## When to Use
- Tasks with several dependent steps.
- When ordering, prerequisites, or state transitions matter.
- Multi-step implementations or debugging sessions.

## Workflow
1. Define goal and non-negotiable constraints.
2. Decompose into minimal ordered steps.
3. For each step, specify:
   - `Preconditions`
   - `Action`
   - `Expected state`
   - `Validation check`
4. Highlight critical invariants that must stay true throughout execution.
5. Flag risky steps requiring rollback paths.

## Output Format

Return an executable plan table with columns:

| Step | Preconditions | Action | Expected State | Check | Rollback |
|------|--------------|--------|----------------|-------|----------|
| ...  | ...          | ...    | ...            | ...   | ...      |

## Related Workflows
- Validate the plan with `/plan-validator-symbolic`.
- Run iterative repair with `/self-correction-loop` during execution.
