---
description: Build and enforce explicit pass/fail validation contracts for autonomous runs. Use when tasks require clear acceptance checks, iterative test loops, and fail-closed behaviour when validation targets are missing.
---

# Validation Gate Runner

## When to Use
- When a task has explicit pass/fail acceptance criteria.
- When iterative testing loops are needed until checks pass.
- When validation must fail closed if acceptance tests are absent.

## Workflow
1. Compile a `ValidationContract` from the task input:
   - List every acceptance criterion as a discrete check.
   - Each check must have: condition, expected outcome, evidence source.
2. Reject execution when no checks are present (fail closed).
3. Run checks in order, noting pass/fail per check.
4. If any check fails:
   - Record the failure with evidence.
   - Attempt corrective action if within scope.
   - Re-run the failed check.
5. Stop when all checks pass or stop conditions are reached.
6. Return final pass/fail state with evidence.

## Output Format
- `Validation summary`:
  - Total checks / passed / failed
  - Per-check result with evidence
  - Overall verdict: `Pass` or `Fail`
- If failed, include:
  - Failing check description
  - Root cause candidate
  - Mitigation hint

## Related Workflows
- Consume run outputs from execution workflows.
- Feed failure data to `/regression-pattern-hunter`.
