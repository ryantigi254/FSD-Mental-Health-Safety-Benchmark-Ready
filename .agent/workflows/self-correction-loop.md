---
description: Use during execution/debugging to iteratively compare expected vs observed results, update hypotheses, and pick the next corrective action.
---

# Self Correction Loop

## When to Use
- When a task is being executed in steps and outcomes can diverge.
- When debugging stalls and hypothesis updates are needed.

## Loop
1. Record expected outcome for the current step.
2. Observe actual result from command/test/output.
3. Classify mismatch:
   - Assumption error
   - Implementation error
   - Environment/config error
   - Data/input error
4. Update hypothesis and choose the smallest next corrective step.
5. Re-test narrowly, then widen validation after a pass.

## Output Format
Keep a compact loop log per cycle:
- `Expected`
- `Observed`
- `Mismatch class`
- `Next action`
- `Validation result`

## Related Workflows
- Commonly follows `/plan-validator-symbolic`.
- Pair with `/regression-pattern-hunter` once the immediate defect is fixed.
