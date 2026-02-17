---
description: Use after a bug fix or behaviour change to find nearby occurrences of the same defect pattern and add targeted regression checks.
---

# Regression Pattern Hunter

## When to Use
- Immediately after bug fixes.
- After risky refactors or interface/contract changes.

## Workflow
1. Identify the fixed defect signature:
   - Failing condition
   - Relevant symbols
   - Data shape or state pattern
2. Search neighbouring modules and callsites for the same signature.
3. Prioritise high-risk matches by execution frequency and impact.
4. Add minimal targeted tests for the top-risk matches.
5. Run narrow tests first, then the closest integration surface.

## Output Format
- `Hotspot list` with file paths and why each is similar.
- `Regression checks` added or recommended.
- `Residual risk` if any untested hotspot remains.

## Related Workflows
- Use with `/self-correction-loop` for post-fix learning.
- Use `/logic-chain-task-planner` when a broad regression sweep is needed.
- Consume structured failures from `/validation-gate-runner`.
