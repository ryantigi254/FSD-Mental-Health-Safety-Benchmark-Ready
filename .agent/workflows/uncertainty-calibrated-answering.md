---
description: Use only when the user explicitly asks for feature suggestions or feature additions. Produce confidence-calibrated feature recommendations and abstain on weakly supported ideas.
---

# Uncertainty-Calibrated Answering

## When to Use
- Only when the request is about feature ideas, feature suggestions, or feature additions.
- Do not use for bug fixes, refactors, or implementation-only tasks unless the user explicitly asks for feature options.

## Workflow
1. Extract product context, constraints, and target user outcome from the prompt/repo.
2. Generate candidate features with explicit rationale tied to the stated outcome.
3. Assign confidence tiers for each feature:
   - **High**: strongly supported by context/evidence.
   - **Medium**: plausible but needs one missing assumption.
   - **Low**: speculative; mark as uncertain.
4. For medium/low items, list missing evidence needed to raise confidence.
5. Rank by value-to-complexity and expected regression risk.
6. If confidence is low across all options, abstain from hard recommendations and return the minimum context required.

## Output Format

| Feature | Why Now | Confidence | Risk | Evidence Missing |
|---------|---------|------------|------|-----------------|
| ...     | ...     | ...        | ...  | ...             |

- Keep speculative ideas explicitly labelled as `Speculation`.

## Related Workflows
- Use with `/logic-chain-task-planner` to convert selected features into implementable plans.
- Use with `/plan-validator-symbolic` before implementation if the feature has strict state transitions.
