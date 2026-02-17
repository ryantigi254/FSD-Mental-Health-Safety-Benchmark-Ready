---
description: Use for long-form generation tasks where factual drift is likely. Detect risky entity-level or claim-level content and verify supporting evidence before final output.
---

# Hallucination Guard Longform

## When to Use
- Long answers, reports, planning docs, PRDs, research summaries, and multi-paragraph technical explanations.
- Whenever factual confidence is uncertain across many entities (names, dates, versions, citations, endpoints).

## Required Dependency
- Always invoke `/source-grounding-enforcer` before finalising output.

## Workflow
1. Split the draft into checkable units:
   - Entity spans (names, dates, IDs, versions, citations).
   - Verifiable claims (metrics, capabilities, legal/security statements).
2. Mark each unit as `Needs verification` unless already grounded in provided context.
3. Invoke `/source-grounding-enforcer` to verify each unit.
4. Rewrite or remove unsupported units.
5. Re-run a quick pass to ensure edits did not introduce new unsupported claims.

## Output Format
- Clean final text.
- A short `Verification notes` block with:
  - `Verified`
  - `Corrected`
  - `Unresolved` (if any)
- Never present unresolved claims as facts.

## Related Workflows
- Primary dependency: `/source-grounding-enforcer`.
- Optional hardening for sensitive topics: `/high-stakes-response-gate`.
