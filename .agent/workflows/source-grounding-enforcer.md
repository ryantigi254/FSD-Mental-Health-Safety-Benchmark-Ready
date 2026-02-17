---
description: Use when claims need evidence or when context is incomplete. Verify claims against provided links first, then perform targeted research when needed, and cite sources in the final output.
---

# Source Grounding Enforcer

## When to Use
- When the response contains factual claims that require proof.
- When links are provided in the prompt.
- When the model lacks enough context and must fetch documentation or external references.

## Priority Order
1. Provided context files.
2. User-provided links.
3. Targeted external research (web search, browser tools).

## Research Workflow
1. Build a claim checklist from the task.
2. If links are provided:
   - Visit each link and extract evidence before doing any broader search.
3. If context is still missing:
   - Use browser tools for page-level extraction.
   - Use web search only for unresolved claims after link checks.
4. Label each claim:
   - `Verified`
   - `Contradicted`
   - `Unresolved`
5. Update the draft so contradicted/unresolved claims are corrected, caveated, or removed.

## Citation Rules
- Cite concrete sources at the end of the response.
- Prefer primary documentation and official references.
- If evidence is weak or absent, state uncertainty explicitly instead of fabricating support.

## Output Format
- `Grounding summary`:
  - Verified claims
  - Contradictions fixed
  - Unresolved items
- `Sources` list (end of response).

## Related Workflows
- Called by `/hallucination-guard-longform`.
- Pair with `/high-stakes-response-gate` for security-sensitive content.
