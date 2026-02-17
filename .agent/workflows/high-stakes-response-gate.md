---
description: Rare-use gate for security-sensitive requests. Enforce stricter evidence, risk checks, and response controls before final output.
---

# High Stakes Response Gate

## When to Use
- Use rarely.
- Trigger primarily for security topics:
  - Vulnerability analysis
  - Authn/authz logic
  - Secrets handling
  - Privilege boundaries
  - Exploit-relevant implementation details

## Gate Workflow
1. Identify security-critical claims or instructions.
2. Require evidence for each critical claim from trusted sources or repo context.
3. Flag uncertainty explicitly; do not overstate confidence.
4. Ensure recommendations minimise new attack surface.
5. Return only defensible, bounded guidance.

## Output Format
- `Security gate summary`:
  - Critical claims checked
  - Evidence status
  - Remaining uncertainty
  - Safe next action

## Related Workflows
- Use `/source-grounding-enforcer` for evidence retrieval.
- Use `/hallucination-guard-longform` for long security write-ups.
