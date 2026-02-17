---
description: Select and route the smallest effective workflow stack for complex tasks. Use when you need to choose dependency-first multi-workflow sequences or compare alternative approaches.
---

# Skill Picker Orchestrator

## When to Use
- Task is multi-step, ambiguous, touches multiple subsystems, or you're unsure which workflows are best.
- Task involves large artefacts (many files, long PDFs, multiple documents).
- Task has strict gates/constraints.

## Routing Workflow

1. Classify the task:
   - Code change / debugging
   - Research / multi-document analysis
   - Long-form factual writing
   - Security-sensitive / high-stakes
   - Feature suggestions/additions
2. Select the smallest workflow stack that covers the task.
3. Order by dependency (dependencies first).
4. If confidence is below threshold, include at least one fallback stack.

## Default Stacks by Task Type

### A) Code change / debugging
`/logic-chain-task-planner` → `/plan-validator-symbolic` → `/self-correction-loop`
- After a fix/refactor: `/regression-pattern-hunter`

### B) Research / multi-document analysis
`/source-grounding-enforcer` → `/validation-gate-runner`

### C) Long-form factual writing
`/hallucination-guard-longform` → `/source-grounding-enforcer`

### D) Feature suggestions/additions
`/uncertainty-calibrated-answering` → `/logic-chain-task-planner` → `/plan-validator-symbolic`

### E) Security-sensitive / high-stakes
`/high-stakes-response-gate` → `/source-grounding-enforcer` → `/hallucination-guard-longform` (if long-form)

## Hard Rules
- Never pick a larger stack when a smaller stack satisfies constraints.
- Never suppress uncertainty; output explicit confidence and fallback.
- Never invent capabilities not present in the available workflows.

## Output Format
1. `Routing summary` with 1–5 bullets.
2. Chosen workflow stack (ordered, dependency-first).
3. Fallback stack (if confidence < 0.80).
4. Selection rationale (1–2 sentences).
