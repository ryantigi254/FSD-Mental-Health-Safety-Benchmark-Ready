---
description: Generate evidence-backed project run reports from repository state. Use when a user asks for run reports, post-run summaries, implementation retrospectives, or correction logs.
---

# Project Run Reporter

Produce concrete prior-run reports only.

## When to Use
- User asks for run reports, post-run summaries, or retrospectives.
- After completing a major implementation phase.
- When correction logs or change summaries are needed.

## Enforced Assumptions
- Treat every entry as a prior run; never label work as an active task.
- Keep reports concise and technically complete.
- Ground every claim in repository evidence.

## Required Evidence Sources
Collect these every run:
1. `git status --short`
2. `git log --oneline --decorate -n <N>` for the reporting window.
3. `git diff --name-only <range>` or scoped equivalent.
4. Test/lint output when relevant to changed behaviour.
5. Key changed files with concrete paths.

If any required evidence is missing, state `insufficient evidence` and identify exactly what is missing.

## Workflow
1. Collect repo evidence for the target window (commits, file diffs, validation outputs, failures).
2. Draft report using the structure below.
3. Mark unsupported claims as `insufficient evidence`.
4. Return only the final report output (no process narration).

## Report Structure
1. **Summary**: 1–5 bullet overview of what happened.
2. **Changes Made**: files changed, with paths and brief descriptions.
3. **What Failed**: errors, near-misses, unexpected behaviour.
4. **Corrections**: what was fixed and how.
5. **Validation**: test/lint results, verification evidence.
6. **Follow-ups**: remaining items or risks.

## Integrity Rules
- Never invent actions, failures, test results, or file changes.
- Never claim completion without evidence (commit, diff, command output, or concrete file change).
- Never cite vague locations; always use concrete file paths.
