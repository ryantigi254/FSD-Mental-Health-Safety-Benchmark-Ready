---
description: Generate, deduplicate, rank, and package testable hypotheses from unstructured research inputs. Use when large text corpora, notes, or reports need conversion into measurable hypotheses.
---

# Hypothesis Factory

## When to Use
- Turn unstructured findings into measurable hypotheses.
- High-volume research summarisation and reporting.
- When preparing research outputs for handoff or publication.

## Workflow
1. Extract draft hypotheses from the source material.
2. Deduplicate and merge near-duplicates.
3. For each hypothesis, ensure it has:
   - Explicit metric
   - Expected direction
   - Evidence strength rating
4. Rank by actionability and evidence strength.
5. Package into output artefacts (CSV, report, summary).

## Ranking Rules
- Keep only measurable hypotheses with explicit metric and expected direction.
- Rank by actionability and evidence strength.
- Drop exact duplicates and near-duplicates.

## Output Format
- Ranked hypothesis table with: hypothesis, metric, direction, evidence strength, actionability score.
- Summary report suitable for handoff.

## Related Workflows
- Use `/latex-overleaf-pdflatex-fixer` for LaTeX report quality.
- Use `/pdf-processing` for final PDF review.
- Use `/docx-editing` for DOCX polish when visual fidelity matters.
