---
description: Use when the user asks to create, scaffold, or edit Jupyter notebooks (.ipynb) for experiments, explorations, or tutorials.
---

# Jupyter Notebook

## When to Use
- Create a new `.ipynb` notebook from scratch.
- Convert rough notes or scripts into a structured notebook.
- Refactor an existing notebook to be more reproducible and skimmable.
- Build experiments or tutorials that will be read or re-run by others.

## Decision Tree
- If the request is exploratory, analytical, or hypothesis-driven → `experiment` style.
- If the request is instructional, step-by-step, or audience-specific → `tutorial` style.
- If editing an existing notebook → treat as a refactor: preserve intent and improve structure.

## Workflow
1. **Lock the intent**: Identify the notebook kind (experiment or tutorial). Capture the objective, audience, and what "done" looks like.
2. **Scaffold**: Create the notebook with proper metadata, kernel spec, and section structure.
3. **Fill with small, runnable steps**: Keep each code cell focused on one step. Add short markdown cells that explain the purpose and expected result.
4. **Apply the right pattern**:
   - For experiments: hypothesis → setup → run → analyse → conclude.
   - For tutorials: context → step-by-step → exercises → summary.
5. **Validate**: Run the notebook top-to-bottom when the environment allows. If execution is not possible, say so explicitly and call out how to validate locally.

## Quality Checklist
- [ ] Title and objective are clear in the first cell
- [ ] Each code cell is focused on one step
- [ ] Markdown cells explain purpose and expected results
- [ ] No large noisy outputs when a short summary works
- [ ] Top-to-bottom execution produces expected results
- [ ] File naming is stable and descriptive
