---
description: Use when writing, fixing, or refactoring .tex files for Overleaf, especially when logs include LaTeX errors, table/figure failures, path issues, float issues, or overfull-box issues.
---

# LaTeX Overleaf pdfLaTeX Fixer

Return corrected, compilable LaTeX directly. Do not return high-level advice.

## When to Use
- Writing, fixing, or refactoring `.tex` files for Overleaf.
- When LaTeX compile logs include errors.
- Table/figure failures, path issues, float issues, overfull-box issues.

## Operating Assumptions
- Target compiler: pdfLaTeX on Overleaf.
- Assume a clean Overleaf project: only files in the project directory are accessible.
- Prefer stable, common packages. Avoid exotic dependencies.

## Workflow
1. Read the error log carefully. Identify and fix the first real error (not cascade errors) first.
2. Apply a minimal patch that fixes that error.
3. Scan for the next real error and repeat until all errors are addressed.
4. Keep edits narrow and preserve document meaning.

## Output Format
Always return all of the following in this order:
1. `Patch summary` with 1–5 bullets describing exactly what changed and why.
2. Corrected LaTeX in one fenced code block.
3. If the user requested `full file`, return the full file content, not snippets.

## Hard Rules
- Never write a raw `#` inside normal text or table headers. Escape as `\#`.
- In `tabular`, `tabularx`, and `longtable`:
  - Never use `\\` inside a cell for line breaks. `\\` ends the row.
  - Use `\newline` for in-cell breaks, or `\makecell{...}` when `makecell` is loaded.
- Never leave unbalanced braces or missing `\end{...}`.
- Never use local filesystem paths that Overleaf cannot access.
- For filenames with spaces or special characters: prefer renaming to simple snake_case.
- Use `keepaspectratio` for figures. Prefer: `\includegraphics[width=\linewidth,height=0.75\textheight,keepaspectratio]{...}`.
- Reduce overfull hbox in tables: use `xurl`, ragged-right `X` columns in `tabularx`.

## Preferred Packages
- `graphicx`, `xcolor`, `booktabs`, `tabularx`, `array`, `hyperref`, `xurl`, `microtype`

## Content Preservation
- If table formatting changes, preserve meaning and citations.
- If a resource is missing, do not remove it silently; keep a named placeholder box.
- Do not tell the user to compile as the main step; implement the fixes directly.
