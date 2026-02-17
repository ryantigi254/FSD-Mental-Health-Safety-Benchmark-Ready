---
description: Use when the task involves reading, creating, or editing .docx documents, especially when formatting or layout fidelity matters.
---

# DOCX Editing

## When to Use
- Read or review DOCX content where layout matters (tables, diagrams, pagination).
- Create or edit DOCX files with professional formatting.
- Validate visual layout before delivery.

## Workflow
1. Prefer visual review (layout, tables, diagrams).
   - If `soffice` and `pdftoppm` are available, convert DOCX → PDF → PNGs.
   - If these tools are missing, ask the user to review rendered pages locally.
2. Use `python-docx` for edits and structured creation (headings, styles, tables, lists).
3. After each meaningful change, re-render and inspect the pages.
4. If visual review is not possible, extract text with `python-docx` as a fallback and call out layout risk.

## Dependencies (install if missing)
```
pip install python-docx pdf2image
```

System tools (for rendering):
- Windows: install LibreOffice and Poppler
- macOS: `brew install libreoffice poppler`
- Linux: `sudo apt-get install -y libreoffice poppler-utils`

## Quality Expectations
- Deliver a client-ready document: consistent typography, spacing, margins, and clear hierarchy.
- Avoid formatting defects: clipped/overlapping text, broken tables, unreadable characters, or default-template styling.
- Charts, tables, and visuals must be legible with correct alignment.
- Use ASCII hyphens only.
- Citations and references must be human-readable.

## Final Checks
- Re-render and inspect every page at 100% zoom before final delivery.
- Fix any spacing, alignment, or pagination issues and repeat the render loop.
