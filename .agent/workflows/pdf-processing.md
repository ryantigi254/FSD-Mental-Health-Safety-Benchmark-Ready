---
description: Use when tasks involve reading, creating, or reviewing PDF files where rendering and layout matter.
---

# PDF Processing

## When to Use
- Read or review PDF content where layout and visuals matter.
- Create PDFs programmatically with reliable formatting.
- Validate final rendering before delivery.

## Workflow
1. Prefer visual review: render PDF pages to PNGs and inspect them.
   - Use `pdftoppm` (Poppler) if available.
   - If unavailable, ask the user to review the output locally.
2. Use `reportlab` to generate PDFs when creating new documents.
3. Use `pdfplumber` (or `pypdf`) for text extraction and quick checks; do not rely on it for layout fidelity.
4. After each meaningful update, re-render pages and verify alignment, spacing, and legibility.

## Dependencies (install if missing)
```
pip install reportlab pdfplumber pypdf
```

System tools (for rendering):
- Windows: install Poppler via `choco install poppler` or download binaries
- macOS: `brew install poppler`
- Linux: `sudo apt-get install -y poppler-utils`

## Quality Expectations
- Maintain polished visual design: consistent typography, spacing, margins, and section hierarchy.
- Avoid rendering issues: clipped text, overlapping elements, broken tables, black squares, or unreadable glyphs.
- Charts, tables, and images must be sharp, aligned, and clearly labelled.
- Use ASCII hyphens only. Avoid Unicode dashes.
- Citations and references must be human-readable; never leave tool tokens or placeholder strings.

## Final Checks
- Do not deliver until inspection shows zero visual or formatting defects.
- Confirm headers/footers, page numbering, and section transitions look polished.
