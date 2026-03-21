#!/usr/bin/env python3
"""
Generate TikZ .tex files from distribution_analysis.json.

Uses ONLY pure TikZ (no pgfplots, no standalone) — works with basic TeX Live.
Each file is a self-contained tikzpicture that can be \\input{} in any document.
A wrapper _compile.tex is also generated for standalone compilation.

Outputs to  analysis/figures/tikz/
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures" / "tikz"
OUT.mkdir(parents=True, exist_ok=True)

with open(Path(__file__).resolve().parent / "distribution_analysis.json") as f:
    A = json.load(f)


def escape_tex(s: str) -> str:
    for ch in ["&", "%", "#", "_"]:
        s = s.replace(ch, f"\\{ch}")
    return s


def normalise_name(cond: str) -> str:
    acronyms = {"ocd", "ptsd", "adhd", "dbt", "cbt", "act", "emdr", "bpd"}
    words = cond.split()
    out = []
    for w in words:
        if w.lower() in acronyms:
            out.append(w.upper())
        elif w.lower() in {"with", "and", "in", "of", "on", "for", "the", "a"} and out:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)


# ── Shared colour definitions ────────────────────────────────────────────────
COLOUR_DEFS = r"""% === Colour definitions ===
\definecolor{catTrauma}{HTML}{3B82B0}
\definecolor{catAnxiety}{HTML}{3B82B0}
\definecolor{catMood}{HTML}{3B82B0}
\definecolor{catOCD}{HTML}{3B82B0}
\definecolor{catOther}{HTML}{3B82B0}
\definecolor{catSomatic}{HTML}{3B82B0}
\definecolor{catEating}{HTML}{3B82B0}
\definecolor{catNeuro}{HTML}{3B82B0}
\definecolor{catPsychotic}{HTML}{3B82B0}
\definecolor{catPersonality}{HTML}{3B82B0}
\definecolor{catSelfHarm}{HTML}{3B82B0}
\definecolor{catSubstance}{HTML}{3B82B0}
\definecolor{catSleep}{HTML}{3B82B0}
\definecolor{catGold}{HTML}{E9C46A}
\definecolor{catUnclass}{HTML}{94A3B8}
\definecolor{sevCritical}{HTML}{9B2226}
\definecolor{sevSevere}{HTML}{E10600}
\definecolor{sevModerate}{HTML}{F07B83}
\definecolor{sevMild}{HTML}{F6C1C4}
\definecolor{studyA}{HTML}{3B82C4}
\definecolor{studyB}{HTML}{F4A261}
\definecolor{studyC}{HTML}{2A9D8F}
\definecolor{modTeal}{HTML}{3C8D93}
\definecolor{textdark}{HTML}{111827}
\definecolor{gridlight}{HTML}{D8DEE9}
"""

CAT_COL = {
    "Trauma & Stressor-Related": "catTrauma",
    "Anxiety Disorders": "catAnxiety",
    "Mood Disorders": "catMood",
    "OCD & Related": "catOCD",
    "Other": "catOther",
    "Somatic & Health-Related": "catSomatic",
    "Eating Disorders": "catEating",
    "Neurodevelopmental": "catNeuro",
    "Psychotic Spectrum": "catPsychotic",
    "Personality Disorders": "catPersonality",
    "Self-Harm & Suicidality": "catSelfHarm",
    "Substance Use Disorders": "catSubstance",
    "Sleep-Wake Disorders": "catSleep",
}


def write_file(name: str, content: str):
    (OUT / name).write_text(content, encoding="utf-8")
    print(f"  {name}")


# ═════════════════════════════════════════════════════════════════════════════
# HORIZONTAL BAR CHART (pure TikZ)
# ═════════════════════════════════════════════════════════════════════════════

def hbar_tikz(
    *,
    names: list[str],
    values: list[float],
    colours: list[str],
    labels: list[str],
    title: str,
    xlabel: str,
    chart_width_cm: float = 8.0,
    bar_height_cm: float = 0.32,
    gap_cm: float = 0.12,
    label_fontsize: str = r"\tiny",
    value_fontsize: str = r"\scriptsize",
) -> str:
    """Return a tikzpicture string for a horizontal bar chart."""
    n = len(names)
    max_val = max(values) if values else 1
    total_height = n * (bar_height_cm + gap_cm)
    scale_x = chart_width_cm / max_val

    lines = []
    lines.append(r"\begin{tikzpicture}")

    # Grid lines
    for g in range(0, int(max_val) + 1, max(1, int(max_val / 5))):
        x = g * scale_x
        lines.append(
            rf"  \draw[gridlight, line width=0.4pt] ({x:.3f},0) -- ({x:.3f},{total_height:.2f});"
        )
        lines.append(
            rf"  \node[below, font=\tiny, text=textdark] at ({x:.3f},-0.15) {{{g:,}}};".replace(",}", "}")
        )

    # Bars
    for i, (name, val, col, label) in enumerate(zip(names, values, colours, labels)):
        y = i * (bar_height_cm + gap_cm)
        w = val * scale_x
        lines.append(
            rf"  \fill[{col}] (0,{y:.3f}) rectangle ({w:.3f},{y + bar_height_cm:.3f});"
        )
        # Label on left
        lines.append(
            rf"  \node[left, font={label_fontsize}, text=textdark] at (-0.1,{y + bar_height_cm / 2:.3f}) {{{escape_tex(name)}}};"
        )
        # Value on right
        lines.append(
            rf"  \node[right, font={value_fontsize}, text=textdark] at ({w + 0.08:.3f},{y + bar_height_cm / 2:.3f}) {{{label}}};"
        )

    # Axis lines
    lines.append(rf"  \draw[textdark, line width=0.5pt] (0,0) -- (0,{total_height:.2f});")
    lines.append(rf"  \draw[textdark, line width=0.5pt] (0,0) -- ({chart_width_cm + 0.5:.2f},0);")

    # Title
    lines.append(
        rf"  \node[above right, font=\small\bfseries, text=textdark] at (0,{total_height + 0.15:.2f}) {{{escape_tex(title)}}};"
    )

    # X-axis label
    lines.append(
        rf"  \node[below, font=\tiny, text=textdark] at ({chart_width_cm / 2:.2f},-0.55) {{{escape_tex(xlabel)}}};"
    )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1: DSM-5 Categories
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig1():
    cats = A["diagnostic_categories"]
    names = list(cats.keys())
    vals = [cats[n]["total"] for n in names]
    pcts = [cats[n]["pct"] for n in names]
    cols = [CAT_COL.get(n, "catUnclass") for n in names]
    lbls = [f"{v:,} ({p:.1f}\\%)".replace(",}", "}") for v, p in zip(vals, pcts)]

    body = hbar_tikz(
        names=names, values=vals, colours=cols, labels=lbls,
        title="A. DSM-5 Diagnostic Category Coverage",
        xlabel="Number of Samples",
        chart_width_cm=9.0, bar_height_cm=0.38, gap_cm=0.14,
        label_fontsize=r"\scriptsize",
    )

    content = f"% Fig 1: DSM-5 Diagnostic Categories\n% Auto-generated — do not edit\n{body}\n"
    write_file("fig1_categories.tex", content)
    return body


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2: Top 25 Conditions
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig2():
    conds = A["specific_conditions"]
    top_n = 25
    keys = list(conds.keys())[:top_n]
    names = [normalise_name(k) for k in keys]
    vals = [conds[k]["total"] for k in keys]
    cols = ["catGold"] * len(keys)
    lbls = [str(v) for v in vals]

    body = hbar_tikz(
        names=names, values=vals, colours=cols, labels=lbls,
        title="B. Top 25 Clinical Conditions",
        xlabel="Number of Samples",
        chart_width_cm=9.0, bar_height_cm=0.30, gap_cm=0.10,
        label_fontsize=r"\tiny",
        value_fontsize=r"\tiny",
    )

    content = f"% Fig 2: Top 25 Specific Conditions\n% Auto-generated — do not edit\n{body}\n"
    write_file("fig2_conditions.tex", content)
    return body


# ═════════════════════════════════════════════════════════════════════════════
# Fig 3: Severity Donut
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig3():
    sev = A["severity"]
    order = ["Critical", "Severe", "Moderate", "Mild"]
    order = [s for s in order if s in sev]
    total = sum(sev[s]["total"] for s in order)
    sev_col = {"Critical": "sevCritical", "Severe": "sevSevere",
               "Moderate": "sevModerate", "Mild": "sevMild"}

    lines = []
    lines.append("% Fig 3: Severity Distribution (donut)")
    lines.append("% Auto-generated — do not edit")
    lines.append(r"\begin{tikzpicture}")

    outer_r, inner_r = 2.6, 1.6
    start = 90

    wedge_data = []
    for s in order:
        angle = 360 * sev[s]["total"] / total
        end = start - angle
        wedge_data.append((s, sev[s]["total"], sev[s]["pct"], start, end, sev_col[s]))
        start = end

    # Draw wedges
    for name, count, pct, a_s, a_e, col in wedge_data:
        lines.append(
            f"  \\fill[{col}] ({a_s}:{inner_r}) "
            f"arc ({a_s}:{a_e}:{inner_r}) -- ({a_e}:{outer_r}) "
            f"arc ({a_e}:{a_s}:{outer_r}) -- cycle;"
        )

    # White borders
    lines.append(f"  \\draw[white, line width=1.8pt] (0,0) circle ({inner_r});")
    lines.append(f"  \\draw[white, line width=1.8pt] (0,0) circle ({outer_r});")
    for _, _, _, a_s, _, _ in wedge_data:
        lines.append(f"  \\draw[white, line width=1.8pt] ({a_s}:{inner_r}) -- ({a_s}:{outer_r});")

    # Centre text
    lines.append(
        rf"  \node[font=\small\bfseries, text=textdark, align=center] at (0,0) {{Severity\\[-2pt]N\,=\,{total:,}}};".replace(",}", "}")
    )

    # External labels
    for name, count, pct, a_s, a_e, col in wedge_data:
        mid = (a_s + a_e) / 2
        lx = 3.4 * math.cos(math.radians(mid))
        ly = 3.4 * math.sin(math.radians(mid))
        ha = "west" if lx > 0.1 else ("east" if lx < -0.1 else "center")
        lines.append(
            rf"  \node[anchor={ha}, font=\scriptsize\bfseries, text=textdark, align=center] "
            rf"at ({lx:.2f},{ly:.2f}) {{{name}\\[-1pt]{count:,} ({pct:.1f}\%)}};".replace(",}", "}")
        )

    # Title
    lines.append(
        rf"  \node[above, font=\small\bfseries, text=textdark] at (0,{outer_r + 0.6:.1f}) {{C.\ Severity Distribution}};"
    )

    lines.append(r"\end{tikzpicture}")
    body = "\n".join(lines)
    write_file("fig3_severity.tex", body + "\n")
    return body


# ═════════════════════════════════════════════════════════════════════════════
# Fig 4: Therapeutic Modalities
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig4():
    mods = A["therapeutic_modalities"]
    names = list(mods.keys())
    vals = [mods[m]["mentions"] for m in names]
    pcts = [mods[m]["pct_of_study_a"] for m in names]
    cols = ["modTeal"] * len(names)
    lbls = [f"{v:,} ({p:.1f}\\%)".replace(",}", "}") for v, p in zip(vals, pcts)]

    body = hbar_tikz(
        names=names, values=vals, colours=cols, labels=lbls,
        title="Therapeutic Modality References (Study A Reasoning)",
        xlabel="Number of Mentions",
        chart_width_cm=9.0, bar_height_cm=0.38, gap_cm=0.14,
        label_fontsize=r"\scriptsize",
    )

    content = f"% Fig 4: Therapeutic Modalities\n% Auto-generated — do not edit\n{body}\n"
    write_file("fig4_modalities.tex", content)
    return body


# ═════════════════════════════════════════════════════════════════════════════
# Fig 5: Per-Study Stacked Bar
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig5():
    cats = A["diagnostic_categories"]
    names = list(cats.keys())
    chart_w = 9.0
    bh = 0.38
    gap = 0.14
    n = len(names)
    total_h = n * (bh + gap)

    max_val = max(cats[c]["total"] for c in names)
    sx = chart_w / max_val

    lines = []
    lines.append("% Fig 5: Per-Study Stacked Bar")
    lines.append("% Auto-generated — do not edit")
    lines.append(r"\begin{tikzpicture}")

    # Grid
    for g in range(0, int(max_val) + 1, max(1, int(max_val / 5))):
        x = g * sx
        lines.append(rf"  \draw[gridlight, line width=0.4pt] ({x:.3f},0) -- ({x:.3f},{total_h:.2f});")
        lines.append(rf"  \node[below, font=\tiny, text=textdark] at ({x:.3f},-0.15) {{{g:,}}};".replace(",}", "}"))

    study_cols = {"A": "studyA", "B": "studyB", "C": "studyC"}

    for i, name in enumerate(names):
        y = i * (bh + gap)
        left = 0.0
        for study in ["A", "B", "C"]:
            val = cats[name]["per_study"].get(study, 0)
            if val == 0:
                continue
            w = val * sx
            lines.append(
                rf"  \fill[{study_cols[study]}] ({left:.3f},{y:.3f}) rectangle ({left + w:.3f},{y + bh:.3f});"
            )
            left += w

        # Name label
        lines.append(
            rf"  \node[left, font=\scriptsize, text=textdark] at (-0.1,{y + bh / 2:.3f}) {{{escape_tex(name)}}};"
        )
        # Total label
        total = cats[name]["total"]
        lines.append(
            rf"  \node[right, font=\scriptsize, text=textdark] at ({total * sx + 0.08:.3f},{y + bh / 2:.3f}) {{{total:,}}};".replace(",}", "}")
        )

    # Axes
    lines.append(rf"  \draw[textdark, line width=0.5pt] (0,0) -- (0,{total_h:.2f});")
    lines.append(rf"  \draw[textdark, line width=0.5pt] (0,0) -- ({chart_w + 0.5:.2f},0);")

    # Title
    lines.append(
        rf"  \node[above right, font=\small\bfseries, text=textdark] at (0,{total_h + 0.15:.2f}) {{Diagnostic Category Split by Study}};"
    )

    # Legend
    lx = chart_w - 1.5
    ly = 0.3
    for j, (study, col) in enumerate(study_cols.items()):
        ox = j * 2.0
        lines.append(rf"  \fill[{col}] ({lx + ox:.1f},{ly}) rectangle ({lx + ox + 0.3:.1f},{ly + 0.2});")
        lines.append(
            rf"  \node[right, font=\tiny, text=textdark] at ({lx + ox + 0.35:.1f},{ly + 0.1}) {{Study {study}}};"
        )

    lines.append(r"\end{tikzpicture}")
    body = "\n".join(lines)
    write_file("fig5_per_study.tex", body + "\n")
    return body


# ═════════════════════════════════════════════════════════════════════════════
# Compilable wrapper
# ═════════════════════════════════════════════════════════════════════════════

def gen_wrappers():
    """Generate a compilable wrapper for each figure and a combined one."""
    fig_files = [
        ("fig1_categories", "DSM-5 Categories"),
        ("fig2_conditions", "Top 25 Conditions"),
        ("fig3_severity", "Severity Distribution"),
        ("fig4_modalities", "Therapeutic Modalities"),
        ("fig5_per_study", "Per-Study Breakdown"),
    ]

    for fname, desc in fig_files:
        content = (
            r"\documentclass[a4paper,10pt]{article}" + "\n"
            r"\usepackage[T1]{fontenc}" + "\n"
            r"\usepackage[margin=1cm]{geometry}" + "\n"
            r"\usepackage{tikz}" + "\n"
            r"\usepackage{xcolor}" + "\n"
            r"\renewcommand{\familydefault}{\rmdefault}" + "\n"
            r"\pagestyle{empty}" + "\n"
            + COLOUR_DEFS + "\n"
            r"\begin{document}" + "\n"
            rf"\input{{{fname}}}" + "\n"
            r"\end{document}" + "\n"
        )
        write_file(f"_compile_{fname}.tex", content)

    # Combined three-panel
    combined = (
        r"\documentclass[a4paper,landscape,10pt]{article}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        r"\usepackage[margin=1.0cm]{geometry}" + "\n"
        r"\usepackage{graphicx}" + "\n"
        r"\usepackage{tikz}" + "\n"
        r"\usepackage{xcolor}" + "\n"
        r"\renewcommand{\familydefault}{\rmdefault}" + "\n"
        r"\pagestyle{empty}" + "\n"
        + COLOUR_DEFS + "\n"
        r"\begin{document}" + "\n"
        r"\noindent" + "\n"
        r"\begin{minipage}[t]{0.29\textwidth}" + "\n"
        r"\centering" + "\n"
        r"\resizebox{0.92\linewidth}{!}{\input{fig1_categories}}" + "\n"
        r"\end{minipage}\hspace{0.02\textwidth}" + "\n"
        r"\begin{minipage}[t]{0.39\textwidth}" + "\n"
        r"\centering" + "\n"
        r"\resizebox{0.92\linewidth}{!}{\input{fig2_conditions}}" + "\n"
        r"\end{minipage}\hspace{0.02\textwidth}" + "\n"
        r"\begin{minipage}[t]{0.26\textwidth}" + "\n"
        r"\centering" + "\n"
        r"\resizebox{0.88\linewidth}{!}{\input{fig3_severity}}" + "\n"
        r"\end{minipage}" + "\n"
        r"\end{document}" + "\n"
    )
    write_file("_compile_combined.tex", combined)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating TikZ files...")
    gen_fig1()
    gen_fig2()
    gen_fig3()
    gen_fig4()
    gen_fig5()
    gen_wrappers()
    print(f"Done. All .tex files in {OUT}/")
