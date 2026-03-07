#!/usr/bin/env python3
"""
Generate a single paper-ready PGFPlots radar figure from distribution_analysis.json.

Outputs:
  analysis/figures/pgfplots/clinical_overview_radar.tex
  analysis/figures/pgfplots/_compile_clinical_overview_radar.tex
"""

import json
import math
from pathlib import Path
import re


BASE = Path(__file__).resolve().parent
OUT = BASE / "figures" / "pgfplots"
OUT.mkdir(parents=True, exist_ok=True)


def esc_tex(s: str) -> str:
    for ch in ["&", "%", "#", "_"]:
        s = s.replace(ch, f"\\{ch}")
    return s


def pct(value: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (100.0 * value) / total


def format_coords(
    values: list[float],
    *,
    base_angle: float = 0.0,
    angle_offset: float = 0.0,
    axis_factors: list[float] | None = None,
    max_radius: float = 115.0,
) -> str:
    step = 360.0 / len(values)
    coords = []
    for i, v in enumerate(values):
        angle = base_angle + angle_offset + (i * step)
        factor = axis_factors[i] if axis_factors is not None else 1.0
        vv = min(max_radius, v * factor)
        coords.append(f"({angle:.2f},{vv:.3f})")
    close_factor = axis_factors[0] if axis_factors is not None else 1.0
    close_v = min(max_radius, values[0] * close_factor)
    coords.append(f"({base_angle + angle_offset + 360.0:.2f},{close_v:.3f})")
    return " ".join(coords)


def nice_rmax(max_value: float) -> int:
    """Round up to a clean radar max so outer series are not clipped."""
    if max_value <= 40:
        return 40
    rounded = int(((max_value * 1.08) + 9) // 10 * 10)
    return max(50, rounded)


def ranked_categories_for_scope(analysis: dict, scope: str) -> list[tuple[str, int]]:
    cat_data = analysis["diagnostic_categories"]
    ranked: list[tuple[str, int]] = []
    for category, info in cat_data.items():
        value = info["total"] if scope == "overall" else info["per_study"][scope]
        ranked.append((category, value))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def select_radar_categories(analysis: dict, top_k: int = 7) -> list[str]:
    selected: list[str] = []
    for scope in ["overall", "A", "B", "C"]:
        ranked = ranked_categories_for_scope(analysis, scope)
        if not ranked:
            continue
        cutoff_index = min(top_k, len(ranked)) - 1
        cutoff_value = ranked[cutoff_index][1]
        for category, value in ranked:
            if value <= 0 or value < cutoff_value:
                continue
            if category not in selected:
                selected.append(category)
    return selected


def order_radar_categories(categories: list[str], analysis: dict) -> list[str]:
    cat_data = analysis["diagnostic_categories"]
    ranked = sorted(
        categories,
        key=lambda category: (-cat_data[category]["total"], categories.index(category)),
    )
    ordered = [""] * len(ranked)
    preferred_slots = [0]
    for offset in range(1, len(ranked)):
        if offset % 2 == 1:
            preferred_slots.append((offset + 1) // 2)
        else:
            preferred_slots.append(len(ranked) - (offset // 2))
    for category, slot in zip(ranked, preferred_slots):
        ordered[slot] = category
    return ordered


def category_display_label(category: str) -> str:
    display_label_map = {
        "Trauma & Stressor-Related": "Trauma /\\\\Stressor-Related",
        "Anxiety Disorders": "Anxiety\\\\Disorders",
        "Mood Disorders": "Mood\\\\Disorders",
        "OCD & Related": "OCD\\\\Related",
        "Other": "Other",
        "Somatic & Health-Related": "Somatic /\\\\Health-Related",
        "Eating Disorders": "Eating\\\\Disorders",
        "Neurodevelopmental": "Neurodevelop-\\\\mental",
        "Sleep-Wake Disorders": "Sleep-Wake\\\\Disorders",
        "Personality Disorders": "Personality\\\\Disorders",
        "Psychotic Spectrum": "Psychotic\\\\Spectrum",
        "Self-Harm & Suicidality": "Self-Harm /\\\\Suicidality",
    }
    return display_label_map.get(category, category.replace(" & ", " /\\\\"))


def build_scale_ticks(y_max: int) -> list[int]:
    if y_max <= 20:
        step = 5
    else:
        step = 10
    return list(range(step, y_max + 1, step))


def sqrt_radius_scale(value: float, y_max: float) -> float:
    if y_max <= 0 or value <= 0:
        return 0.0
    return math.sqrt(value / y_max) * y_max


def emphasise_top_half_radius(radius: float, angle: float, y_max: float) -> float:
    vertical = math.sin(math.radians(angle))
    horizontal = abs(math.cos(math.radians(angle)))
    if vertical <= 0.15 and horizontal <= 0.45:
        return radius
    top_strength = max(0.0, vertical)
    side_strength = 0.0 if horizontal <= 0.45 else (horizontal - 0.45) / 0.55
    # Taper the asymmetric boost for near-zero values so sparse series do not
    # turn into exaggerated spikes when adjacent axes sit at zero.
    emphasis_weight = min(1.0, radius / (y_max * 0.30)) if y_max > 0 else 0.0
    boost = 1.0
    if top_strength > 0.0:
        boost += (0.78 + (0.38 * min(1.0, top_strength))) * emphasis_weight
    if side_strength > 0.0:
        boost += (0.34 + (0.24 * min(1.0, side_strength))) * emphasis_weight
    return min(y_max * 0.94, radius * boost)


def transform_radar_series(values: list[float], y_max: float, base_angle: float = -90.0) -> list[float]:
    step = 360.0 / len(values)
    transformed: list[float] = []
    for index, value in enumerate(values):
        angle = base_angle + (index * step)
        radius = sqrt_radius_scale(value, y_max)
        transformed.append(emphasise_top_half_radius(radius, angle, y_max))
    return transformed


def manual_label_commands(categories: list[str], y_max: int, base_angle: float = -90.0) -> str:
    step = 360.0 / len(categories)
    commands: list[str] = []
    for index, category in enumerate(categories):
        angle = base_angle + (index * step)
        radians = math.radians(angle)
        x = math.cos(radians)
        y = math.sin(radians)
        label_radius = y_max * (1.03 if y > 0.15 else 1.11 if y < -0.15 else 1.07)
        if abs(y) > 0.86:
            anchor = "south" if y > 0 else "north"
        elif x > 0.12:
            anchor = "west"
        elif x < -0.12:
            anchor = "east"
        else:
            anchor = "south" if y > 0 else "north"
        node_name = f"label{index}"
        commands.append(
            rf"\node[font=\Large\bfseries, align=center, text=gray!88!black, text width=3.5cm, anchor={anchor}] "
            rf"({node_name}) at (axis cs:{angle:.2f},{label_radius:.2f}) {{{category_display_label(category)}}};"
        )
        commands.append(
            rf"\draw[gray!50, line width=0.35pt] (axis cs:{angle:.2f},{y_max}) -- ({node_name}.{anchor});"
        )
    return "\n    ".join(commands)


def build_radar_tex(analysis: dict) -> str:
    cat_data = analysis["diagnostic_categories"]
    categories = order_radar_categories(select_radar_categories(analysis, top_k=7), analysis)

    overall = []
    study_a = []
    study_b = []
    study_c = []
    for category in categories:
        info = cat_data[category]
        overall.append(pct(info["total"], analysis["total_samples"]))
        study_a.append(pct(info["per_study"]["A"], analysis["per_study"]["A"]))
        study_b.append(pct(info["per_study"]["B"], analysis["per_study"]["B"]))
        study_c.append(pct(info["per_study"]["C"], analysis["per_study"]["C"]))

    r_max = max(max(overall), max(study_a), max(study_b), max(study_c))
    y_max = nice_rmax(r_max)
    scale_ticks = build_scale_ticks(y_max)
    plot_overall = transform_radar_series(overall, float(y_max))
    plot_study_a = transform_radar_series(study_a, float(y_max))
    plot_study_b = transform_radar_series(study_b, float(y_max))
    plot_study_c = transform_radar_series(study_c, float(y_max))
    y_ticks = ",".join(f"{sqrt_radius_scale(value, float(y_max)):.3f}" for value in scale_ticks)
    y_tick_labels = ",".join("{" + f"{value / 100.0:.1f}" + "}" for value in scale_ticks)
    tick_labels = ",".join("{}" for _ in categories)
    label_commands = manual_label_commands(categories, y_max)

    return rf"""% Auto-generated PGFPlots radar figure. Do not edit by hand.
\begin{{tikzpicture}}
  \definecolor{{studyApurple}}{{HTML}}{{4B1D9A}}
  \begin{{polaraxis}}[
    width=0.72\linewidth,
    height=0.72\linewidth,
    at={{(0,0.48cm)}},
    ymin=0, ymax={y_max},
    ytick={{{y_ticks}}},
    yticklabels={{{y_tick_labels}}},
    xtick=data,
    xticklabels={{{tick_labels}}},
    clip=false,
    tick label style={{font=\large\bfseries, text=gray!68!black}},
    yticklabel style={{font=\large\bfseries, text=gray!68!black}},
    xticklabel style={{font=\footnotesize\mdseries, align=center, text width=2.2cm, text=gray!78!black}},
    grid=both,
    major grid style={{draw=gray!25}},
    minor grid style={{draw=gray!12}},
    axis line style={{draw=gray!45}},
    legend style={{
      at={{(axis description cs:0.5,-0.13)}},
      anchor=north,
      legend columns=4,
      font=\small\bfseries,
      draw=gray!35,
      fill=white
    }},
  ]
    \addplot+[line width=1.6pt, line join=round, color=studyApurple, mark=*, mark size=2.1pt, fill=studyApurple!45!white, fill opacity=0.05] coordinates {{{format_coords(plot_study_a, base_angle=-90.0, angle_offset=0.0, max_radius=float(y_max))}}};
    \addlegendentry{{Study A}}
    \addplot+[line width=1.6pt, line join=round, color=red!88!black, mark=*, mark size=1.5pt, fill=red!72!white, fill opacity=0.16] coordinates {{{format_coords(plot_overall, base_angle=-90.0, angle_offset=0.0, max_radius=float(y_max))}}};
    \addlegendentry{{Overall}}
    \addplot+[line width=1.6pt, line join=round, color=orange!96!black, mark=*, mark size=1.5pt, fill=orange!78!white, fill opacity=0.14] coordinates {{{format_coords(plot_study_b, base_angle=-90.0, angle_offset=0.0, max_radius=float(y_max))}}};
    \addlegendentry{{Study B}}
    \addplot+[line width=1.6pt, line join=round, color=teal!88!black, mark=*, mark size=1.5pt, fill=teal!76!white, fill opacity=0.14] coordinates {{{format_coords(plot_study_c, base_angle=-90.0, angle_offset=0.0, max_radius=float(y_max))}}};
    \addlegendentry{{Study C}}
    % Manual axis labels + leader lines for readability on a square-root radial scale.
    {label_commands}
  \end{{polaraxis}}
\end{{tikzpicture}}
"""


def build_summary_table(analysis: dict) -> str:
    severity = analysis["severity"]
    moderate_pct = severity.get("Moderate", {}).get("pct", 0.0)
    mild_pct = severity.get("Mild", {}).get("pct", 0.0)
    severe_pct = severity.get("Severe", {}).get("pct", 0.0)
    critical_pct = severity.get("Critical", {}).get("pct", 0.0)
    top_modality_name, top_modality_info = next(iter(analysis["therapeutic_modalities"].items()))
    return (
        r"\begin{center}" + "\n"
        r"{\scriptsize" + "\n"
        r"\renewcommand{\arraystretch}{1.15}" + "\n"
        r"\begin{tabular}{|l|r|}" + "\n"
        r"\hline" + "\n"
        r"\multicolumn{2}{|c|}{\textbf{Clinical Coverage Summary}}\\\hline" + "\n"
        + f"Total samples & {analysis['total_samples']:,}\\\\\\hline\n"
        + f"Study A / B / C & {analysis['per_study']['A']:,} / {analysis['per_study']['B']:,} / {analysis['per_study']['C']:,}\\\\\\hline\n"
        + f"DSM-5 categories & {len(analysis['diagnostic_categories'])}\\\\\\hline\n"
        + f"Unique conditions & {analysis['total_unique_conditions']}\\\\\\hline\n"
        + f"Moderate severity & {moderate_pct:.1f}\\%\\\\\\hline\n"
        + f"Mild severity & {mild_pct:.1f}\\%\\\\\\hline\n"
        + f"Severe severity & {severe_pct:.1f}\\%\\\\\\hline\n"
        + f"Critical severity & {critical_pct:.1f}\\%\\\\\\hline\n"
        + f"Top modality & {esc_tex(top_modality_name)} ({top_modality_info['pct_of_study_a']:.1f}\\%)\\\\\\hline\n"
        r"\end{tabular}" + "\n"
        r"}" + "\n"
        r"\end{center}"
    )


def build_wrapper_tex(figure_title: str, summary_table: str) -> str:
    return r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[paperwidth=12in,paperheight=9in,margin=0.35in]{geometry}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{setspace}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{polar}
\renewcommand{\familydefault}{\rmdefault}
\pagestyle{empty}
\begin{document}
\begin{center}
{\Large\bfseries __FIGURE_TITLE__}\\[1pt]
\end{center}
\vspace{-0.68cm}

\begin{center}
\resizebox{0.79\textwidth}{!}{\input{clinical_overview_radar}}
\end{center}
\vspace{1pt}

\noindent{\small Figure: Radar profile of the top diagnostic categories that rank highly in Overall and Study A/B/C, plotted with a square-root radial scale, top-side emphasis, and decimal ring labels.}
\end{document}
""".replace("__FIGURE_TITLE__", esc_tex(figure_title))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def main():
    figure_title = "Clinical Benchmark Overview (PGFPlots Radar)"
    figure_slug = slugify(figure_title)
    figure_dir = OUT / figure_slug
    figure_dir.mkdir(parents=True, exist_ok=True)

    with open(BASE / "distribution_analysis.json", encoding="utf-8") as f:
        analysis = json.load(f)
    radar_tex = build_radar_tex(analysis)
    summary_table = build_summary_table(analysis)
    wrapper_tex = build_wrapper_tex(figure_title, summary_table)

    radar_path = figure_dir / "clinical_overview_radar.tex"
    compile_path = figure_dir / "_compile_clinical_overview_radar.tex"
    radar_path.write_text(radar_tex, encoding="utf-8")
    compile_path.write_text(wrapper_tex, encoding="utf-8")

    readme = (
        "# PGFPlots Figures\n\n"
        "Each figure lives in its own folder named from the figure title.\n\n"
        "| Folder | Figure |\n"
        "|---|---|\n"
        f"| `{figure_slug}` | {figure_title} |\n"
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {radar_path}")
    print(f"Wrote {compile_path}")
    print(f"Wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
