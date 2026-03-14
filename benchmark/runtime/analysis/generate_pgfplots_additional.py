#!/usr/bin/env python3
"""
Generate additional paper-style PGFPlots figures (beyond the radar) from distribution_analysis.json.

Outputs in analysis/figures/pgfplots/<figure-slug>/
  - figure source .tex
  - _compile_*.tex wrapper
  - compiled .pdf (when pdflatex is run)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "figures" / "pgfplots"
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_ANALYSIS_JSON = BASE / "distribution_analysis.json"


def esc_tex(s: str) -> str:
    for ch in ["\\", "&", "%", "#", "_"]:
        s = s.replace(ch, f"\\{ch}")
    return s


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def normalise_condition_name(cond: str) -> str:
    acronyms = {"ocd", "ptsd", "adhd", "dbt", "cbt", "act", "emdr", "bpd"}
    words = cond.split()
    out = []
    for w in words:
        wl = w.lower()
        if wl in acronyms:
            out.append(w.upper())
        elif wl in {"with", "and", "in", "of", "on", "for", "the", "a"} and out:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional PGFPlots figures from an analysis JSON.")
    parser.add_argument("--analysis-json", type=Path, default=DEFAULT_ANALYSIS_JSON)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--title-prefix", default="")
    return parser.parse_args()


def write_figure(figure_title: str, body_tex: str, compile_name: str, caption: str, *, landscape: bool = False, margin: str = "1.2cm") -> tuple[Path, Path]:
    figure_slug = slugify(figure_title)
    figure_dir = OUT / figure_slug
    figure_dir.mkdir(parents=True, exist_ok=True)

    body_name = f"{compile_name}.tex"
    compile_tex_name = f"_compile_{compile_name}.tex"

    wrapper = (
        r"\documentclass[a4paper,10pt]{article}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        + (r"\usepackage[landscape,margin=" + margin + r"]{geometry}" + "\n" if landscape else r"\usepackage[margin=" + margin + r"]{geometry}" + "\n")
        + r"\usepackage{lmodern}" + "\n"
        + r"\usepackage{microtype}" + "\n"
        + r"\usepackage{tikz}" + "\n"
        + r"\usepackage{pgfplots}" + "\n"
        + r"\usepgfplotslibrary{groupplots}" + "\n"
        + r"\pgfplotsset{compat=1.18}" + "\n"
        + r"\pagestyle{empty}" + "\n"
        + r"\begin{document}" + "\n"
        + r"\begin{center}" + "\n"
        + rf"{{\Large\bfseries {esc_tex(figure_title)}}}\\[6pt]" + "\n"
        + r"\input{" + compile_name + r"}" + "\n"
        + r"\end{center}" + "\n"
        + rf"\noindent{{\small {esc_tex(caption)}}}" + "\n"
        + r"\end{document}" + "\n"
    )

    body_path = figure_dir / body_name
    compile_path = figure_dir / compile_tex_name
    body_path.write_text(body_tex, encoding="utf-8")
    compile_path.write_text(wrapper, encoding="utf-8")
    return body_path, compile_path


def build_fig1_category_panel(analysis: dict) -> tuple[str, str, str, str]:
    cats = analysis["diagnostic_categories"]
    cat_items = sorted(cats.items(), key=lambda kv: kv[1]["total"], reverse=True)
    cat_labels = ",".join("{" + esc_tex(k) + "}" for k, _ in cat_items)
    cat_coords = " ".join(f"({v['total']},{i})" for i, (_, v) in enumerate(cat_items))

    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.78\textwidth,
  height=0.56\textwidth,
  xbar,
  xmin=0,
  ytick=data,
  yticklabels={{{cat_labels}}},
  yticklabel style={{font=\scriptsize, text width=4.0cm, align=right}},
  xlabel={{Samples}},
  title={{DSM-5 Category Coverage}},
  nodes near coords,
  every node near coord/.append style={{font=\scriptsize, xshift=2pt}},
  grid=major,
  grid style={{draw=gray!20}},
  axis line style={{draw=gray!45}},
]
\addplot+[draw=none, fill=blue!62!black] coordinates {{{cat_coords}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return (
        "Diagnostic Category Coverage (PGFPlots)",
        body,
        "fig1_diagnostic_category_coverage",
        "Standalone panel from Figure 1: DSM-5 category coverage.",
    )


def build_fig1_top_conditions_panel(analysis: dict) -> tuple[str, str, str, str]:
    conds = analysis["specific_conditions"]
    top_conds = list(conds.items())[:18]
    cond_labels = ",".join("{" + esc_tex(normalise_condition_name(k)) + "}" for k, _ in reversed(top_conds))
    cond_coords = " ".join(f"({v['total']},{i})" for i, (_, v) in enumerate(reversed(top_conds)))

    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.72\textwidth,
  height=0.72\textheight,
  xbar,
  xmin=0,
  ytick=data,
  yticklabels={{{cond_labels}}},
  yticklabel style={{font=\scriptsize, text width=6.2cm, align=right}},
  xlabel={{Samples}},
  title={{Top 18 Conditions}},
  nodes near coords,
  every node near coord/.append style={{font=\tiny, xshift=2pt}},
  grid=major,
  grid style={{draw=gray!20}},
  axis line style={{draw=gray!45}},
]
\addplot+[draw=none, fill=orange!78!black] coordinates {{{cond_coords}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return (
        "Top Conditions (PGFPlots)",
        body,
        "fig1_top_conditions",
        "Standalone panel from Figure 1: top 18 clinical conditions by sample count.",
    )


def build_fig1_severity_panel(analysis: dict) -> tuple[str, str, str, str]:
    sev = analysis["severity"]
    sev_order = [s for s in ["Critical", "Severe", "Moderate", "Mild"] if s in sev]
    sev_cols = {
        "Critical": "severityCritical",
        "Severe": "severitySevere",
        "Moderate": "severityModerate",
        "Mild": "severityMild",
    }

    total = sum(sev[s]["total"] for s in sev_order)
    angle = 90.0
    arc_commands: list[str] = []
    label_commands: list[str] = []
    for level in sev_order:
        value = sev[level]["total"]
        pct = sev[level]["pct"]
        sweep = 360.0 * value / total if total else 0.0
        end_angle = angle - sweep
        mid_angle = angle - (sweep / 2.0)
        arc_commands.append(
            rf"\path[fill={sev_cols[level]}, draw=white, line width=1.1pt] "
            rf"(0,0) -- ({angle:.4f}:2.35) arc ({angle:.4f}:{end_angle:.4f}:2.35) -- "
            rf"({end_angle:.4f}:1.35) arc ({end_angle:.4f}:{angle:.4f}:1.35) -- cycle;"
        )
        if level == "Critical":
            label_anchor = "east"
            label_radius = 3.30
        else:
            label_anchor = "west" if abs(mid_angle) <= 90 else "east"
            label_radius = 3.05
        label_commands.append(
            rf"\node[font=\small\bfseries, align=center, text=black, anchor={label_anchor}] "
            rf"at ({mid_angle:.4f}:{label_radius:.2f}) {{{esc_tex(level)}\\{value:,} ({pct:.1f}\%)}};"
        )
        angle = end_angle

    body = rf"""\begin{{tikzpicture}}
\definecolor{{severityCritical}}{{HTML}}{{9B2226}}
\definecolor{{severitySevere}}{{HTML}}{{E10600}}
\definecolor{{severityModerate}}{{HTML}}{{F07B83}}
\definecolor{{severityMild}}{{HTML}}{{F6C1C4}}
{chr(10).join(arc_commands)}
\fill[white] (0,0) circle (1.15);
\draw[white, line width=1.0pt] (0,0) circle (1.15);
{chr(10).join(label_commands)}
\node[font=\large\bfseries] at (0,3.7) {{Severity Mix}};
\node[font=\normalsize\bfseries, align=center] at (0,0) {{Severity\\N={total:,}}};
\end{{tikzpicture}}
"""
    return (
        "Severity Mix (PGFPlots)",
        body,
        "fig1_severity_mix",
        "Standalone panel from Figure 1: severity mix shown as an annular severity chart.",
    )


def build_fig1_standalone_specs(analysis: dict) -> list[tuple[str, str, str, str]]:
    return [
        build_fig1_category_panel(analysis),
        build_fig1_top_conditions_panel(analysis),
        build_fig1_severity_panel(analysis),
    ]


def build_output_specs(analysis: dict) -> list[tuple[str, str, str, str]]:
    omit = set(analysis.get("omit_figures", []))
    specs: list[tuple[str, str, str, str]] = []
    fig_map = {
        "fig2_per_study_breakdown": build_fig2,
        "fig3_therapeutic_modalities": build_fig3,
        "fig4_all_conditions_by_category": build_fig4,
    }
    for name, builder in fig_map.items():
        if name == "fig3_therapeutic_modalities" and not analysis.get("therapeutic_modalities"):
            continue
        if name not in omit:
            specs.append(builder(analysis))
    for spec in build_fig1_standalone_specs(analysis):
        if spec[2] not in omit:
            specs.append(spec)
    return specs


def build_fig1(analysis: dict) -> tuple[str, str, str, bool]:
    cats = analysis["diagnostic_categories"]
    cat_items = sorted(cats.items(), key=lambda kv: kv[1]["total"], reverse=True)

    conds = analysis["specific_conditions"]
    top_conds = list(conds.items())[:18]

    sev = analysis["severity"]
    sev_order = [s for s in ["Critical", "Severe", "Moderate", "Mild"] if s in sev]
    sev_cols = {
        "Critical": "#9B2226",
        "Severe": "#E10600",
        "Moderate": "#F07B83",
        "Mild": "#F6C1C4",
    }

    cat_labels = ",".join("{" + esc_tex(k) + "}" for k, _ in cat_items)
    cat_coords = " ".join(f"({v['total']},{i})" for i, (_, v) in enumerate(cat_items))

    cond_labels = ",".join("{" + esc_tex(normalise_condition_name(k)) + "}" for k, _ in reversed(top_conds))
    cond_coords = " ".join(f"({v['total']},{i})" for i, (_, v) in enumerate(reversed(top_conds)))

    sev_coords = " ".join(f"({esc_tex(s)},{sev[s]['pct']:.2f})" for s in sev_order)
    sev_col_list = ",".join(sev_cols[s] for s in sev_order)

    body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=3 by 1, horizontal sep=1.3cm}},
  width=0.30\textwidth,
  height=0.56\textwidth,
  grid=major,
  grid style={{draw=gray!20}},
  axis line style={{draw=gray!45}},
  tick label style={{font=\footnotesize}},
  label style={{font=\small}}
]

\nextgroupplot[
  title={{A. DSM-5 Category Coverage}},
  xbar,
  xmin=0,
  ytick=data,
  yticklabels={{{cat_labels}}},
  yticklabel style={{font=\scriptsize, text width=3.5cm, align=right}},
  xlabel={{Samples}},
  nodes near coords,
  every node near coord/.append style={{font=\tiny, xshift=2pt}},
]
\addplot+[draw=none, fill=blue!62!black] coordinates {{{cat_coords}}};

\nextgroupplot[
  title={{B. Top 18 Conditions}},
  xbar,
  xmin=0,
  ytick=data,
  yticklabels={{{cond_labels}}},
  yticklabel style={{font=\scriptsize, text width=4.4cm, align=right}},
  xlabel={{Samples}},
  nodes near coords,
  every node near coord/.append style={{font=\tiny, xshift=2pt}},
]
\addplot+[draw=none, fill=orange!78!black] coordinates {{{cond_coords}}};

\nextgroupplot[
  title={{C. Severity Mix (\%)}},
  ybar,
  ymin=0,
  ymax=52,
  symbolic x coords={{{','.join(esc_tex(s) for s in sev_order)}}},
  xtick=data,
  xlabel={{Severity}},
  ylabel={{Percent}},
  nodes near coords,
  every node near coord/.append style={{font=\scriptsize}},
  x tick label style={{rotate=18, anchor=east, font=\scriptsize}}
]
\addplot+[draw=none, fill=red!40!white, point meta=explicit symbolic] coordinates {{{sev_coords}}};
\addplot+[draw=none, fill=none] coordinates {{}};
\end{{groupplot}}
\end{{tikzpicture}}
"""

    return (
        "Clinical Coverage Profile (PGFPlots)",
        body,
        "fig1_clinical_coverage_profile",
        "Figure 1: DSM-5 categories, top conditions, and severity mix.",
    )


def build_fig2(analysis: dict) -> tuple[str, str, str, bool]:
    cats = analysis["diagnostic_categories"]
    cat_names = list(cats.keys())
    cat_labels = ",".join("{" + esc_tex(c) + "}" for c in cat_names)
    study_labels = analysis.get("study_labels", {})
    study_keys = list(analysis.get("per_study", {}).keys())
    palette = ["purple!78!black", "red!78!black", "orange!90!black", "teal!80!black", "blue!65!black", "gray!65"]
    legend_columns = min(3, max(1, len(study_keys)))
    plot_lines = []
    for index, study_key in enumerate(study_keys):
        coords = " ".join(
            f"({cats[c]['per_study'].get(study_key, 0)},{i})" for i, c in enumerate(cat_names)
        )
        fill = palette[index % len(palette)]
        plot_lines.append(rf"\addplot+[draw=none, fill={fill}] coordinates {{{coords}}};")
        plot_lines.append(rf"\addlegendentry{{{esc_tex(study_labels.get(study_key, study_key))}}}")

    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.88\textwidth,
  height=0.55\textwidth,
  xbar stacked,
  xmin=0,
  ytick=data,
  yticklabels={{{cat_labels}}},
  y dir=reverse,
  yticklabel style={{font=\scriptsize, text width=4.4cm, align=right}},
  xlabel={{Samples}},
  legend style={{at={{(0.5,1.01)}}, anchor=south, legend columns={legend_columns}, draw=none, font=\small}},
  grid=major,
  grid style={{draw=gray!20}},
  axis line style={{draw=gray!45}},
]
{chr(10).join(plot_lines)}
\end{{axis}}
\end{{tikzpicture}}
"""
    return (
        "Diagnostic Category Split by Study (PGFPlots)",
        body,
        "fig2_per_study_breakdown",
        "Figure 2: Stacked per-study contribution for each DSM-5 category.",
    )


def build_fig3(analysis: dict) -> tuple[str, str, str, bool]:
    mods = list(analysis["therapeutic_modalities"].items())
    mod_labels = ",".join("{" + esc_tex(k) + "}" for k, _ in mods)
    coords = " ".join(f"({v['mentions']},{i})" for i, (_, v) in enumerate(mods))

    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.84\textwidth,
  height=0.52\textwidth,
  xbar,
  xmin=0,
  ytick=data,
  yticklabels={{{mod_labels}}},
  y dir=reverse,
  yticklabel style={{font=\scriptsize, text width=4.6cm, align=right}},
  xlabel={{Mentions in Study A Gold Reasoning}},
  title={{Therapeutic Modality Mentions (PGFPlots)}},
  nodes near coords,
  every node near coord/.append style={{font=\scriptsize, xshift=2pt}},
  grid=major,
  grid style={{draw=gray!20}},
  axis line style={{draw=gray!45}},
]
\addplot+[draw=none, fill=teal!68!black] coordinates {{{coords}}};
\end{{axis}}
\end{{tikzpicture}}
"""
    return (
        "Therapeutic Modality Mentions (PGFPlots)",
        body,
        "fig3_therapeutic_modalities",
        "Figure 3: Modality references extracted from Study A gold reasoning.",
    )


def build_fig4(analysis: dict) -> tuple[str, str, str, bool]:
    conds = analysis["specific_conditions"]
    cats = list(analysis["diagnostic_categories"].keys())

    cat_cols = {
        "Trauma & Stressor-Related": "blue!70!black",
        "Anxiety Disorders": "blue!62!black",
        "Mood Disorders": "blue!56!black",
        "OCD & Related": "blue!50!black",
        "Somatic & Health-Related": "cyan!55!black",
        "Eating Disorders": "cyan!48!black",
        "Neurodevelopmental": "teal!56!black",
        "Psychotic Spectrum": "teal!62!black",
        "Personality Disorders": "teal!50!black",
        "Self-Harm & Suicidality": "red!55!black",
        "Substance Use Disorders": "orange!75!black",
        "Sleep-Wake Disorders": "purple!55!black",
        "Other": "gray!60",
    }

    grouped: dict[str, list[tuple[str, int]]] = {c: [] for c in cats}
    for cond, info in conds.items():
        grouped[info["category"]].append((normalise_condition_name(cond), info["total"]))
    for c in grouped:
        grouped[c].sort(key=lambda t: t[1], reverse=True)

    rows: list[tuple[str, int, str]] = []
    separators: list[tuple[int, str]] = []
    pos = 0
    for c in cats:
        items = grouped.get(c, [])
        if not items:
            continue
        separators.append((pos, c))
        for name, total in items:
            rows.append((name, total, c))
            pos += 1
        pos += 0.5

    y_positions: list[float] = []
    current_y = 0.0
    group_idx = 0
    next_sep = separators[group_idx + 1][0] if group_idx + 1 < len(separators) else float("inf")
    for i in range(len(rows)):
        y_positions.append(current_y)
        current_y += 1.0
        if i + 1 == next_sep:
            current_y += 0.8
            group_idx += 1
            next_sep = separators[group_idx + 1][0] if group_idx + 1 < len(separators) else float("inf")

    rows.reverse()
    y_positions.reverse()

    ylabels = ",".join("{" + esc_tex(name) + "}" for name, _, _ in rows)
    yticks = ",".join(f"{y:.2f}" for y in y_positions)
    max_val = max((total for _, total, _ in rows), default=1)
    x_max = max(10, int(max_val * 1.14))
    y_min = min(y_positions, default=0.0) - 0.8
    y_max = max(y_positions, default=1.0) + 0.8

    bar_height = 0.82
    half_bar = bar_height / 2.0
    bar_commands: list[str] = []
    value_commands: list[str] = []
    for (name, total, cat), y in zip(rows, y_positions):
        fill = cat_cols.get(cat, "gray!55")
        bar_commands.append(
            rf"\path[fill={fill}, draw=white, line width=0.35pt] "
            rf"(axis cs:0,{y - half_bar:.2f}) rectangle (axis cs:{total},{y + half_bar:.2f});"
        )
        if total >= 100:
            value_commands.append(
                rf"\node[anchor=west, font=\tiny, text=black] at (axis cs:{total + max_val * 0.008:.2f},{y:.2f}) {{{total}}};"
            )

    present_categories = [cat for cat in cats if grouped.get(cat)]
    category_band_commands: list[str] = []
    for idx, cat in enumerate(present_categories):
        cat_y = [y for (_, _, row_cat), y in zip(rows, y_positions) if row_cat == cat]
        if not cat_y:
            continue
        top_y = max(cat_y)
        bottom_y = min(cat_y)
        category_band_commands.append(
            rf"\node[anchor=west, font=\scriptsize\bfseries, text={cat_cols.get(cat, 'gray!55')}, fill=white, inner xsep=1pt] "
            rf"at (axis cs:{x_max * 0.01:.2f},{top_y + 0.52:.2f}) {{{esc_tex(cat)}}};"
        )
        if idx < len(present_categories) - 1:
            category_band_commands.append(
                rf"\draw[gray!28, line width=0.3pt] (axis cs:0,{bottom_y - 0.52:.2f}) -- (axis cs:{x_max},{bottom_y - 0.52:.2f});"
            )

    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth,
  height=0.88\textheight,
  xmin=0,
  xmax={x_max},
  ymin={y_min:.2f},
  ymax={y_max:.2f},
  xticklabel style={{font=\scriptsize}},
  scaled x ticks=false,
  /pgf/number format/1000 sep={{,}},
  ytick={{{yticks}}},
  yticklabels={{{ylabels}}},
  yticklabel style={{font=\tiny, text width=6.4cm, align=right}},
  xlabel={{Samples}},
  grid=major,
  grid style={{draw=gray!16}},
  axis line style={{draw=gray!45}},
  enlarge y limits=false,
  clip=false,
]
{chr(10).join(bar_commands)}
{chr(10).join(value_commands)}
{chr(10).join(category_band_commands)}
\end{{axis}}
\end{{tikzpicture}}
"""

    return (
        "All Clinical Conditions by Category (PGFPlots)",
        body,
        "fig4_all_conditions_by_category",
        "Figure 4: All conditions coloured by DSM-5 category.",
    )


def main() -> None:
    global OUT
    args = parse_args()
    OUT = args.output_root
    OUT.mkdir(parents=True, exist_ok=True)

    with open(args.analysis_json, encoding="utf-8") as f:
        analysis = json.load(f)

    specs = build_output_specs(analysis)

    rows = []
    for title, body, compile_name, caption in specs:
        if args.title_prefix:
            title = f"{args.title_prefix} — {title}"
        landscape = compile_name in {"fig4_all_conditions_by_category", "fig1_top_conditions"}
        body_path, compile_path = write_figure(
            title,
            body,
            compile_name,
            caption,
            landscape=landscape,
            margin="1.0cm" if landscape else "1.2cm",
        )
        rows.append((body_path.parent.name, title, compile_path.name))
        print(f"Wrote {body_path}")
        print(f"Wrote {compile_path}")

    existing_entries: dict[str, tuple[str, str]] = {}
    readme_path = OUT / "README.md"
    if readme_path.exists():
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| `"):
                continue
            parts = [part.strip() for part in line.strip().split("|")[1:-1]]
            if len(parts) != 3:
                continue
            folder = parts[0].strip("`")
            figure = parts[1]
            wrapper = parts[2]
            existing_entries[folder] = (figure, wrapper)

    for folder, title, wrapper in rows:
        existing_entries[folder] = (title, f"`{wrapper}`")

    # Preserve existing radar folder entry if present.
    for radar_folder in [
        "clinical_benchmark_overview_pgfplots_radar",
        "controllability_split_overview_pgfplots_radar",
        "controllability_splits_small_scale_pgfplots_radar",
        "controllability_splits_scaled_large_resolved_pgfplots_radar",
    ]:
        if (OUT / radar_folder).exists() and radar_folder not in existing_entries:
            existing_entries[radar_folder] = (
                radar_folder.replace("_", " ").title(),
                "`_compile_clinical_overview_radar.tex`",
            )

    readme_lines = [
        "# PGFPlots Figures",
        "",
        "Each figure lives in its own folder named from the figure title.",
        "",
        "| Folder | Figure | Compile Wrapper |",
        "|---|---|---|",
    ]
    for folder in sorted(existing_entries):
        title, wrapper = existing_entries[folder]
        readme_lines.append(f"| `{folder}` | {title} | {wrapper} |")

    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
