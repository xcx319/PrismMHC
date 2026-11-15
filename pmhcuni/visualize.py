from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


CSS = """
:root {
  --bg: #f6f1e8;
  --bg2: #efe6d6;
  --ink: #1f2a30;
  --muted: #66757c;
  --card: rgba(255,255,255,0.82);
  --line: rgba(31,42,48,0.10);
  --accent: #bc5a45;
  --accent2: #246a73;
  --accent3: #d99b4d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(217,155,77,0.24), transparent 28%),
    radial-gradient(circle at top right, rgba(36,106,115,0.18), transparent 32%),
    linear-gradient(180deg, var(--bg), var(--bg2));
}
main { max-width: 1380px; margin: 0 auto; padding: 32px 20px 56px; }
header.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,255,255,0.58));
  border: 1px solid var(--line);
  border-radius: 28px;
  padding: 28px;
  backdrop-filter: blur(10px);
  box-shadow: 0 14px 50px rgba(31,42,48,0.08);
}
h1, h2, h3 { margin: 0 0 10px; font-family: "Iowan Old Style", "Palatino Linotype", serif; }
p, li, td, th, a, span, div { line-height: 1.5; }
p.lead { max-width: 900px; color: var(--muted); margin: 0; }
.grid { display: grid; gap: 18px; margin-top: 22px; }
.grid.cards { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 18px;
  box-shadow: 0 8px 28px rgba(31,42,48,0.06);
}
.metric { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; }
.kicker { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }
.section { margin-top: 26px; }
.section-head { display: flex; justify-content: space-between; align-items: end; gap: 12px; margin-bottom: 12px; }
.section-head p { margin: 0; color: var(--muted); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.94rem; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
th { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
a { color: var(--accent2); text-decoration: none; }
a:hover { text-decoration: underline; }
.badge {
  display: inline-block; border-radius: 999px; padding: 4px 10px;
  background: rgba(188,90,69,0.12); color: var(--accent); font-size: 0.82rem;
}
.stack { display: grid; gap: 18px; }
svg { width: 100%; height: auto; display: block; }
.nav { margin-top: 16px; display: flex; gap: 14px; flex-wrap: wrap; }
.footer { margin-top: 28px; color: var(--muted); font-size: 0.9rem; }
@media (max-width: 768px) {
  main { padding: 18px 12px 40px; }
  header.hero { padding: 20px; border-radius: 20px; }
  .card { border-radius: 18px; }
}
"""


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_") or "item"


def _ensure_int_labels(length: int, prefix: str = "") -> list[str]:
    return [f"{prefix}{i}" for i in range(1, length + 1)]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _interp_color(value: float, vmin: float, vmax: float) -> str:
    low = _hex_to_rgb("efe6d6")
    mid = _hex_to_rgb("d99b4d")
    high = _hex_to_rgb("bc5a45")
    if vmax <= vmin:
        rgb = mid
    else:
        t = (value - vmin) / (vmax - vmin)
        t = min(max(t, 0.0), 1.0)
        if t < 0.5:
            a = t / 0.5
            rgb = tuple(round(low[i] + a * (mid[i] - low[i])) for i in range(3))
        else:
            a = (t - 0.5) / 0.5
            rgb = tuple(round(mid[i] + a * (high[i] - mid[i])) for i in range(3))
    return "#%02x%02x%02x" % rgb


def _render_heatmap_svg(
    matrix: list[list[float]],
    title: str,
    row_tick_every: int | None = None,
    col_labels: list[str] | None = None,
) -> str:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return "<p>No heatmap data.</p>"

    flat = [value for row in matrix for value in row]
    vmin = min(flat)
    vmax = max(flat)
    width = 1180
    height = min(max(360, rows * 3 + 120), 880)
    left = 84
    top = 56
    plot_w = width - left - 24
    plot_h = height - top - 48
    cell_w = plot_w / cols
    cell_h = plot_h / rows
    row_tick_every = row_tick_every or max(1, rows // 18)
    col_labels = col_labels or _ensure_int_labels(cols)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<text x="24" y="30" font-size="22" font-family="IBM Plex Sans" fill="#1f2a30">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="12" fill="#fffdfa" stroke="rgba(31,42,48,0.12)"/>',
    ]

    for i, row in enumerate(matrix):
        y = top + i * cell_h
        if i % row_tick_every == 0 or i == rows - 1:
            label = str(i + 1)
            parts.append(
                f'<text x="{left - 12}" y="{y + cell_h * 0.7:.1f}" text-anchor="end" font-size="11" fill="#66757c">{label}</text>'
            )
        for j, value in enumerate(row):
            x = left + j * cell_w
            color = _interp_color(value, vmin, vmax)
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.3:.2f}" height="{cell_h + 0.3:.2f}" fill="{color}">'
                f'<title>MHC {i + 1}, peptide {j + 1}: {value:.6f}</title></rect>'
            )
    for j, label in enumerate(col_labels):
        x = left + (j + 0.5) * cell_w
        parts.append(
            f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" font-size="11" fill="#66757c">{escape(label)}</text>'
        )
    parts.append(f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" font-size="12" fill="#66757c">MHC position</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle" font-size="12" fill="#66757c">Peptide position</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _render_bar_svg(
    labels: list[str],
    values: list[float],
    title: str,
    color: str,
) -> str:
    if not labels or not values:
        return "<p>No bar data.</p>"
    width = max(720, len(labels) * 38 + 120)
    height = 320
    left = 52
    bottom = 48
    top = 40
    plot_w = width - left - 20
    plot_h = height - top - bottom
    vmax = max(values) or 1.0
    bar_w = max(10, plot_w / len(values) * 0.72)
    gap = plot_w / len(values)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}">',
        f'<text x="18" y="24" font-size="22" font-family="IBM Plex Sans" fill="#1f2a30">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - 12}" y2="{height - bottom}" stroke="rgba(31,42,48,0.18)"/>',
    ]
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        h = 0 if vmax == 0 else (value / vmax) * plot_h
        y = height - bottom - h
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="6" fill="{color}"><title>{escape(label)}: {value:.6f}</title></rect>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="11" fill="#66757c">{escape(label)}</text>'
        )
    parts.append('</svg>')
    return ''.join(parts)


def _render_profile_strip(values: list[float], color: str = "#246a73") -> str:
    if not values:
        return ""
    width = 220
    height = 26
    vmax = max(values) or 1.0
    cell_w = width / len(values)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    for idx, value in enumerate(values):
        alpha = 0.18 + 0.82 * (value / vmax if vmax else 0.0)
        parts.append(
            f'<rect x="{idx * cell_w:.2f}" y="0" width="{cell_w + 0.4:.2f}" height="{height}" fill="{color}" opacity="{alpha:.3f}"><title>Position {idx + 1}: {value:.6f}</title></rect>'
        )
    parts.append('</svg>')
    return ''.join(parts)


def _render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>No table data.</p>"
    head = ''.join(f'<th>{escape(label)}</th>' for _, label in columns)
    body_rows = []
    for row in rows:
        tds = ''.join(f'<td>{row.get(key, "")}</td>' for key, _ in columns)
        body_rows.append(f'<tr>{tds}</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def _page(title: str, subtitle: str, body: str, nav: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <main>
    <header class="hero">
      <div class="kicker">pMHC-uni Explainability</div>
      <h1>{escape(title)}</h1>
      <p class="lead">{escape(subtitle)}</p>
      <div class="nav">{nav}</div>
    </header>
    {body}
    <div class="footer">Generated from pMHC-uni explanation outputs.</div>
  </main>
</body>
</html>"""


def generate_visual_reports(explanation_dir: str | Path) -> dict[str, Path]:
    explanation_dir = Path(explanation_dir)
    viz_dir = explanation_dir / "visualizations"
    sample_dir = viz_dir / "samples"
    allele_dir = viz_dir / "alleles"
    sample_dir.mkdir(parents=True, exist_ok=True)
    allele_dir.mkdir(parents=True, exist_ok=True)

    explanations = json.loads((explanation_dir / "explanations.json").read_text(encoding="utf-8"))
    token_df = pd.read_csv(explanation_dir / "token_importance.csv")
    allele_token_df = pd.read_csv(explanation_dir / "allele_token_importance.csv")
    allele_matrix_df = pd.read_csv(explanation_dir / "allele_interactions.csv")
    sample_top_df = pd.read_csv(explanation_dir / "sample_top_interactions.csv")

    sample_links = []
    allele_counts = token_df[["sample_id", "mhc"]].drop_duplicates().groupby("mhc").size().to_dict()

    for sample in explanations:
        sample_id = int(sample["sample_id"])
        mhc = sample["mhc"]
        peptide = sample["peptide"]
        prob = float(sample["probability"])
        logit = float(sample["logit"])
        sample_name = f"sample_{sample_id:04d}_{_safe_name(mhc)}_{_safe_name(peptide)}.html"
        sample_path = sample_dir / sample_name

        peptide_rows = sample["peptide_importance"]
        peptide_labels = [f"P{row['position']}" for row in peptide_rows]
        peptide_values = [float(row["importance"]) for row in peptide_rows]

        mhc_rows = sorted(sample["mhc_importance"], key=lambda x: x["importance"], reverse=True)[:25]
        mhc_labels = [f"M{row['position']}" for row in mhc_rows]
        mhc_values = [float(row["importance"]) for row in mhc_rows]

        top_rows = []
        for row in sample["top_interactions"]:
            top_rows.append(
                {
                    "mhc_position": f"{row['mhc_position']} ({row['mhc_residue']})",
                    "peptide_position": f"{row['peptide_position']} ({row['peptide_residue']})",
                    "interaction_score": f"{row['interaction_score']:.6f}",
                }
            )

        _interaction_label = "MHC \u2194 Peptide interaction rollout"
        body = f"""
        <section class="section">
          <div class="grid cards">
            <div class="card"><div class="kicker">Allele</div><div class="metric">{escape(mhc)}</div></div>
            <div class="card"><div class="kicker">Peptide</div><div class="metric">{escape(peptide)}</div></div>
            <div class="card"><div class="kicker">Binding Probability</div><div class="metric">{prob:.4f}</div></div>
            <div class="card"><div class="kicker">Logit</div><div class="metric">{logit:.4f}</div></div>
          </div>
        </section>
        <section class="section stack">
          <div class="card">{_render_heatmap_svg(sample['interaction_matrix'], _interaction_label, col_labels=peptide_labels)}</div>
          <div class="grid cards">
            <div class="card">{_render_bar_svg(peptide_labels, peptide_values, 'Peptide position importance', '#246a73')}</div>
            <div class="card">{_render_bar_svg(mhc_labels, mhc_values, 'Top MHC positions by importance', '#bc5a45')}</div>
          </div>
          <div class="card">
            <div class="section-head"><h2>Top residue interactions</h2><span class="badge">Top {len(top_rows)}</span></div>
            {_render_table(top_rows, [('mhc_position', 'MHC Position'), ('peptide_position', 'Peptide Position'), ('interaction_score', 'Score')])}
          </div>
        </section>
        """
        nav = '<a href="../index.html">Overview</a><a href="../allele_comparison.html">Allele Comparison</a>'
        sample_path.write_text(_page(f"Sample {sample_id:04d}", f"{mhc} \u00d7 {peptide}", body, nav=nav), encoding="utf-8")
        sample_links.append({
            "sample_id": sample_id,
            "mhc": mhc,
            "peptide": peptide,
            "probability": f"{prob:.4f}",
            "page": f"samples/{sample_name}",
        })

    allele_links = []
    for mhc, subset in allele_matrix_df.groupby("mhc"):
        peptide_count = int(subset["peptide_position"].max())
        mhc_count = int(subset["mhc_position"].max())
        matrix = [[0.0 for _ in range(peptide_count)] for _ in range(mhc_count)]
        for row in subset.itertuples(index=False):
            matrix[int(row.mhc_position) - 1][int(row.peptide_position) - 1] = float(row.mean_interaction)

        allele_token_subset = allele_token_df[allele_token_df["mhc"] == mhc]
        pep_imp = allele_token_subset[allele_token_subset["segment"] == "peptide"].sort_values("position")
        pep_labels = [f"P{int(pos)}" for pos in pep_imp["position"].tolist()]
        pep_values = pep_imp["mean_importance"].tolist()

        mhc_imp = allele_token_subset[allele_token_subset["segment"] == "mhc"].sort_values("mean_importance", ascending=False).head(25)
        mhc_labels = [f"M{int(pos)}" for pos in mhc_imp["position"].tolist()]
        mhc_values = mhc_imp["mean_importance"].tolist()

        top_pairs = subset.sort_values("mean_interaction", ascending=False).head(20)
        table_rows = [
            {
                "mhc_position": int(row.mhc_position),
                "peptide_position": int(row.peptide_position),
                "mean_interaction": f"{float(row.mean_interaction):.6f}",
                "count": int(row.count),
            }
            for row in top_pairs.itertuples(index=False)
        ]

        allele_name = f"{_safe_name(mhc)}.html"
        allele_path = allele_dir / allele_name
        body = f"""
        <section class="section">
          <div class="grid cards">
            <div class="card"><div class="kicker">Allele</div><div class="metric">{escape(mhc)}</div></div>
            <div class="card"><div class="kicker">Samples</div><div class="metric">{int(allele_counts.get(mhc, 0))}</div></div>
            <div class="card"><div class="kicker">Peptide Positions</div><div class="metric">{peptide_count}</div></div>
            <div class="card"><div class="kicker">MHC Positions</div><div class="metric">{mhc_count}</div></div>
          </div>
        </section>
        <section class="section stack">
          <div class="card">{_render_heatmap_svg(matrix, 'Mean allele interaction map', col_labels=pep_labels)}</div>
          <div class="grid cards">
            <div class="card">{_render_bar_svg(pep_labels, pep_values, 'Mean peptide-position importance', '#246a73')}</div>
            <div class="card">{_render_bar_svg(mhc_labels, mhc_values, 'Top MHC positions by mean importance', '#bc5a45')}</div>
          </div>
          <div class="card">
            <div class="section-head"><h2>Top aggregated interactions</h2><span class="badge">Top {len(table_rows)}</span></div>
            {_render_table(table_rows, [('mhc_position', 'MHC Position'), ('peptide_position', 'Peptide Position'), ('mean_interaction', 'Mean Score'), ('count', 'Count')])}
          </div>
        </section>
        """
        nav = '<a href="../index.html">Overview</a><a href="../allele_comparison.html">Allele Comparison</a>'
        allele_path.write_text(_page(f"Allele {mhc}", "Aggregated positional preferences across explained samples.", body, nav=nav), encoding="utf-8")
        top_pep = pep_imp.sort_values("mean_importance", ascending=False).head(3)["position"].tolist()
        top_mhc = mhc_imp.head(3)["position"].tolist()
        allele_links.append({
            "mhc": mhc,
            "samples": int(allele_counts.get(mhc, 0)),
            "top_peptide_positions": ", ".join(f"P{int(x)}" for x in top_pep),
            "top_mhc_positions": ", ".join(f"M{int(x)}" for x in top_mhc),
            "profile_svg": _render_profile_strip(pep_values),
            "page": f"alleles/{allele_name}",
        })

    comparison_rows = []
    for row in allele_links:
        comparison_rows.append(
            {
                "allele": f'<a href="{row["page"]}">{escape(row["mhc"])}</a>',
                "samples": row["samples"],
                "dominant_peptide_positions": row["top_peptide_positions"],
                "dominant_mhc_positions": row["top_mhc_positions"],
                "peptide_profile": row["profile_svg"],
            }
        )
    comparison_table = _render_table(
        comparison_rows,
        [
            ("allele", "Allele"),
            ("samples", "Samples"),
            ("dominant_peptide_positions", "Top Peptide Positions"),
            ("dominant_mhc_positions", "Top MHC Positions"),
            ("peptide_profile", "Peptide Profile"),
        ],
    )
    comparison_body = f"""
    <section class="section">
      <div class="grid cards">
        <div class="card"><div class="kicker">Alleles Compared</div><div class="metric">{len(allele_links)}</div></div>
        <div class="card"><div class="kicker">Explained Samples</div><div class="metric">{len(sample_links)}</div></div>
      </div>
    </section>
    <section class="section"><div class="card">{comparison_table}</div></section>
    """
    comparison_path = viz_dir / "allele_comparison.html"
    comparison_path.write_text(_page("Allele Comparison", "Compare residue-position preferences across HLA alleles.", comparison_body, nav='<a href="index.html">Overview</a>'), encoding="utf-8")

    sample_rows = [
        {
            "sample_id": f'<a href="{row["page"]}">{row["sample_id"]}</a>',
            "mhc": escape(row["mhc"]),
            "peptide": escape(row["peptide"]),
            "probability": row["probability"],
        }
        for row in sample_links
    ]
    allele_rows = [
        {
            "allele": f'<a href="{row["page"]}">{escape(row["mhc"])}</a>',
            "samples": row["samples"],
            "top_peptide_positions": row["top_peptide_positions"],
            "top_mhc_positions": row["top_mhc_positions"],
        }
        for row in allele_links
    ]
    index_body = f"""
    <section class="section">
      <div class="grid cards">
        <div class="card"><div class="kicker">Explained Samples</div><div class="metric">{len(sample_links)}</div></div>
        <div class="card"><div class="kicker">Alleles</div><div class="metric">{len(allele_links)}</div></div>
        <div class="card"><div class="kicker">Artifacts</div><div class="metric">HTML + CSV</div></div>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><h2>Sample Pages</h2><p>Per-sample interaction heatmaps and residue saliency.</p></div>
      <div class="card">{_render_table(sample_rows, [('sample_id', 'Sample'), ('mhc', 'Allele'), ('peptide', 'Peptide'), ('probability', 'Probability')])}</div>
    </section>
    <section class="section">
      <div class="section-head"><h2>Allele Pages</h2><p>Aggregated interaction preferences and dominant positions.</p></div>
      <div class="card">{_render_table(allele_rows, [('allele', 'Allele'), ('samples', 'Samples'), ('top_peptide_positions', 'Top Peptide Positions'), ('top_mhc_positions', 'Top MHC Positions')])}</div>
    </section>
    """
    index_path = viz_dir / "index.html"
    index_path.write_text(_page("Explainability Dashboard", "Sample-level and allele-level visual reports generated from pMHC-uni outputs.", index_body, nav='<a href="allele_comparison.html">Allele Comparison</a>'), encoding="utf-8")

    return {
        "index": index_path,
        "allele_comparison": comparison_path,
        "sample_dir": sample_dir,
        "allele_dir": allele_dir,
    }
