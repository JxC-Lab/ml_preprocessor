"""
reporter.py — Generates a rich HTML report comparing a DataFrame
before and after preprocessing.

Usage:
    from ml_preprocessor.reporter import generate_report
    generate_report(df_before, df_after, output_path="report.html")

Or from CLI:
    python -m ml_preprocessor run --config cfg.yaml --input data.csv --output out.csv --report report.html
"""

from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col_stats(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in df.columns:
        s = df[col]
        n = len(s)
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / n * 100, 1) if n else 0
        dtype = str(s.dtype)
        n_unique = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            rows.append({
                "column": col,
                "dtype": dtype,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_unique": n_unique,
                "mean": round(float(s.mean()), 4) if n_missing < n else None,
                "std":  round(float(s.std()),  4) if n_missing < n else None,
                "min":  round(float(s.min()),  4) if n_missing < n else None,
                "p25":  round(float(s.quantile(0.25)), 4) if n_missing < n else None,
                "p50":  round(float(s.quantile(0.50)), 4) if n_missing < n else None,
                "p75":  round(float(s.quantile(0.75)), 4) if n_missing < n else None,
                "max":  round(float(s.max()),  4) if n_missing < n else None,
                "kind": "numeric",
                "top_values": [],
            })
        else:
            vc = s.value_counts(dropna=True).head(5)
            rows.append({
                "column": col,
                "dtype": dtype,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_unique": n_unique,
                "mean": None, "std": None, "min": None,
                "p25": None, "p50": None, "p75": None, "max": None,
                "kind": "categorical",
                "top_values": [{"label": str(k), "count": int(v)} for k, v in vc.items()],
            })
    return rows


def _histogram_data(s: pd.Series, bins: int = 20) -> list[dict]:
    """Return histogram bin data for a numeric series."""
    clean = s.dropna()
    if len(clean) == 0:
        return []
    counts, edges = np.histogram(clean, bins=bins)
    return [
        {"x": round(float(edges[i]), 3), "y": int(counts[i])}
        for i in range(len(counts))
    ]


def _build_histograms(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """Build histogram data for numeric columns that exist in both."""
    histograms = {}
    common_numeric = [
        c for c in df_before.columns
        if c in df_after.columns
        and pd.api.types.is_numeric_dtype(df_before[c])
        and pd.api.types.is_numeric_dtype(df_after[c])
    ]
    for col in common_numeric:
        histograms[col] = {
            "before": _histogram_data(df_before[col]),
            "after":  _histogram_data(df_after[col]),
        }
    return histograms


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Preprocessing Report — {title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  :root {{
    --bg:        #0d0f14;
    --surface:   #161a22;
    --border:    #252c3b;
    --text:      #cdd6f0;
    --muted:     #5a6380;
    --accent:    #4f9cf9;
    --accent2:   #7de8b0;
    --accent3:   #f9a74f;
    --danger:    #f97c7c;
    --radius:    6px;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ font-size: 14px; scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    line-height: 1.6;
  }}

  /* ── Layout ── */
  .header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 2rem 3rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
  }}
  .header h1 {{
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.02em;
  }}
  .header h1 span {{ color: var(--accent); }}
  .header .meta {{ font-size: 0.78rem; color: var(--muted); font-family: var(--mono); }}

  .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem 3rem; }}

  /* ── Summary cards ── */
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
  }}
  .card .label {{
    font-size: 0.72rem;
    font-family: var(--mono);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }}
  .card .value {{
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
    color: #fff;
  }}
  .card .sub {{ font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }}
  .card.accent-blue  .value {{ color: var(--accent); }}
  .card.accent-green .value {{ color: var(--accent2); }}
  .card.accent-amber .value {{ color: var(--accent3); }}
  .card.accent-red   .value {{ color: var(--danger); }}

  /* ── Section headers ── */
  .section-title {{
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin: 2.5rem 0 1.2rem;
  }}
  .section-title span {{ color: var(--accent); }}

  /* ── Column table ── */
  .col-table {{ width: 100%; border-collapse: collapse; }}
  .col-table th {{
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    text-align: left;
    padding: 0.5rem 0.8rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
  }}
  .col-table td {{
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
    white-space: nowrap;
  }}
  .col-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .badge {{
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.65rem;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-weight: 600;
  }}
  .badge-num  {{ background: rgba(79,156,249,0.15); color: var(--accent); }}
  .badge-cat  {{ background: rgba(125,232,176,0.15); color: var(--accent2); }}
  .badge-new  {{ background: rgba(249,167,79,0.15);  color: var(--accent3); }}
  .badge-drop {{ background: rgba(249,124,124,0.15); color: var(--danger); }}
  .missing-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 120px;
  }}
  .missing-bar {{
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }}
  .missing-bar-fill {{
    height: 100%;
    background: var(--danger);
    border-radius: 2px;
  }}
  .missing-bar-fill.ok {{ background: var(--accent2); }}
  .mono {{ font-family: var(--mono); }}

  /* ── Histograms ── */
  .hist-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }}
  .hist-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
  }}
  .hist-card .col-name {{
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .hist-card .legend {{ display: flex; gap: 1rem; font-size: 0.68rem; color: var(--muted); }}
  .hist-card .legend span {{ display: flex; align-items: center; gap: 0.3rem; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  canvas {{ width: 100% !important; }}

  /* ── Diff section ── */
  .diff-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
  }}
  .diff-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
  }}
  .diff-box h3 {{
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
  }}
  .col-chip {{
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.72rem;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    margin: 0.2rem;
    background: var(--border);
    color: var(--text);
  }}
  .col-chip.added   {{ background: rgba(125,232,176,0.12); color: var(--accent2); border: 1px solid rgba(125,232,176,0.25); }}
  .col-chip.removed {{ background: rgba(249,124,124,0.10); color: var(--danger);  border: 1px solid rgba(249,124,124,0.2); }}

  /* ── Tabs ── */
  .tab-bar {{ display: flex; gap: 0.2rem; margin-bottom: 1.2rem; }}
  .tab-btn {{
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.4rem 1rem;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
  }}
  .tab-btn.active, .tab-btn:hover {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  footer {{
    text-align: center;
    font-size: 0.72rem;
    color: var(--muted);
    font-family: var(--mono);
    padding: 2rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
  }}
</style>
</head>
<body>

<header class="header">
  <div>
    <h1>ml_preprocessor <span>//</span> Preprocessing Report</h1>
    <div class="meta">Generated on {generated_at} &nbsp;|&nbsp; {title}</div>
  </div>
  <div class="meta" style="text-align:right">
    Input shape: <b style="color:#fff">{rows_before} × {cols_before}</b> &nbsp;→&nbsp;
    Output shape: <b style="color:#fff">{rows_after} × {cols_after}</b>
  </div>
</header>

<div class="container">

  <!-- Summary cards -->
  <div class="summary-grid">
    <div class="card accent-blue">
      <div class="label">Rows</div>
      <div class="value">{rows_before}</div>
      <div class="sub">→ {rows_after} after preprocessing</div>
    </div>
    <div class="card accent-blue">
      <div class="label">Columns Before</div>
      <div class="value">{cols_before}</div>
      <div class="sub">{n_numeric_before} numeric · {n_cat_before} categorical</div>
    </div>
    <div class="card accent-green">
      <div class="label">Columns After</div>
      <div class="value">{cols_after}</div>
      <div class="sub">+{n_added} added · -{n_removed} removed</div>
    </div>
    <div class="card accent-amber">
      <div class="label">Missing (Before)</div>
      <div class="value">{pct_missing_before}%</div>
      <div class="sub">{n_missing_before} total missing cells</div>
    </div>
    <div class="card accent-green">
      <div class="label">Missing (After)</div>
      <div class="value">{pct_missing_after}%</div>
      <div class="sub">{n_missing_after} total missing cells</div>
    </div>
    <div class="card">
      <div class="label">Memory</div>
      <div class="value mono" style="font-size:1.2rem">{mem_before}</div>
      <div class="sub">→ {mem_after} after</div>
    </div>
  </div>

  <!-- Column diff -->
  <div class="section-title"><span>01</span> — Column Changes</div>
  <div class="diff-grid">
    <div class="diff-box">
      <h3>▲ Added columns ({n_added})</h3>
      <div id="added-cols">{added_cols_html}</div>
    </div>
    <div class="diff-box">
      <h3>▼ Removed columns ({n_removed})</h3>
      <div id="removed-cols">{removed_cols_html}</div>
    </div>
  </div>

  <!-- Tabs: Before / After -->
  <div class="section-title"><span>02</span> — Column Statistics</div>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="showTab('before')">Before</button>
    <button class="tab-btn" onclick="showTab('after')">After</button>
  </div>

  <div class="tab-panel active" id="tab-before">
    <div style="overflow-x:auto">
      <table class="col-table">
        <thead><tr>
          <th>Column</th><th>Type</th><th>Missing</th>
          <th>Unique</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th>
        </tr></thead>
        <tbody>{stats_before_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="tab-panel" id="tab-after">
    <div style="overflow-x:auto">
      <table class="col-table">
        <thead><tr>
          <th>Column</th><th>Type</th><th>Missing</th>
          <th>Unique</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th>
        </tr></thead>
        <tbody>{stats_after_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Histograms -->
  <div class="section-title"><span>03</span> — Distributions (Before vs After)</div>
  <div class="hist-grid" id="hist-grid"></div>

</div>

<footer>ml_preprocessor v1.0.0 &nbsp;·&nbsp; MIT License &nbsp;·&nbsp; {generated_at}</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const HIST_DATA = {hist_data_json};

function showTab(name) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

function buildHistograms() {{
  const grid = document.getElementById('hist-grid');
  const cols = Object.keys(HIST_DATA);
  if (cols.length === 0) {{
    grid.innerHTML = '<p style="color:var(--muted);font-size:.8rem">No common numeric columns found.</p>';
    return;
  }}
  cols.forEach(col => {{
    const d = HIST_DATA[col];
    const card = document.createElement('div');
    card.className = 'hist-card';
    card.innerHTML = `
      <div class="col-name">
        <span>${{col}}</span>
        <span class="legend">
          <span><span class="dot" style="background:#4f9cf9"></span>before</span>
          <span><span class="dot" style="background:#7de8b0"></span>after</span>
        </span>
      </div>
      <canvas id="hist-${{col.replace(/[^a-zA-Z0-9]/g,'_')}}" height="120"></canvas>
    `;
    grid.appendChild(card);

    const canvasId = 'hist-' + col.replace(/[^a-zA-Z0-9]/g,'_');
    const ctx = document.getElementById(canvasId).getContext('2d');

    const beforeLabels = d.before.map(b => b.x.toFixed(2));
    const beforeValues = d.before.map(b => b.y);
    const afterLabels  = d.after.map(b => b.x.toFixed(2));
    const afterValues  = d.after.map(b => b.y);

    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: beforeLabels.length >= afterLabels.length ? beforeLabels : afterLabels,
        datasets: [
          {{
            label: 'Before',
            data: beforeValues,
            backgroundColor: 'rgba(79,156,249,0.35)',
            borderColor: 'rgba(79,156,249,0.8)',
            borderWidth: 1,
            borderRadius: 2,
          }},
          {{
            label: 'After',
            data: afterValues,
            backgroundColor: 'rgba(125,232,176,0.25)',
            borderColor: 'rgba(125,232,176,0.7)',
            borderWidth: 1,
            borderRadius: 2,
          }}
        ]
      }},
      options: {{
        responsive: true,
        animation: {{ duration: 600 }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            backgroundColor: '#161a22',
            borderColor: '#252c3b',
            borderWidth: 1,
            titleColor: '#fff',
            bodyColor: '#cdd6f0',
            titleFont: {{ family: 'IBM Plex Mono', size: 11 }},
            bodyFont:  {{ family: 'IBM Plex Mono', size: 11 }},
          }}
        }},
        scales: {{
          x: {{
            ticks: {{ color: '#5a6380', font: {{ family: 'IBM Plex Mono', size: 9 }}, maxRotation: 45 }},
            grid:  {{ color: '#1e2435' }},
          }},
          y: {{
            ticks: {{ color: '#5a6380', font: {{ family: 'IBM Plex Mono', size: 9 }} }},
            grid:  {{ color: '#1e2435' }},
          }}
        }}
      }}
    }});
  }});
}}

buildHistograms();
</script>
</body>
</html>
"""


def _fmt_memory(df: pd.DataFrame) -> str:
    mem = df.memory_usage(deep=True).sum()
    if mem < 1024:
        return f"{mem} B"
    elif mem < 1024 ** 2:
        return f"{mem / 1024:.1f} KB"
    else:
        return f"{mem / 1024 ** 2:.1f} MB"


def _stats_row(s: dict) -> str:
    badge = "badge-num" if s["kind"] == "numeric" else "badge-cat"
    badge_label = "num" if s["kind"] == "numeric" else "cat"

    pct = s["pct_missing"]
    fill_cls = "ok" if pct == 0 else ""
    missing_cell = f"""
      <div class='missing-bar-wrap'>
        <div class='missing-bar'><div class='missing-bar-fill {fill_cls}' style='width:{pct}%'></div></div>
        <span class='mono' style='font-size:.72rem;color:{"var(--accent2)" if pct==0 else "var(--danger)"}'>{pct}%</span>
      </div>"""

    def fmt(v):
        if v is None:
            return "<span style='color:var(--muted)'>—</span>"
        if isinstance(v, float) and math.isnan(v):
            return "<span style='color:var(--muted)'>—</span>"
        return f"<span class='mono'>{v}</span>"

    return f"""<tr>
      <td><span class='mono' style='color:#fff'>{s["column"]}</span></td>
      <td><span class='badge {badge}'>{badge_label}</span> <span style='font-size:.7rem;color:var(--muted)'>{s["dtype"]}</span></td>
      <td>{missing_cell}</td>
      <td><span class='mono'>{s["n_unique"]}</span></td>
      <td>{fmt(s["mean"])}</td>
      <td>{fmt(s["std"])}</td>
      <td>{fmt(s["min"])}</td>
      <td>{fmt(s["max"])}</td>
    </tr>"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    df_before: pd.DataFrame,
    df_after:  pd.DataFrame,
    output_path: str | Path = "preprocessing_report.html",
    title: str = "Dataset",
) -> Path:
    """
    Generate an HTML preprocessing report comparing df_before and df_after.

    Parameters
    ----------
    df_before   : raw DataFrame (before preprocessing)
    df_after    : preprocessed DataFrame
    output_path : path to write the HTML report
    title       : dataset name shown in the report header

    Returns
    -------
    Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols_before = set(df_before.columns)
    cols_after  = set(df_after.columns)
    added   = sorted(cols_after  - cols_before)
    removed = sorted(cols_before - cols_after)

    stats_before = _col_stats(df_before)
    stats_after  = _col_stats(df_after)

    n_missing_before = int(df_before.isna().sum().sum())
    n_missing_after  = int(df_after.isna().sum().sum())
    total_cells_b = df_before.shape[0] * df_before.shape[1]
    total_cells_a = df_after.shape[0]  * df_after.shape[1]
    pct_missing_before = round(n_missing_before / total_cells_b * 100, 1) if total_cells_b else 0
    pct_missing_after  = round(n_missing_after  / total_cells_a * 100, 1) if total_cells_a else 0

    histograms = _build_histograms(df_before, df_after)

    added_html   = "".join(f"<span class='col-chip added'>{c}</span>"   for c in added)   or "<span style='color:var(--muted);font-size:.8rem'>none</span>"
    removed_html = "".join(f"<span class='col-chip removed'>{c}</span>" for c in removed) or "<span style='color:var(--muted);font-size:.8rem'>none</span>"

    html = _HTML_TEMPLATE.format(
        title=title,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        rows_before=df_before.shape[0],
        cols_before=df_before.shape[1],
        rows_after=df_after.shape[0],
        cols_after=df_after.shape[1],
        n_numeric_before=len(df_before.select_dtypes("number").columns),
        n_cat_before=len(df_before.select_dtypes(["object", "category"]).columns),
        n_added=len(added),
        n_removed=len(removed),
        n_missing_before=n_missing_before,
        n_missing_after=n_missing_after,
        pct_missing_before=pct_missing_before,
        pct_missing_after=pct_missing_after,
        mem_before=_fmt_memory(df_before),
        mem_after=_fmt_memory(df_after),
        added_cols_html=added_html,
        removed_cols_html=removed_html,
        stats_before_rows="".join(_stats_row(s) for s in stats_before),
        stats_after_rows="".join(_stats_row(s) for s in stats_after),
        hist_data_json=json.dumps(histograms, ensure_ascii=False),
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path
