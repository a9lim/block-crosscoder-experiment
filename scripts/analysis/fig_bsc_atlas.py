"""The BSC atlas: every sane-frequency block, hoverable (plotly HTML).

The shape-space census (`p4b_census.png`) upgraded to an interactive
atlas: each block of both surviving 4b arms is one mark at (code
participation ratio, top-2 eigenvalue mass), colored by where its
decoder energy peaks in depth, sized by firing frequency. Hovering any
mark shows the block's card: name (when the findings named it), firing
frequency, code spectrum summary, and the per-site decoder-share and
eval site-energy profiles as text sparklines. Named blocks carry a dot
outline + annotation; clique members (J>0.9 co-activation components)
render as ×.

One figure answers "what did the dictionary learn": rank-1 codes on the
right-bottom, planar ring/line manifolds on the top shelf, the clique
tiling pinned at (1,1), and the depth story in color.

  python scripts/analysis/fig_bsc_atlas.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import _style as st
from _names import CLIQUES, NAMES

DATA = Path("data/analysis")
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
BARS = "▁▂▃▄▅▆▇█"

ARMS = {
    "primary": ("evalstats_pilot.npz", "geometry_pilot.npz"),
    "renorm": ("evalstats_pilot_renorm.npz", "geometry_pilot_renorm.npz"),
}


def depth_colors(n: int = 8):
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("depth", st.BLUE_RAMP)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def sparkline(v: np.ndarray) -> str:
    v = np.asarray(v, np.float64)
    v = v / max(v.max(), 1e-12)
    return "".join(BARS[min(int(x * (len(BARS) - 1) + 0.5), len(BARS) - 1)]
                   for x in v)


def arm_panel(fig, col, arm, ev_file, geo_file):
    ev = np.load(DATA / ev_file)
    geo = np.load(DATA / geo_file)
    evals = np.linalg.eigvalsh(ev["zz"].astype(np.float64))[:, ::-1]
    evals = np.clip(evals, 0, None)
    tot = np.maximum(evals.sum(1), 1e-12)
    pr = tot ** 2 / np.maximum((evals ** 2).sum(1), 1e-24)
    top2 = evals[:, :2].sum(1) / tot
    freq = ev["fire_count"] / int(ev["n_tokens"])
    share = geo["share"]          # [G, S] decoder energy share
    senergy = ev["site_energy"]   # [G, S] eval-split activation energy
    peak = share.argmax(1)
    sane = (freq >= 1e-4) & (freq <= 0.05)
    names = NAMES[arm]
    clique = set(CLIQUES[arm])
    cols = depth_colors()

    def card(g: int) -> str:
        nm = names.get(g)
        head = f"<b>b{g}</b>" + (f" — {nm}" if nm else "")
        if g in clique:
            head += " [clique]"
        return (f"{head}<br>freq {freq[g]:.4f}   code PR {pr[g]:.2f}   "
                f"top-2 {top2[g]:.2f}<br>"
                f"decoder share  {sparkline(share[g])}  (peak L{SITES[peak[g]]})<br>"
                f"site energy    {sparkline(senergy[g])}<br>"
                f"spectrum {' '.join(f'{e/tot[g]:.2f}' for e in evals[g])}")

    idx = np.where(sane)[0]
    plain = np.array([g for g in idx if g not in clique and g not in names])
    fig.add_trace(go.Scatter(
        x=pr[plain], y=top2[plain], mode="markers",
        marker=dict(size=3 + 2.5 * (np.log10(freq[plain]) + 4),
                    color=[cols[peak[g]] for g in plain], opacity=0.55),
        hovertext=[card(g) for g in plain], hoverinfo="text",
        name=f"{arm} blocks", showlegend=False), 1, col)
    cl = np.array([g for g in idx if g in clique])
    if len(cl):
        fig.add_trace(go.Scatter(
            x=pr[cl], y=top2[cl], mode="markers",
            marker=dict(size=7, symbol="x",
                        color=[cols[peak[g]] for g in cl]),
            hovertext=[card(g) for g in cl], hoverinfo="text",
            name=f"{arm} clique", showlegend=False), 1, col)
    nb = np.array([g for g in idx if g in names])
    fig.add_trace(go.Scatter(
        x=pr[nb], y=top2[nb], mode="markers",
        marker=dict(size=9, color=[cols[peak[g]] for g in nb],
                    line=dict(color=st.INK, width=1.5)),
        hovertext=[card(g) for g in nb], hoverinfo="text",
        name=f"{arm} named", showlegend=False), 1, col)
    for g in nb:
        fig.add_annotation(
            x=pr[g], y=top2[g], text=f"b{g}", xref=f"x{col if col>1 else ''}",
            yref=f"y{col if col>1 else ''}", showarrow=False, yshift=10,
            font=dict(size=8, color=st.INK2))
    n_hidden = int((~sane).sum())
    fig.add_annotation(
        x=0.02, y=0.02, xref=f"x{col if col>1 else ''} domain",
        yref=f"y{col if col>1 else ''} domain", showarrow=False,
        text=(f"{arm}: {sane.sum()} blocks shown "
              f"({n_hidden} outside freq [1e-4, 0.05])"),
        font=dict(size=10, color=st.MUTED), xanchor="left")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=[f"{a} arm" for a in ARMS], horizontal_spacing=0.04)
    for col, (arm, (ev_file, geo_file)) in enumerate(ARMS.items(), 1):
        arm_panel(fig, col, arm, ev_file, geo_file)
    cols = depth_colors()
    for i, L in enumerate(SITES):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=cols[i]), name=f"peak L{L}"))
    fig.update_layout(
        title=("The BSC atlas — every sane-frequency block in shape space "
               "(hover for the block card)<br><sup>x: code participation "
               "ratio (1 = rank-1 code, 4 = isotropic); y: top-2 eigenvalue "
               "mass (1 = planar); color: depth of decoder-energy peak; "
               "size: firing frequency; ×: clique member; outlined+labeled: "
               "named in the findings</sup>"),
        height=680, width=1400, paper_bgcolor=st.SURFACE,
        plot_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        legend=dict(font=dict(size=10), orientation="v"),
        hoverlabel=dict(font=dict(family="Menlo, monospace", size=11)),
    )
    for ax in ("xaxis", "xaxis2"):
        fig.update_layout({ax: dict(title="code participation ratio",
                                    gridcolor=st.GRID, range=[0.9, 4.05])})
    fig.update_layout(yaxis=dict(title="top-2 eigenvalue mass",
                                 gridcolor=st.GRID, range=[0.45, 1.02]),
                      yaxis2=dict(gridcolor=st.GRID, range=[0.45, 1.02]))
    fig.write_html(OUT / "p4b_atlas.html", include_plotlyjs=True)
    print("atlas written", flush=True)


if __name__ == "__main__":
    main()
