"""3D geometry views for the 4b pilot (plotly HTML), geometry-pass set.

  p4b_geo_share_3d.html      per-block depth-energy share surface,
                             primary vs renorm side by side
  p4b_month_flow_3d.html     month class means in ONE fixed joint-PCA
  p4b_cardinal_flow_3d.html  basis across all depths (no per-depth
                             refit, no Procrustes): the manifold's
                             actual drift/rotation through the stack,
                             the component the aligned stacks gauge away
  p4b_crossarm_cardinal_3d.html
                             the cardinal line seen through b2146
                             (primary) and b3194 (renorm) frames,
                             per-depth planes, arms overlaid after a
                             per-depth cross-arm Procrustes

Viz-gauge honesty: whitened bases differ per site, so the fixed joint
basis (like the Procrustes alignment of the stack figures) is a
visualization gauge, not a claim of a shared coordinate system; the
audited numbers live in crossarm_pilot4b.json / geometry4b_summary.json.

  python scripts/analysis/fig_geometry4b_3d.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import _style as st

DATA = Path("data/analysis")
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
S = len(SITES)
MONTHS = st.MONTHS
CARDINALS = ["one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen", "twenty"]


def share_surface() -> None:
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "surface"}] * 2],
        subplot_titles=("primary", "renorm (F7)"),
        horizontal_spacing=0.02)
    for col, name in ((1, "geometry_pilot"), (2, "geometry_pilot_renorm")):
        share = np.load(DATA / f"{name}.npz")["share"]
        order = np.argsort(share @ np.arange(S))
        fig.add_trace(go.Surface(
            z=share[order], x=list(range(S)), y=list(range(share.shape[0])),
            colorscale="Blues", cmin=0, cmax=0.8,
            showscale=(col == 2),
            colorbar=dict(title="share", len=0.6) if col == 2 else None,
        ), row=1, col=col)
    for scene in ("scene", "scene2"):
        fig.update_layout(**{scene: dict(
            xaxis=dict(title="site", ticktext=[f"L{s}" for s in SITES],
                       tickvals=list(range(S))),
            yaxis_title="block (depth-centroid order)",
            zaxis_title="energy share",
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9)),
        )})
    fig.update_layout(
        title="Per-block depth-energy allocation (Gram rows sum to 1): "
              "the L30 cliff vs renorm's plateau",
        height=640, width=1200, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK))
    fig.write_html(OUT / "p4b_geo_share_3d.html", include_plotlyjs=True)
    print("share 3d written", flush=True)


def flow(family: str, means_key: str, labels: list[str]) -> None:
    zm = np.load(DATA / "zoo_means_zoo4b.npz")
    M = zm[means_key].transpose(1, 0, 2).astype(np.float64)  # [S, C, d]
    C = M.shape[1]
    Mc = M - M.mean(1, keepdims=True)          # center per site
    X = Mc.reshape(S * C, -1)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = (X @ Vt[:3].T).reshape(S, C, 3)
    colors = st.cyclic_colors(C)
    fig = go.Figure()
    for c in range(C):
        fig.add_trace(go.Scatter3d(
            x=P[:, c, 0], y=P[:, c, 1], z=P[:, c, 2],
            mode="lines+markers",
            line=dict(color=colors[c], width=4),
            marker=dict(size=[3 + 2.5 * s / (S - 1) for s in range(S)],
                        color=colors[c]),
            name=labels[c],
        ))
    for s in range(S):
        loop = list(range(C)) + ([0] if family == "month" else [])
        fig.add_trace(go.Scatter3d(
            x=P[s, loop, 0], y=P[s, loop, 1], z=P[s, loop, 2],
            mode="lines",
            line=dict(color="rgba(137,135,129,0.35)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    var3 = float((np.linalg.svd(X, compute_uv=False)[:3] ** 2).sum()
                 / (np.linalg.svd(X, compute_uv=False) ** 2).sum())
    fig.update_layout(
        title=f"The {family} manifold drifting through gemma-3-4b depth — "
              f"one fixed joint-PCA basis, no per-depth refit "
              f"(viz gauge; {var3:.0%} of class-mean variance). "
              f"Marker size grows with depth L9-L30.",
        height=720, width=980, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                   camera=dict(eye=dict(x=1.6, y=1.4, z=1.0))),
    )
    fig.write_html(OUT / f"p4b_{family}_flow_3d.html", include_plotlyjs=True)
    print(f"{family} flow 3d written", flush=True)


def procrustes_2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def crossarm_cardinal() -> None:
    zm = np.load(DATA / "zoo_means_zoo4b.npz")
    M = zm["cardinal_means"].transpose(1, 0, 2).astype(np.float64)
    fp = np.load(DATA / "frames_pilot_primary.npz")
    fr = np.load(DATA / "frames_pilot_renorm.npz")
    Fp = fp["frames"][:, fp["blocks"].tolist().index(2146)]  # [S, 4, d]
    Fr = fr["frames"][:, fr["blocks"].tolist().index(3194)]
    C = M.shape[1]
    colors = st.cyclic_colors(C)
    zstep = 1.2
    fig = go.Figure()
    prev = None
    for s in range(S):
        planes = []
        for F in (Fp, Fr):
            Y = (M[s] - M[s].mean(0)) @ F[s].T          # [C, 4] block code
            _, _, Vt = np.linalg.svd(Y, full_matrices=False)
            Q = Y @ Vt[:2].T
            Q /= max(np.sqrt((Q**2).mean()), 1e-9)      # per-depth RMS gauge
            planes.append(Q)
        planes[1] = planes[1] @ procrustes_2d(planes[0], planes[1])
        if prev is not None:
            R = procrustes_2d(prev, planes[0])
            planes = [p @ R for p in planes]
        prev = planes[0]
        z0 = [s * zstep] * C
        for c in range(C):  # cross-arm mismatch connectors
            fig.add_trace(go.Scatter3d(
                x=[planes[0][c, 0], planes[1][c, 0]],
                y=[planes[0][c, 1], planes[1][c, 1]], z=[z0[0]] * 2,
                mode="lines", line=dict(color="rgba(227,73,72,0.5)", width=2),
                showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=planes[0][:, 0], y=planes[0][:, 1], z=z0,
            mode="lines+markers+text",
            marker=dict(size=5, color=colors, symbol="circle"),
            line=dict(color=st.INK2, width=3),
            text=CARDINALS if s == S - 1 else None,
            textfont=dict(size=8, color=st.INK2),
            name=f"L{SITES[s]} primary b2146" if s == 0 else None,
            showlegend=(s == 0)))
        fig.add_trace(go.Scatter3d(
            x=planes[1][:, 0], y=planes[1][:, 1], z=z0,
            mode="lines+markers",
            marker=dict(size=4, color=colors, symbol="diamond-open"),
            line=dict(color="rgba(82,81,78,0.55)", width=2, dash="dot"),
            name="renorm b3194" if s == 0 else None,
            showlegend=(s == 0)))
    fig.update_layout(
        title="The cardinal number-line through both arms' capturing "
              "blocks — per-depth code planes, renorm Procrustes-mapped "
              "onto primary (viz gauge). Red rungs = cross-arm mismatch.",
        height=760, width=1000, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        scene=dict(
            xaxis_title="plane 1", yaxis_title="plane 2",
            zaxis=dict(title="depth", ticktext=[f"L{s}" for s in SITES],
                       tickvals=[s * zstep for s in range(S)]),
            camera=dict(eye=dict(x=1.7, y=1.5, z=0.9))),
    )
    fig.write_html(OUT / "p4b_crossarm_cardinal_3d.html",
                   include_plotlyjs=True)
    print("crossarm cardinal 3d written", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    share_surface()
    flow("month", "month_means", MONTHS)
    flow("cardinal", "cardinal_means", CARDINALS)
    crossarm_cardinal()


if __name__ == "__main__":
    main()
