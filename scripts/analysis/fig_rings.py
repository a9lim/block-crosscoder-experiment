"""Figures: phase-0/0.5 cross-layer ring geometry (gemma-3-4b, 65k SAEs).

The 4b story rebuilt as figures from the pinned artifacts:
  - the month decoder ring per depth, stacked in 3D with viz-grade
    Procrustes alignment between consecutive depths (gauge fixing only;
    per-depth geometry untouched)
  - code-map / CCA heatmaps across depth pairs (audited numbers from
    cross_layer.json)
  - the L22 "ring below the clustering threshold" panel
  - the GPT-2 weekday control ring (cluster 937)

  python scripts/analysis/fig_rings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _style as st

st.apply()
ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data/analysis/npz/phase0_geometry.npz"
CL = ROOT / "data/analysis/phase05/cross_layer.json"
OUT = ROOT / "figures/interim"
DEPTHS = [9, 17, 22, 29]


def class_decoders(z, depth: int, family: str):
    """Per-class decoder rows via the top-1 class->feature map (may repeat)."""
    members = z[f"d{depth}_{family}_members"]
    c2f = z[f"d{depth}_{family}_class_to_feat"]
    dec = z[f"d{depth}_{family}_dec"]
    rows = np.full((len(c2f), dec.shape[1]), np.nan, dtype=np.float32)
    for cix, f in enumerate(c2f):
        if f >= 0:
            rows[cix] = dec[list(members).index(f)]
    return rows


def procrustes_2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation/reflection R aligning B to A (both [n,2], centered)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def harmonic_plane(rows: np.ndarray):
    """First-harmonic (Fourier over class index) plane of class rows.

    This is the supervised projection the adjacency/angle-order statistics
    detect: the plane spanned by sum_k x_k cos(2πk/C), sum_k x_k sin(2πk/C).
    Returns per-class 2D coords + the first-harmonic power fraction.
    """
    ok = ~np.isnan(rows[:, 0])
    C = rows.shape[0]
    X = rows.copy()
    X[~ok] = 0.0
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    Xc = Xn - Xn[ok].mean(0)
    Xc[~ok] = 0.0
    k = np.arange(C)
    F = np.fft.fft(Xc, axis=0)  # over class index
    power = (np.abs(F[1 : C // 2 + 1]) ** 2).sum(1)
    h1_frac = float(power[0] / power.sum())
    u = np.real(F[1])
    v = -np.imag(F[1])
    u /= np.linalg.norm(u)
    v -= u * (v @ u)
    v /= np.linalg.norm(v)
    P = np.full((C, 2), np.nan, dtype=np.float32)
    P[ok] = np.stack([Xc[ok] @ u, Xc[ok] @ v], axis=1)
    return P, h1_frac


def month_planes(z):
    """Per-depth 2D harmonic-plane coords of the month class decoders,
    Procrustes-aligned between consecutive depths (gauge only)."""
    planes = {}
    prev = None
    for depth in DEPTHS:
        rows = class_decoders(z, depth, "month")
        P, h1 = harmonic_plane(rows)
        ok = ~np.isnan(P[:, 0])
        if prev is not None:
            both = ok & ~np.isnan(prev[:, 0])
            R = procrustes_2d(prev[both], P[both])
            P = P @ R
        planes[depth] = (P, h1)
        prev = P
    return planes


def fig_ring_stack_3d(z) -> None:
    import plotly.graph_objects as go

    planes = month_planes(z)
    colors = st.cyclic_colors(12)
    fig = go.Figure()
    zpos = {d: i * 1.2 for i, d in enumerate(DEPTHS)}
    # same-month connectors across depths
    for cix in range(12):
        xs, ys, zs = [], [], []
        for d in DEPTHS:
            P, _ = planes[d]
            if not np.isnan(P[cix, 0]):
                xs.append(P[cix, 0]); ys.append(P[cix, 1]); zs.append(zpos[d])
        if len(xs) > 1:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color="rgba(137,135,129,0.45)", width=2),
                showlegend=False, hoverinfo="skip",
            ))
    for d in DEPTHS:
        P, var2 = planes[d]
        ok = ~np.isnan(P[:, 0])
        order = [c for c in range(12) if ok[c]]
        loop = order + [order[0]]
        fig.add_trace(go.Scatter3d(
            x=P[loop, 0], y=P[loop, 1], z=[zpos[d]] * len(loop),
            mode="lines", line=dict(color=st.INK2, width=3),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=P[ok, 0], y=P[ok, 1], z=[zpos[d]] * int(ok.sum()),
            mode="markers+text",
            marker=dict(size=5, color=[colors[c] for c in range(12) if ok[c]]),
            text=[st.MONTHS[c] for c in range(12) if ok[c]],
            textposition="top center",
            textfont=dict(size=9, color=st.INK2),
            name=f"layer {d} ({int(ok.sum())} distinct, "
                 f"1st harmonic {var2:.0%})",
        ))
    fig.update_layout(
        title="Month decoder rings across gemma-3-4b depths (65k SAEs) — "
              "calendar-Fourier (1st-harmonic) planes, supervised "
              "projection; consecutive depths Procrustes-aligned",
        height=680, width=980, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        scene=dict(
            xaxis_title="ring plane 1", yaxis_title="ring plane 2",
            zaxis=dict(
                title="depth",
                ticktext=[f"L{d}" for d in DEPTHS],
                tickvals=[zpos[d] for d in DEPTHS],
            ),
            camera=dict(eye=dict(x=1.7, y=1.5, z=0.9)),
        ),
    )
    fig.write_html(OUT / "rings_cross_layer_3d.html", include_plotlyjs=True)

    # PNG small-multiples companion.
    fig2, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, d in zip(axes, DEPTHS):
        P, var2 = planes[d]
        ok = ~np.isnan(P[:, 0])
        order = [c for c in range(12) if ok[c]]
        loop = order + [order[0]]
        ax.plot(P[loop, 0], P[loop, 1], color=st.GRID, lw=1.2, zorder=1)
        for c in range(12):
            if ok[c]:
                ax.scatter(*P[c], s=40, color=colors[c],
                           edgecolor=st.INK, lw=0.6, zorder=3)
                ax.annotate(st.MONTHS[c], xy=P[c], xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=st.INK2)
        ax.set_title(f"L{d} — {int(ok.sum())} distinct feats, "
                     f"1st harmonic {var2:.0%}", fontsize=9.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle(
        "Month decoder rings by depth — calendar-Fourier plane (supervised "
        "projection; the component the order statistics detect). "
        "L17 undersplits", y=1.04,
    )
    fig2.savefig(OUT / "rings_by_depth.png")
    plt.close(fig2)


def fig_codemap_heatmap() -> None:
    cl = json.loads(CL.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for ax, key, label in (
        (axes[0], "codemap_heldout_r2_ab", "held-out code-map R² (row to col)"),
        (axes[1], "cca_mean", "mean CCA correlation"),
    ):
        mat = np.full((4, 4), np.nan)
        for pair, res in cl["month"]["pairs"].items():
            a, b = (DEPTHS.index(int(x)) for x in pair.split("->"))
            mat[a, b] = res[key]
            mat[b, a] = res.get("codemap_heldout_r2_ba", res[key]) \
                if key.startswith("codemap") else res[key]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center",
                            va="center", fontsize=9,
                            color=st.SURFACE if mat[i, j] > 0.55 else st.INK2)
        ax.set_xticks(range(4), [f"L{d}" for d in DEPTHS])
        ax.set_yticks(range(4), [f"L{d}" for d in DEPTHS])
        ax.grid(False)
        ax.set_title(label, fontsize=10)
    fig.suptitle(
        "Phase 0.5, month family: codes correspond across depth (9-22-29) "
        "while raw spans sit at chance — L17's undersplit dictionary is the "
        "odd one out", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT / "rings_codemap_heatmap.png")
    plt.close(fig)


def fig_l22_threshold(z) -> None:
    rows = class_decoders(z, 22, "month")
    ok = ~np.isnan(rows[:, 0])
    X = rows[ok] / np.linalg.norm(rows[ok], axis=1, keepdims=True)
    cos = X @ X.T
    n = int(ok.sum())
    adj = [cos[i, (i + 1) % n] for i in range(n)]
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    colors = st.cyclic_colors(12)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.6),
                                  gridspec_kw={"width_ratios": [1, 1.05]})
    order = list(range(n)) + [0]
    ax.plot(P[order, 0], P[order, 1], color=st.GRID, lw=1.2)
    for c in range(n):
        ax.scatter(*P[c], s=44, color=colors[c], edgecolor=st.INK, lw=0.6,
                   zorder=3)
        ax.annotate(st.MONTHS[c], xy=P[c], xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=st.INK2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("L22 65k month decoders, top PCA plane")
    im = ax2.imshow(cos, cmap="Blues", vmin=-0.2, vmax=1)
    ax2.set_xticks(range(n), st.MONTHS[:n], fontsize=7)
    ax2.set_yticks(range(n), st.MONTHS[:n], fontsize=7)
    ax2.grid(False)
    cb = fig.colorbar(im, ax=ax2, fraction=0.045)
    cb.set_label("decoder cosine", fontsize=8.5)
    ax2.set_title(
        f"adjacent cos ≤ {max(adj):.2f} — every clustering threshold τ=0.5\n"
        "treats the ring as 12 singletons", fontsize=10,
    )
    fig.suptitle("The ring rides below the clustering threshold (H1's gap, "
                 "phase 0)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "rings_l22_threshold.png")
    plt.close(fig)


def fig_control(z) -> None:
    codes = z["ctl_weekday_codes"]  # [n, members]
    cls = z["ctl_weekday_token_cls"]
    dec = z["ctl_weekday_dec"]
    # per-class decoder: firing-affinity-weighted mean of member features
    aff = np.stack([codes[cls == c].mean(0) for c in range(7)])  # [7, m]
    w = aff / np.maximum(aff.sum(1, keepdims=True), 1e-9)
    rows = w @ (dec / np.linalg.norm(dec, axis=1, keepdims=True))
    P, h1 = harmonic_plane(rows)
    colors = st.cyclic_colors(7)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    loop = list(range(7)) + [0]
    ax.plot(P[loop, 0], P[loop, 1], color=st.GRID, lw=1.2, zorder=1)
    for c in range(7):
        ax.scatter(*P[c], s=52, color=colors[c], edgecolor=st.INK, lw=0.7,
                   zorder=3)
        ax.annotate(st.WEEKDAYS[c], xy=P[c], xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=st.INK2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        "Positive control: GPT-2 L7 weekday cluster (Engels), "
        "calendar-Fourier plane\nof affinity-weighted class decoders — "
        f"1st harmonic {h1:.0%} of Fourier power"
    )
    fig.savefig(OUT / "rings_control_weekday.png")
    plt.close(fig)


def main() -> None:
    z = np.load(NPZ)
    fig_ring_stack_3d(z)
    fig_codemap_heatmap()
    fig_l22_threshold(z)
    fig_control(z)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
