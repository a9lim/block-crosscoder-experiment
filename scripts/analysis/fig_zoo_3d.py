"""3D cross-depth manifold stacks for the 4b pilot (plotly HTML).

The interim `rings_cross_layer_3d.html` convention carried to the pilot
artifacts: per-depth 2D planes stacked along a depth axis, consecutive
depths Procrustes-aligned (viz-grade gauge fixing only — per-depth
geometry untouched), same-class connectors across depths.

Views:
  - stream: month/weekday class means of the raw whitened stream at the
    8 pilot sites, first-harmonic (calendar-Fourier) planes, with faint
    per-token clouds
  - frames: the same tokens seen through b595's / b862's per-site
    decoder frames (the captured block's view of the manifold)
  - zoo: any family means extracted from a generalized-probe npz
    (`zoo_means_pilot4b.npz`) — cyclic families get harmonic planes,
    linear families PCA planes with Spearman order along PC1

Viz gauge: class-mean planes are RMS-normalized per depth (shape, not
scale, is the cross-depth comparison; the honest per-depth scale story
is the allocation/whitener figure set).

  python scripts/analysis/fig_pilot4b_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import _style as st
from tier_a_ring_tests import MONTHS, WEEKDAYS

DATA = Path("data/analysis")
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
ZSTEP = 1.2

FAMILY_LABELS = {
    "month": MONTHS,
    "weekday": WEEKDAYS,
    "ordinal": ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th",
                "9th", "10th", "11th", "12th", "13th", "14th", "15th",
                "16th", "17th", "18th", "19th", "20th"],
    "cardinal": ["one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                 "fourteen", "fifteen", "sixteen", "seventeen",
                 "eighteen", "nineteen", "twenty"],
    "digit": [str(d) for d in range(10)],
    "season": ["winter", "spring", "summer", "autumn"],
    "compass": ["N", "E", "S", "W"],
}
CYCLIC = {"month", "weekday", "season", "compass"}
# atlas tranche: labels ride in from the probe npz meta (country names
# are too many to hardcode twice); color fits harmonic planes on the
# hue-wheel prefix and projects the achromatic classes into them
HUE_PREFIX = 6


def procrustes_2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def harmonic_basis(X: np.ndarray):
    """First-harmonic plane basis of centered class rows X [C, d]."""
    C = X.shape[0]
    F = np.fft.fft(X, axis=0)
    power = (np.abs(F[1 : C // 2 + 1]) ** 2).sum(1)
    h1 = float(power[0] / power.sum())
    u = np.real(F[1])
    v = -np.imag(F[1])
    u /= np.linalg.norm(u)
    v -= u * (v @ u)
    v /= np.linalg.norm(v)
    return np.stack([u, v], 1), h1


def pca_basis(X: np.ndarray):
    """PCA top-2 basis of centered class rows; stat = Spearman |rho| of
    class order along PC1 (the linear-family order statistic)."""
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    p1 = X @ Vt[0]
    from scipy.stats import spearmanr

    rho = float(abs(spearmanr(np.arange(len(p1)), p1).statistic))
    return Vt[:2].T, rho


def stack_planes(mean_stack, cyclic: bool, fit_idx=None):
    """[S, C, d] class means → per-depth aligned 2D planes + stat.

    fit_idx: fit the plane basis on this class subset only (hue wheel),
    then project every class into it.
    """
    planes = []
    prev = None
    for s in range(mean_stack.shape[0]):
        X = mean_stack[s] - mean_stack[s].mean(0)
        Xf = X if fit_idx is None else X[fit_idx]
        basis, stat = (harmonic_basis if cyclic else pca_basis)(Xf)
        P = X @ basis
        scale = max(np.sqrt((P**2).mean()), 1e-9)  # viz gauge: per-depth RMS
        P = P / scale
        if prev is not None:
            R = procrustes_2d(prev, P)
            P = P @ R
            basis = basis @ R
        planes.append((P, basis, stat, scale))
        prev = P
    return planes


def stack_figure(planes, labels, title, stat_name, clouds=None,
                 ring_idx=None):
    C = len(labels)
    colors = st.cyclic_colors(C)
    fig = go.Figure()
    zpos = [i * ZSTEP for i in range(len(SITES))]
    for cix in range(C):
        fig.add_trace(go.Scatter3d(
            x=[planes[s][0][cix, 0] for s in range(len(SITES))],
            y=[planes[s][0][cix, 1] for s in range(len(SITES))],
            z=zpos, mode="lines",
            line=dict(color="rgba(137,135,129,0.45)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    for s, (P, _, stat, _) in enumerate(planes):
        if clouds is not None:
            Q = clouds[s]
            fig.add_trace(go.Scatter3d(
                x=Q[:, 0], y=Q[:, 1], z=[zpos[s]] * len(Q),
                mode="markers",
                marker=dict(size=1.5, color=[colors[c] for c in clouds[s + len(SITES)]],
                            opacity=0.25),
                showlegend=False, hoverinfo="skip",
            ))
        loop = (list(range(C)) + [0]) if ring_idx is None else \
            (list(ring_idx) + [ring_idx[0]] if len(ring_idx) else [])
        if loop:
            fig.add_trace(go.Scatter3d(
                x=P[loop, 0], y=P[loop, 1], z=[zpos[s]] * len(loop),
                mode="lines", line=dict(color=st.INK2, width=3),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=[zpos[s]] * C,
            mode="markers+text",
            marker=dict(size=5, color=colors),
            text=labels, textposition="top center",
            textfont=dict(size=9, color=st.INK2),
            name=f"L{SITES[s]} ({stat_name} {stat:.0%})",
        ))
    fig.update_layout(
        title=title, height=720, width=1000, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        scene=dict(
            xaxis_title="plane 1", yaxis_title="plane 2",
            zaxis=dict(title="depth", ticktext=[f"L{s}" for s in SITES],
                       tickvals=zpos),
            camera=dict(eye=dict(x=1.7, y=1.5, z=0.9)),
        ),
    )
    return fig


def cap_means(za, is_cap, fi: int, C: int):
    fam, cls = za["fam"], za["cls"]
    m = (fam == fi) & is_cap
    a, c = za["acts"][m], cls[m]
    return np.stack([[a[c == k, s].mean(0) for k in range(C)]
                     for s in range(len(SITES))]), a, c


def token_clouds(a, c, planes, per_depth: int = 500):
    """Project a token subsample into each depth's plane; returns
    clouds list: [S] coord arrays + [S] class arrays (concatenated)."""
    rng = np.random.default_rng(0)
    coords, classes = [], []
    for s, (_, basis, _, scale) in enumerate(planes):
        mu = a[:, s].mean(0)
        idx = rng.choice(len(a), min(per_depth, len(a)), replace=False)
        # basis already carries the Procrustes rotation; scale is the
        # class-mean plane's RMS gauge, shared so cloud spread is honest
        coords.append((a[idx, s] - mu) @ basis / scale)
        classes.append(c[idx])
    return coords + classes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    za = np.load(DATA / "calendar_probe_acts_pilot4b.npz")
    zc = np.load(
        DATA / "block_codes_bsc_lam0.001_seed0_G4096_k32_renorm_pilot4b.npz")
    is_cap = zc["is_cap"]
    blocks = zc["blocks"].tolist()

    # -- stream views ----------------------------------------------------
    for family, fi, block in (("month", 1, 595), ("weekday", 0, 862)):
        labels = FAMILY_LABELS[family]
        C = len(labels)
        means, a, c = cap_means(za, is_cap, fi, C)
        planes = stack_planes(means, cyclic=True)
        fig = stack_figure(
            planes, labels,
            f"The {family} manifold across gemma-3-4b depth — raw whitened "
            f"stream,<br>first-harmonic planes, consecutive depths "
            f"Procrustes-aligned (viz gauge)",
            "1st harmonic", clouds=token_clouds(a, c, planes))
        fig.write_html(OUT / f"p4b_stream_{family}_3d.html",
                       include_plotlyjs=True)

        # -- the captured block's view (frames) --------------------------
        frames = zc["frames"][:, blocks.index(block)]  # [S, b, d]
        proj_means = np.stack(
            [[(a[c == k, s] @ frames[s].T).mean(0) for k in range(C)]
             for s in range(len(SITES))])
        fplanes = stack_planes(proj_means, cyclic=True)
        fig = stack_figure(
            fplanes, labels,
            f"The same {family} tokens through b{block}'s rotating per-site "
            f"frames<br>(renorm arm — the captured block's view)",
            "1st harmonic")
        fig.write_html(OUT / f"p4b_b{block}_frames_3d.html",
                       include_plotlyjs=True)
        print(f"{family}: stream + b{block} frames written", flush=True)

    # -- zoo stream views (means extracted from the zoo probes) ----------
    from block_crosscoder_experiment.phase0.labels import FAMILIES

    zoos = [p for p in (DATA / "zoo_means_zoo4b.npz",
                        DATA / "zoo_means_atlas4b.npz",
                        DATA / "zoo_means_pilot4b.npz") if p.exists()]
    seen = set()
    for zoo in zoos:
        zm = np.load(zoo)
        families = json.loads(str(zm["meta"]))["families"]
        for family in families:
            key = f"{family}_means"
            if key not in zm or family in seen:
                continue
            seen.add(family)
            means = zm[key].transpose(1, 0, 2)  # [C,S,d] -> [S,C,d]
            labels = FAMILY_LABELS.get(family, FAMILIES[family])
            cyc = family in CYCLIC
            fit_idx = list(range(HUE_PREFIX)) if family == "color" else None
            ring_idx = fit_idx if family == "color" else \
                [] if family == "country" else None
            planes = stack_planes(means, cyclic=cyc or family == "color",
                                  fit_idx=fit_idx)
            fig = stack_figure(
                planes, labels,
                f"The {family} family across gemma-3-4b depth — raw "
                f"whitened stream,<br>"
                + ("hue-wheel harmonic planes (achromatic projected in)"
                   if family == "color" else
                   "first-harmonic planes" if cyc else
                   "PCA planes (Spearman order along PC1)"),
                "1st harmonic" if cyc or family == "color" else "|rho|",
                ring_idx=ring_idx)
            fig.write_html(OUT / f"p4b_zoo_{family}_3d.html",
                           include_plotlyjs=True)
            print(f"zoo {family}: written", flush=True)
    if not zoos:
        print("no zoo means npz yet — zoo views skipped")


if __name__ == "__main__":
    main()
