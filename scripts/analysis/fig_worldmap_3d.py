"""The world map in activation space (and in a block's code), 3D stack.

Gurnee & Tegmark (2023) found linear lat/lon in Llama's stream; here the
question is BSC-shaped: the stream side is the replication on gemma-3-4b
whitened sites, the dictionary side asks whether a single 4-dim block
code carries the same map.

Per depth: country class means → top PREDICTORS PCs → leave-one-out
linear decode of standardized (lat, lon); each country is plotted at its
*predicted* position, colored by continent, with an error whisker to its
true position (gray ghost). Depth runs up the z-axis; if the atlas zoo
tests found a country-capturing block, its code-plane LOO map joins as
the top layer.

  python scripts/analysis/fig_worldmap_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import _style as st
from _geo import CONTINENT_ORDER, COUNTRY_GEO
from block_crosscoder_experiment.phase0.labels import COUNTRIES

DATA = Path("data/analysis")
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
ZSTEP = 1.2
PREDICTORS = 10
MIN_COUNT = 5

CONT_COLOR = dict(zip(CONTINENT_ORDER, st.CAT[:5]))
# anchor labels only — 48 names on one layer is a smear; hover has the rest
ANCHORS = {"England", "Russia", "China", "Japan", "India", "Australia",
           "Canada", "Brazil", "Egypt", "Iceland", "Indonesia", "Chile"}


def loo_decode(X: np.ndarray, Y: np.ndarray):
    """Leave-one-out linear decode; returns predictions + R² per column.

    X [C, p] predictors (already centered/reduced), Y [C, 2] targets
    (standardized). Each row is predicted by a model fit on the rest.
    """
    C = X.shape[0]
    Xa = np.column_stack([X, np.ones(C)])
    pred = np.empty_like(Y)
    for i in range(C):
        m = np.arange(C) != i
        beta, *_ = np.linalg.lstsq(Xa[m], Y[m], rcond=None)
        pred[i] = Xa[i] @ beta
    r2 = 1 - ((Y - pred) ** 2).sum(0) / (Y ** 2).sum(0)
    return pred, r2


def loo_perm_p(X: np.ndarray, Y: np.ndarray, n_perms: int = 1000,
               seed: int = 0) -> float:
    _, r2 = loo_decode(X, Y)
    obs = r2.mean()
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perms):
        _, r2p = loo_decode(X, Y[rng.permutation(len(Y))])
        ge += r2p.mean() >= obs
    return (1 + ge) / (n_perms + 1)


def reduce_means(M: np.ndarray, k: int = PREDICTORS) -> np.ndarray:
    X = M - M.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:k].T


def layer_traces(fig, pred, Y, names, conts, zlev, label, show_names,
                 legend=False):
    for cont in CONTINENT_ORDER:
        m = np.array([c == cont for c in conts])
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=pred[m, 0], y=pred[m, 1], z=[zlev] * int(m.sum()),
            mode="markers+text" if show_names else "markers",
            marker=dict(size=4.5, color=CONT_COLOR[cont]),
            text=[n if n in ANCHORS else ""
                  for n, mm in zip(names, m) if mm] if show_names else None,
            textposition="top center",
            textfont=dict(size=8, color=st.INK2),
            name=cont, legendgroup=cont, showlegend=legend,
            hovertext=[f"{n} ({label})" for n, mm in zip(names, m) if mm],
            hoverinfo="text"))
    # error whiskers to the true positions
    xs, ys, zs = [], [], []
    for i in range(len(names)):
        xs += [pred[i, 0], Y[i, 0], None]
        ys += [pred[i, 1], Y[i, 1], None]
        zs += [zlev, zlev, None]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="rgba(137,135,129,0.5)", width=1.5),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=Y[:, 0], y=Y[:, 1], z=[zlev] * len(names), mode="markers",
        marker=dict(size=2.5, color=st.MUTED, opacity=0.7),
        showlegend=False, hoverinfo="skip"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zm = np.load(DATA / "zoo_means_atlas4b.npz")
    tests = json.loads(
        (DATA / "zoo_block_tests_atlas4b.json").read_text())
    means = zm["country_means"]  # [C, S, d]
    counts = np.array(
        tests[next(iter(tests))]["country"]["class_counts"])
    keep = counts >= MIN_COUNT
    names = [c for c, k in zip(COUNTRIES, keep) if k]
    conts = [COUNTRY_GEO[n][2] for n in names]
    latlon = np.array([COUNTRY_GEO[n][:2] for n in names])
    # plot in (lon, lat) so the layer reads as a map
    Y = (latlon - latlon.mean(0)) / latlon.std(0)
    Ymap = Y[:, ::-1]

    fig = go.Figure()
    stats = []
    for s, L in enumerate(SITES):
        X = reduce_means(means[keep][:, s])
        pred, r2 = loo_decode(X, Y)
        p = loo_perm_p(X, Y)
        stats.append((f"L{L}", r2, p))
        layer_traces(fig, pred[:, ::-1], Ymap, names, conts, s * ZSTEP,
                     f"L{L}", show_names=(s == len(SITES) - 1),
                     legend=(s == 0))

    # dictionary layers: the best country block's code-plane map, per arm
    zlev = len(SITES) * ZSTEP + ZSTEP
    ticks = [f"L{L}" for L in SITES]
    tickvals = [i * ZSTEP for i in range(len(SITES))]
    for arm in tests:
        entry = tests[arm]["country"]
        codes_f = DATA / f"zoo_codes_{arm}_atlas4b.npz"
        if not codes_f.exists():
            continue
        zc = np.load(codes_f)
        families = json.loads(str(zc["meta"]))["families"]
        fi = families.index("country")
        bix = zc["blocks"].tolist().index(entry["best_block"])
        m = zc["fam"] == fi
        cls = zc["cls"][m]
        z_tok = zc["z_sel"][m][:, bix].astype(np.float32)
        cm = np.stack([z_tok[cls == k].mean(0) if (cls == k).any()
                       else np.zeros(z_tok.shape[1])
                       for k in range(len(COUNTRIES))])[keep]
        X = cm - cm.mean(0)
        pred, r2 = loo_decode(X, Y)
        p = loo_perm_p(X, Y)
        stats.append((f"{arm} b{entry['best_block']}", r2, p))
        layer_traces(fig, pred[:, ::-1], Ymap, names, conts, zlev,
                     f"{arm} b{entry['best_block']}", show_names=False)
        ticks.append(f"b{entry['best_block']} ({arm})")
        tickvals.append(zlev)
        zlev += ZSTEP

    chunks = [
        f"{lab}: lat {r2[0]:.2f} lon {r2[1]:.2f} (p {p:.3g})"
        for lab, r2, p in stats]
    lines = "<br>".join("   ".join(chunks[i:i + 4])
                        for i in range(0, len(chunks), 4))
    fig.update_layout(
        title=("The world map in gemma-3-4b's stream — and in single "
               "4-dim block codes<br><sup>per layer: countries at their "
               "LOO-decoded (lon, lat), whiskers to truth (gray); "
               f"{PREDICTORS} stream PCs / 4 code dims as predictors. "
               "LOO R²:<br>" + lines + "</sup>"),
        height=820, width=1150, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        scene=dict(
            xaxis_title="decoded longitude (std)",
            yaxis_title="decoded latitude (std)",
            zaxis=dict(title="", ticktext=ticks, tickvals=tickvals,
                       tickfont=dict(size=9)),
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.7)),
        ),
        legend=dict(font=dict(size=10)),
    )
    fig.write_html(OUT / "p4b_worldmap_3d.html", include_plotlyjs=True)
    for lab, r2, p in stats:
        print(f"{lab}: LOO R2 lat {r2[0]:.3f} lon {r2[1]:.3f} p {p:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
