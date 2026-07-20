"""Anatomy of one BSC block: one shared code, eight rotating frames.

The explainer figure for what a block-sparse crosscoder unit *is*. Two
linked 3D scenes per flagship block:

  left  — THE SHARED CODE: every labeled token's 4-dim block code z_g,
          scattered in the block's code PCA basis (one point per token,
          regardless of depth — the code is the depth-invariant object),
          class means threaded in class order (ring loop / line path).
  right — EIGHT ROTATING FRAMES: the stream's class means in per-depth
          manifold planes (harmonic for cyclic, PCA for linear;
          consecutive depths Procrustes-aligned, per-depth RMS viz
          gauge), overlaid with the block's DECODED class means
          D_g^s z̄_k projected into the same planes (open "ghost"
          markers): the same code re-embedded at every depth by the
          per-site frames. Per-depth labels carry the frame⋂stream
          plane cosine, and the block's decoder share of that site.

The ghost/stream in-plane radius ratio is honest amplitude: the
fraction of the stream manifold's class displacement this one block
reconstructs (renorm decodes are un-rescaled by the store's site-RMS
scalars first).

Blocks: b595 (month ring, renorm arm), b2146 (cardinal line, primary).

  python scripts/analysis/fig_block_anatomy_3d.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import _style as st
from _names import SITE_RENORM_SCALARS
from fig_pilot4b_3d import (FAMILY_LABELS, harmonic_basis, pca_basis,
                            procrustes_2d)

DATA = Path("data/analysis")
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
ZSTEP = 1.2


def seq_colors(n: int):
    """Sequential blue ramp for ordered (line) classes."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    return [mcolors.to_hex(cm.Blues(0.25 + 0.7 * i / max(n - 1, 1)))
            for i in range(n)]


def orthonormal(V: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(V)
    return Q


def plane_cos(A: np.ndarray, B: np.ndarray) -> float:
    """Mean top-2 principal cosine between the spans of A, B [d, 2]."""
    s = np.linalg.svd(orthonormal(A).T @ orthonormal(B), compute_uv=False)
    return float(s[:2].mean())


def stack_planes_with_ghost(mean_stack, ghost_stack, cyclic):
    """Per-depth aligned planes for stream means + ghost decodes.

    Basis is fit on the STREAM means only (the model's own manifold is
    the reference frame); ghosts are projected into that basis and share
    the stream's RMS gauge, so ghost radius / stream radius is the
    block's honest amplitude fraction per depth.
    """
    out = []
    prev = None
    for s in range(mean_stack.shape[0]):
        X = mean_stack[s] - mean_stack[s].mean(0)
        Gh = ghost_stack[s] - ghost_stack[s].mean(0)
        basis, stat = (harmonic_basis if cyclic else pca_basis)(X)
        P = X @ basis
        Q = Gh @ basis
        scale = max(np.sqrt((P ** 2).mean()), 1e-9)
        P, Q = P / scale, Q / scale
        if prev is not None:
            R = procrustes_2d(prev, P)
            P, Q, basis = P @ R, Q @ R, basis @ R
        _, _, Vt_g = np.linalg.svd(Gh, full_matrices=False)
        fcos = plane_cos(basis, Vt_g[:2].T)
        out.append((P, Q, stat, fcos))
        prev = P
    return out


def anatomy_figure(z_tok, cls_tok, mean_stack, ghost_stack, labels, cyclic,
                   share, title, colors):
    C = len(labels)
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("the one shared code (4-dim, all depths at once)",
                        "eight rotating frames re-embed it at every depth"),
        horizontal_spacing=0.02,
    )

    # -- left: token codes in the block's code PCA basis ------------------
    zc = z_tok - z_tok.mean(0)
    _, sv, Vt = np.linalg.svd(zc, full_matrices=False)
    P3 = zc @ Vt[:3].T
    top2 = float((sv[:2] ** 2).sum() / (sv ** 2).sum())
    rng = np.random.default_rng(0)
    keep = rng.permutation(len(P3))[:4000]
    fig.add_trace(go.Scatter3d(
        x=P3[keep, 0], y=P3[keep, 1], z=P3[keep, 2], mode="markers",
        marker=dict(size=1.6, color=[colors[c] for c in cls_tok[keep]],
                    opacity=0.35),
        name="tokens", showlegend=False, hoverinfo="skip"), 1, 1)
    cm3 = np.stack([P3[cls_tok == k].mean(0) if (cls_tok == k).any()
                    else np.full(3, np.nan) for k in range(C)])
    loop = (list(range(C)) + [0]) if cyclic else list(range(C))
    fig.add_trace(go.Scatter3d(
        x=cm3[loop, 0], y=cm3[loop, 1], z=cm3[loop, 2],
        mode="lines+markers+text", text=[labels[i] for i in loop],
        textposition="top center", textfont=dict(size=10, color=st.INK),
        line=dict(color=st.INK2, width=4),
        marker=dict(size=5, color=[colors[i] for i in loop]),
        name=f"class means (top-2 var {top2:.0%})", showlegend=False,
        hovertext=[labels[i] for i in loop], hoverinfo="text"), 1, 1)

    # -- right: stream manifold + ghost decodes, stacked by depth ---------
    planes = stack_planes_with_ghost(mean_stack, ghost_stack, cyclic)
    zpos = [i * ZSTEP for i in range(len(SITES))]
    stat_name = "harm" if cyclic else "|ρ|"
    # line families with many classes: label milestones only, or the
    # depth stack drowns in text
    if C > 12:
        mask = set(range(0, C, 5)) | {C - 1}
        depth_labels = [labels[k] if k in mask else "" for k in range(C)]
    else:
        depth_labels = labels
    for cix in range(C):
        fig.add_trace(go.Scatter3d(
            x=[planes[s][0][cix, 0] for s in range(len(SITES))],
            y=[planes[s][0][cix, 1] for s in range(len(SITES))],
            z=zpos, mode="lines",
            line=dict(color="rgba(137,135,129,0.4)", width=2),
            showlegend=False, hoverinfo="skip"), 1, 2)
    for s, (P, Q, stat, fcos) in enumerate(planes):
        path = list(range(C)) + ([0] if cyclic else [])
        fig.add_trace(go.Scatter3d(
            x=P[path, 0], y=P[path, 1], z=[zpos[s]] * len(path),
            mode="lines", line=dict(color=st.INK2, width=3),
            showlegend=False, hoverinfo="skip"), 1, 2)
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=[zpos[s]] * C, mode="markers+text",
            marker=dict(size=4.5, color=colors),
            text=depth_labels, textposition="top center",
            textfont=dict(size=8, color=st.INK2),
            hovertext=labels, hoverinfo="text",
            name=(f"L{SITES[s]} — stream ({stat_name} {stat:.0%}), "
                  f"frame cos {fcos:.2f}, share {share[s]:.0%}"),
            legendgroup=f"L{SITES[s]}"), 1, 2)
        fig.add_trace(go.Scatter3d(
            x=Q[:, 0], y=Q[:, 1], z=[zpos[s]] * C, mode="markers",
            marker=dict(size=6, color=colors, symbol="diamond-open",
                        line=dict(width=2)),
            name=f"L{SITES[s]} — block decode", legendgroup=f"L{SITES[s]}",
            showlegend=False,
            hovertext=[f"{labels[k]} decode" for k in range(C)],
            hoverinfo="text"), 1, 2)

    fig.update_layout(
        title=title, height=760, width=1500, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        legend=dict(font=dict(size=10)),
        scene=dict(
            xaxis_title="code PC1", yaxis_title="code PC2",
            zaxis_title="code PC3",
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.8)),
        ),
        scene2=dict(
            xaxis_title="manifold plane 1", yaxis_title="manifold plane 2",
            zaxis=dict(title="depth", ticktext=[f"L{s}" for s in SITES],
                       tickvals=zpos),
            camera=dict(eye=dict(x=1.7, y=1.5, z=0.9)),
        ),
    )
    return fig


def month_b595():
    za = np.load(DATA / "calendar_probe_acts_pilot4b.npz")
    bc = np.load(
        DATA / "block_codes_bsc_lam0.001_seed0_G4096_k32_renorm_pilot4b.npz")
    geo = np.load(DATA / "geometry_pilot_renorm.npz")
    bix = bc["blocks"].tolist().index(595)
    m = (za["fam"] == 1) & bc["is_cap"]
    acts, cls = za["acts"][m], za["cls"][m]
    z_tok = bc["z_sel"][m][:, bix].astype(np.float32)
    frames = bc["frames"][:, bix]  # [S, b, d]
    C = 12
    zbar = np.stack([z_tok[cls == k].mean(0) for k in range(C)])
    mean_stack = np.stack([[acts[cls == k, s].mean(0) for k in range(C)]
                           for s in range(len(SITES))])
    # renorm decode lives in scalar-rescaled whitened space; undo per site
    ghost_stack = np.stack(
        [(zbar @ frames[s]) / SITE_RENORM_SCALARS[s]
         for s in range(len(SITES))])
    share = geo["share"][595]
    fig = anatomy_figure(
        z_tok, cls, mean_stack, ghost_stack, FAMILY_LABELS["month"], True,
        share,
        "Anatomy of a BSC block — b595, the month ring (renorm arm): "
        "one 4-dim code shared across depth,<br>per-site decoder frames "
        "re-embedding it into the stream's own rotating manifold planes",
        st.cyclic_colors(12))
    fig.write_html(OUT / "p4b_anatomy_b595_month.html", include_plotlyjs=True)
    print("b595 anatomy written", flush=True)


def cardinal_b2146():
    zm = np.load(DATA / "zoo_means_zoo4b.npz")
    zc = np.load(DATA / "zoo_codes_primary_zoo4b.npz")
    fr = np.load(DATA / "frames_pilot_primary.npz")
    geo = np.load(DATA / "geometry_pilot.npz")
    families = json.loads(str(zc["meta"]))["families"]
    fi = families.index("cardinal")
    bix_c = zc["blocks"].tolist().index(2146)
    bix_f = fr["blocks"].tolist().index(2146)
    m = zc["fam"] == fi
    cls = zc["cls"][m]
    z_tok = zc["z_sel"][m][:, bix_c].astype(np.float32)
    frames = fr["frames"][:, bix_f]
    C = 20
    zbar = np.stack([z_tok[cls == k].mean(0) if (cls == k).any()
                     else np.zeros(4) for k in range(C)])
    mean_stack = zm["cardinal_means"].transpose(1, 0, 2)  # [S, C, d]
    ghost_stack = np.stack([zbar @ frames[s] for s in range(len(SITES))])
    share = geo["share"][2146]
    fig = anatomy_figure(
        z_tok, cls, mean_stack, ghost_stack, FAMILY_LABELS["cardinal"],
        False, share,
        "Anatomy of a BSC block — b2146, the cardinal number-line "
        "(primary arm): one 4-dim code shared across depth,<br>per-site "
        "decoder frames re-embedding it into the stream's own manifold "
        "planes",
        seq_colors(20))
    fig.write_html(OUT / "p4b_anatomy_b2146_cardinal.html",
                   include_plotlyjs=True)
    print("b2146 anatomy written", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    month_b595()
    cardinal_b2146()
