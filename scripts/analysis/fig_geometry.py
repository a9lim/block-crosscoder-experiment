"""Figures: learned geometry of the trained 0.9/0.9.5 crosscoders.

Reads data/analysis/npz/, writes figures/interim/ (PNG + one plotly HTML)
and figures/interim/geometry_summary.json with the headline numbers.

  python scripts/analysis/fig_geometry.py
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
NPZ = ROOT / "data/analysis/npz"
OUT = ROOT / "figures/interim"
OUT.mkdir(parents=True, exist_ok=True)
LAYERS = np.array(st.SITES)
SUMMARY: dict = {}


def load(name: str):
    z = np.load(NPZ / f"geometry_{name}.npz")
    meta = json.loads(str(z["meta"]))
    return z, meta


def pr(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Participation ratio of squared values along axis."""
    p = x.astype(np.float64) ** 2
    return (p.sum(axis) ** 2) / np.maximum((p**2).sum(axis), 1e-30)


def fig_share_heatmap() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=False)
    for ax, name, label in (
        (axes[0], "winner", "baseline (winner, lr 1.2e-3)"),
        (axes[1], "renorm_lr3e-4", "site-renorm arm"),
    ):
        z, _ = load(name)
        share = z["share"]  # [G, S]
        centroid = share @ np.arange(6)
        order = np.argsort(centroid)
        im = ax.imshow(share[order].T, aspect="auto", cmap="Blues",
                       vmin=0, vmax=0.8, interpolation="nearest")
        ax.set_yticks(range(6), [f"L{s}" for s in st.SITES])
        ax.set_ylabel(label, fontsize=9)
        ax.grid(False)
    axes[1].set_xlabel("block (sorted by depth centroid of its energy share)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("decoder energy share", fontsize=9)
    axes[0].set_title(
        "Where each block spends its decoder energy across depth — "
        "renorm re-levels the budget"
    )
    fig.savefig(OUT / "geo_share_heatmap.png")
    plt.close(fig)
    for name in ("winner", "renorm_lr3e-4", "G4096_k32", "scalar_winner"):
        z, m = load(name)
        SUMMARY.setdefault("share_site_mean", {})[name] = (
            z["share"].mean(0).round(4).tolist()
        )


def fig_frame_rotation() -> None:
    fig, (axm, ax) = plt.subplots(
        1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [0.82, 1]}
    )
    # Left: site-by-site alignment matrix (winner) — the shear zone.
    z, _ = load("winner")
    mcw = z["pair_cos"].astype(np.float32).mean(-1)  # [G, P]
    mat = np.full((6, 6), np.nan)
    for p, (a, b_) in enumerate(z["pairs"]):
        mat[a, b_] = mat[b_, a] = np.median(mcw[:, p])
    np.fill_diagonal(mat, 1.0)
    im = axm.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            axm.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=8,
                     color=st.SURFACE if mat[i, j] > 0.55 else st.INK2)
    axm.set_xticks(range(6), [f"L{s}" for s in st.SITES])
    axm.set_yticks(range(6), [f"L{s}" for s in st.SITES])
    axm.grid(False)
    axm.set_title("median frame alignment, site × site (winner)\n"
                  "rotation concentrates at the L13→L17 crossing")
    SUMMARY["frame_alignment_matrix_winner"] = np.round(mat, 3).tolist()
    series = [
        ("winner", "BSC G=1024 (winner)", st.CAT[0], "o"),
        ("G4096_k32", "BSC G=4096", st.CAT[6], "s"),
        ("scalar_winner", "scalar G=4096", st.CAT[5], "^"),
    ]
    for name, label, color, marker in series:
        z, m = load(name)
        pc = z["pair_cos"].astype(np.float32)  # [G, P, b]
        pairs = z["pairs"]
        mc = pc.mean(-1)  # [G, P]
        gaps = LAYERS[pairs[:, 1]] - LAYERS[pairs[:, 0]]
        gx = sorted(set(gaps.tolist()))
        med = [np.median(mc[:, gaps == g]) for g in gx]
        lo = [np.percentile(mc[:, gaps == g], 10) for g in gx]
        hi = [np.percentile(mc[:, gaps == g], 90) for g in gx]
        ax.plot(gx, med, marker=marker, ms=5, color=color, label=label)
        ax.fill_between(gx, lo, hi, color=color, alpha=0.12, lw=0)
        SUMMARY.setdefault("frame_rotation_median_by_gap", {})[name] = dict(
            zip(map(int, gx), np.round(med, 3).tolist())
        )
    z, _ = load("winner")
    nc = z["null_pair_cos"].astype(np.float32).mean(-1)
    ax.axhline(float(np.median(nc)), color=st.MUTED, lw=1.2, ls=":")
    ax.annotate("shuffled-block null", xy=(0.985, float(np.median(nc))),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=st.MUTED,
                ha="right")
    ax.set_xlabel("layer gap between sites")
    ax.set_ylabel("mean principal cosine between block frames")
    ax.set_title("Frames rotate smoothly with depth — alignment decays with layer gap,\n"
                 "far above chance everywhere (median, 10–90% band over blocks)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5)
    fig.savefig(OUT / "geo_frame_rotation.png")
    plt.close(fig)


def fig_dimensions() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    # Left: cross-site stacked spectral dimension.
    for name, label, color in (
        ("winner", "BSC winner (λ=1e-3)", st.CAT[0]),
        ("lam0_at_winner", "BSC λ=0", st.CAT[4]),
        ("renorm_lr3e-4", "BSC renorm", st.CAT[5]),
        ("G4096_k32", "BSC G=4096", st.CAT[2]),
    ):
        z, _ = load(name)
        d = pr(z["stacked_svals"])
        xs = np.linspace(2, 24, 120)
        kde = np.array([
            np.mean(np.exp(-0.5 * ((d - x) / 0.45) ** 2)) for x in xs
        ]) / (0.45 * np.sqrt(2 * np.pi))
        ax1.plot(xs, kde, color=color, label=label, lw=1.8)
        SUMMARY.setdefault("stacked_pr_median", {})[name] = float(np.median(d))
    ax1.axvline(4, color=st.MUTED, lw=1, ls=":")
    ax1.axvline(24, color=st.MUTED, lw=1, ls=":")
    ax1.annotate("one shared\nsubspace (4)", xy=(4, ax1.get_ylim()[1]),
                 xytext=(4, -2), textcoords="offset points", fontsize=8,
                 color=st.MUTED, va="top")
    ax1.annotate("independent\nper site (24)", xy=(24, ax1.get_ylim()[1]),
                 xytext=(-4, -2), textcoords="offset points", fontsize=8,
                 color=st.MUTED, va="top", ha="right")
    ax1.set_xlabel("effective dimension of the 6-site decoder stack (b=4)")
    ax1.set_ylabel("density over blocks")
    ax1.set_title("Blocks reuse most of their subspace across depth")
    ax1.legend(fontsize=8)

    # Right: scalar-arm cross-site direction spread (cap 6).
    for name, label, color in (
        ("scalar_winner", "scalar winner", st.CAT[5]),
        ("scalar_base", "scalar lr 3e-4", st.CAT[3]),
    ):
        z, _ = load(name)
        d = pr(z["stacked_svals"])
        xs = np.linspace(1, 6, 120)
        kde = np.array([
            np.mean(np.exp(-0.5 * ((d - x) / 0.12) ** 2)) for x in xs
        ]) / (0.12 * np.sqrt(2 * np.pi))
        ax2.plot(xs, kde, color=color, label=label, lw=1.8)
        SUMMARY.setdefault("stacked_pr_median", {})[name] = float(np.median(d))
    ax2.axvline(1, color=st.MUTED, lw=1, ls=":")
    ax2.annotate("same direction\nat every site (1)", xy=(1, ax2.get_ylim()[1]),
                 xytext=(4, -2), textcoords="offset points", fontsize=8,
                 color=st.MUTED, va="top")
    ax2.set_xlabel("effective dimension of the 6 per-site directions (b=1)")
    ax2.set_title("Scalar features rotate too — median ~1.6 directions")
    ax2.legend(fontsize=8)
    fig.suptitle("Cross-site spectral footprint of each dictionary unit", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "geo_dimensions.png")
    plt.close(fig)


def fig_freq_l0() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, label, color in (
        ("winner", "BSC G=1024 k=16", st.CAT[0]),
        ("G4096_k32", "BSC G=4096 k=32", st.CAT[2]),
        ("scalar_winner", "scalar G=4096 k=64", st.CAT[5]),
    ):
        z = np.load(NPZ / f"evalstats_{name}.npz")
        n = int(z["n_tokens"])
        freq = np.sort(z["fire_count"] / n)[::-1]
        ax1.plot(np.arange(1, len(freq) + 1), freq, color=color, label=label,
                 lw=1.8)
        h = z["l0_hist"]
        ax2.plot(np.arange(len(h)), h / h.sum(), color=color, lw=1.8,
                 label=label)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("block rank by firing frequency")
    ax1.set_ylabel("firing frequency (eval, threshold mode)")
    ax1.set_title("No dead blocks anywhere; G=4096 grows a rare tail")
    ax1.legend(fontsize=8.5)
    ax2.set_xlabel("active blocks per token (threshold mode)")
    ax2.set_ylabel("fraction of tokens")
    ax2.set_xlim(0, 130)
    ax2.set_title("Per-token L0 around the calibrated targets (16 / 32 / 64)")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "geo_freq_l0.png")
    plt.close(fig)


def fig_code_anisotropy() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, label, color in (
        ("winner", "BSC G=1024", st.CAT[0]),
        ("G4096_k32", "BSC G=4096", st.CAT[2]),
    ):
        z = np.load(NPZ / f"evalstats_{name}.npz")
        ev = np.linalg.eigvalsh(z["zz"].astype(np.float64))
        d = pr(np.sqrt(np.clip(ev, 0, None)))
        ax.hist(d, bins=np.linspace(1, 4, 46), density=True, alpha=0.55,
                color=color, label=label)
        SUMMARY.setdefault("code_pr", {})[name] = {
            "median": float(np.median(d)),
            "frac_below_1p5": float((d < 1.5).mean()),
        }
    ax.set_xlabel("effective dimension of the block's code (eval second moment)")
    ax.set_ylabel("density over blocks")
    ax.set_title("Codes are genuinely multi-dimensional — and drift scalar-ward at G=4096")
    ax.legend(fontsize=8.5)
    fig.savefig(OUT / "geo_code_anisotropy.png")
    plt.close(fig)


def fig_share_3d() -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    zw, _ = load("winner")
    zr, _ = load("renorm_lr3e-4")
    both = np.concatenate([zw["share"], zr["share"]], axis=0)
    mu = both.mean(0)
    u, s, vt = np.linalg.svd(both - mu, full_matrices=False)
    proj = (both - mu) @ vt[:3].T
    pw, prj = proj[:1024], proj[1024:]

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("baseline (winner)", "site-renorm arm"),
    )
    for col, (P, share) in enumerate(((pw, zw["share"]), (prj, zr["share"])), 1):
        peak = share.argmax(1)
        for si in range(6):
            m = peak == si
            if not m.any():
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=P[m, 0], y=P[m, 1], z=P[m, 2],
                    mode="markers",
                    marker=dict(size=2.6, color=st.BLUE_RAMP[si]),
                    name=f"peak L{st.SITES[si]}",
                    legendgroup=f"L{st.SITES[si]}",
                    showlegend=(col == 1),
                    text=[
                        f"block {i}<br>share {np.round(share[i], 2).tolist()}"
                        for i in np.nonzero(m)[0]
                    ],
                    hoverinfo="text",
                ),
                row=1, col=col,
            )
    fig.update_layout(
        title="Depth-allocation geometry of every block (PCA of 6-site energy-share profiles, shared axes)",
        height=560, width=1060,
        paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(OUT / "geo_share_3d.html", include_plotlyjs=True)


def main() -> None:
    fig_share_heatmap()
    fig_frame_rotation()
    fig_dimensions()
    fig_freq_l0()
    fig_code_anisotropy()
    fig_share_3d()
    (OUT / "geometry_summary.json").write_text(
        json.dumps(SUMMARY, indent=2) + "\n"
    )
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
