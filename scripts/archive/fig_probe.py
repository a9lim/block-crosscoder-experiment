"""Figures: the calendar probe through the trained 1b crosscoders.

Centerpiece: block 23 of the 0.9.5 winner carries the month ring as one
block — token cloud + class means in the code's top plane, the per-depth
availability story, selectivity maps, and the honest weekday null.

  python scripts/analysis/fig_probe.py
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
SUMMARY: dict = {}


def load_probe():
    za = np.load(NPZ / "calendar_probe_acts.npz")
    meta = json.loads(str(za["meta"]))
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    cap = set()
    for family in ("month", "weekday"):
        for t in meta["label_maps"][family]:
            if tk.decode([int(t)]).strip()[:1].isupper():
                cap.add(int(t))
    is_cap = np.isin(za["token_ids"], list(cap))
    return za, meta, is_cap


def ring_stats(means: np.ndarray):
    X = means - means.mean(0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    ang = np.arctan2(P[:, 1], P[:, 0])
    C = len(ang)
    order = np.argsort(ang)
    pos = np.empty(C, int)
    pos[order] = np.arange(C)
    d = np.abs(np.diff(np.concatenate([pos, pos[:1]])))
    d = np.minimum(d, C - d)
    hits = int((d == 1).sum())
    top2 = float((S[:2] ** 2).sum() / (S**2).sum())
    return P, Vt, ang, hits, top2


def fig_block23_ring(za, is_cap) -> None:
    fam, cls = za["fam"], za["cls"]
    mask = (fam == 1) & is_cap
    z = np.load(NPZ / "calendar_probe_codes_winner.npz")
    zc = z["z_lab"].astype(np.float32)[mask][:, 23]  # [n, 4]
    c = cls[mask]
    means = np.stack([zc[c == k].mean(0) for k in range(12)])
    P, Vt, ang, hits, top2 = ring_stats(means)
    tokP = (zc - zc.mean(0)) @ Vt[:2].T
    colors = st.cyclic_colors(12)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 5.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for k in range(12):
        m = c == k
        ax.scatter(tokP[m, 0], tokP[m, 1], s=4, color=colors[k], alpha=0.18,
                   lw=0)
    loop = np.concatenate([P, P[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color=st.INK2, lw=1.1, zorder=4)
    for k in range(12):
        ax.scatter(*P[k], s=42, color=colors[k], edgecolor=st.INK, lw=0.8,
                   zorder=5)
        r = 1.16 + 0.06 * (k % 2)
        ax.annotate(st.MONTHS[k], xy=P[k] * r, fontsize=9, color=st.INK,
                    ha="center", va="center", zorder=6)
    ax.set_xlabel("code plane axis 1")
    ax.set_ylabel("code plane axis 2")
    ax.set_title(
        f"Block 23 code, top plane of class means — calendar order {hits}/12,\n"
        f"plane holds {top2:.0%} of class-mean variance (perm. p < 5e-5)"
    )
    ax.set_aspect("equal")

    # Right: month index vs unwrapped angle — monotone staircase.
    a = np.rad2deg(ang)
    a_un = np.unwrap(np.deg2rad(a[np.argsort(np.argsort(-a))]))  # keep raw
    ax2.plot(range(12), a, "o-", color=st.CAT[0], ms=6)
    for k in range(12):
        ax2.annotate(st.MONTHS[k], xy=(k, a[k]), xytext=(0, 8),
                     textcoords="offset points", fontsize=8, ha="center",
                     color=st.INK2)
    ax2.set_xlabel("calendar index")
    ax2.set_ylabel("angle in code plane (deg)")
    ax2.set_title("class angle vs calendar index — one clean cycle")
    fig.suptitle(
        "The trained BSC's month block (winner, G=1024, 16M tokens): "
        "one block = the month manifold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(OUT / "probe_block23_ring.png")
    plt.close(fig)
    SUMMARY["block23"] = {"hits": hits, "top2_var": top2,
                          "angles_deg": np.round(a, 1).tolist()}


def fig_ring_depth(za, is_cap) -> None:
    fam, cls = za["fam"], za["cls"]
    mask = (fam == 1) & is_cap
    acts = za["acts"][mask]
    c = cls[mask]
    fr = np.load(NPZ / "winner_block_frames.npz")
    D23 = fr["D23"]
    raw_hits, raw_v2, blk_hits, blk_v2 = [], [], [], []
    for s in range(6):
        m = np.stack([acts[c == k, s].mean(0) for k in range(12)])
        _, _, _, h, v = ring_stats(m)
        raw_hits.append(h)
        raw_v2.append(v)
        proj = acts[:, s] @ D23[s].T
        m = np.stack([proj[c == k].mean(0) for k in range(12)])
        _, _, _, h, v = ring_stats(m)
        blk_hits.append(h)
        blk_v2.append(v)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(6)
    ax.plot(x, raw_hits, "o-", color=st.MUTED,
            label="raw whitened stream, class-mean top plane")
    ax.plot(x, blk_hits, "o-", color=st.CAT[0],
            label="inside block 23's per-site decoder subspace")
    ax.axhline(12, color=st.GRID, lw=1, zorder=0)
    ax.set_xticks(x, [f"L{s}" for s in st.SITES])
    ax.set_ylim(0, 13)
    ax.set_xlabel("site (gemma-3-1b layer)")
    ax.set_ylabel("calendar-adjacent pairs in ring order (of 12)")
    ax.set_title(
        "The stream's month ring fades from the naive top plane after L17 —\n"
        "block 23's rotating frames keep it at every depth"
    )
    ax.legend(fontsize=8.5, loc="lower left")
    fig.savefig(OUT / "probe_ring_depth.png")
    plt.close(fig)
    SUMMARY["ring_by_depth"] = {
        "raw_hits": raw_hits, "block23_hits": blk_hits,
        "raw_top2": np.round(raw_v2, 2).tolist(),
        "block23_top2": np.round(blk_v2, 2).tolist(),
    }


def fig_selectivity(za, is_cap) -> None:
    fam, cls = za["fam"], za["cls"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    z = np.load(NPZ / "calendar_probe_codes_winner.npz")
    p_lab = z["p_lab"].astype(np.float32)
    p_bg = z["p_bg"].astype(np.float32)
    mu, sd = p_bg.mean(0), p_bg.std(0) + 1e-6
    for ax, fi, family, hi_blocks in (
        (axes[0], 1, "month", {23: st.CAT[0], 797: st.CAT[4], 366: st.CAT[2]}),
        (axes[1], 0, "weekday", {640: st.CAT[5]}),
    ):
        sel = (p_lab[fam == fi].mean(0) - mu) / sd
        order = np.argsort(sel)[::-1]
        ax.bar(range(40), sel[order[:40]], color=st.BASELINE, width=0.8)
        for rank, b in enumerate(order[:40]):
            if int(b) in hi_blocks:
                ax.bar([rank], [sel[b]], color=hi_blocks[int(b)], width=0.8)
                ax.annotate(f"block {b}", xy=(rank, sel[b]), xytext=(2, 3),
                            textcoords="offset points", fontsize=8.5,
                            color=hi_blocks[int(b)])
        ax.set_xlabel("block rank by selectivity")
        ax.set_ylabel("z-scored score vs background")
        ax.set_title(f"{family}-selective blocks (winner)")
    fig.suptitle("Family selectivity is concentrated in a handful of blocks", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "probe_selectivity.png")
    plt.close(fig)


def fig_weekday_null(za, is_cap) -> None:
    fam, cls = za["fam"], za["cls"]
    mask = fam == 0
    z = np.load(NPZ / "calendar_probe_codes_winner.npz")
    zc = z["z_lab"].astype(np.float32)[mask][:, 640]
    c = cls[mask]
    means = np.stack([zc[c == k].mean(0) for k in range(7)])
    P, Vt, ang, hits, top2 = ring_stats(means)
    tokP = (zc - zc.mean(0)) @ Vt[:2].T
    colors = st.cyclic_colors(7)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k in range(7):
        m = c == k
        ax.scatter(tokP[m, 0], tokP[m, 1], s=6, color=colors[k], alpha=0.3,
                   lw=0)
    loop = np.concatenate([P, P[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color=st.INK2, lw=1.0, zorder=4)
    for k in range(7):
        ax.scatter(*P[k], s=46, color=colors[k], edgecolor=st.INK, lw=0.8,
                   zorder=5)
        ax.annotate(st.WEEKDAYS[k], xy=P[k] * 1.18, fontsize=9, color=st.INK,
                    ha="center", va="center")
    ax.set_title(
        f"Weekday block 640: fires on 100% of weekday tokens,\n"
        f"but calendar order {hits}/7 (p≈0.43) — capture without ring geometry"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("code plane axis 1")
    ax.set_ylabel("code plane axis 2")
    fig.savefig(OUT / "probe_weekday_null.png")
    plt.close(fig)
    SUMMARY["weekday_block640"] = {"hits": hits, "top2_var": top2}


def fig_block23_3d(za, is_cap) -> None:
    import plotly.graph_objects as go

    fam, cls = za["fam"], za["cls"]
    mask = (fam == 1) & is_cap
    z = np.load(NPZ / "calendar_probe_codes_winner.npz")
    zc = z["z_lab"].astype(np.float32)[mask][:, 23]
    c = cls[mask]
    means = np.stack([zc[c == k].mean(0) for k in range(12)])
    X = zc - zc.mean(0)
    _, _, Vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    P3 = X @ Vt[:3].T
    M3 = (means - zc.mean(0)) @ Vt[:3].T
    colors = st.cyclic_colors(12)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for k in range(12):
        m = np.nonzero(c == k)[0]
        m = m[rng.permutation(len(m))[:250]]
        fig.add_trace(go.Scatter3d(
            x=P3[m, 0], y=P3[m, 1], z=P3[m, 2], mode="markers",
            marker=dict(size=2.2, color=colors[k], opacity=0.45),
            name=st.MONTHS[k], legendgroup=st.MONTHS[k],
            hovertext=[f"{st.MONTHS[k]}"] * len(m), hoverinfo="text",
        ))
    loop = np.concatenate([M3, M3[:1]])
    fig.add_trace(go.Scatter3d(
        x=loop[:, 0], y=loop[:, 1], z=loop[:, 2],
        mode="lines+markers+text",
        text=st.MONTHS + [""], textposition="top center",
        textfont=dict(size=11, color=st.INK),
        line=dict(color=st.INK2, width=4),
        marker=dict(size=5, color=colors + [colors[0]]),
        name="class means (calendar loop)",
    ))
    fig.update_layout(
        title="Block 23 of the trained BSC: the month manifold as one block "
              "(code space, class-mean PCA axes; 250 tokens/class shown)",
        height=640, width=980, paper_bgcolor=st.SURFACE,
        font=dict(family="system-ui, sans-serif", color=st.INK),
        scene=dict(
            xaxis_title="ring plane 1", yaxis_title="ring plane 2",
            zaxis_title="third axis",
        ),
    )
    fig.write_html(OUT / "probe_block23_3d.html", include_plotlyjs=True)


def main() -> None:
    za, meta, is_cap = load_probe()
    fig_block23_ring(za, is_cap)
    fig_ring_depth(za, is_cap)
    fig_selectivity(za, is_cap)
    fig_weekday_null(za, is_cap)
    fig_block23_3d(za, is_cap)
    (OUT / "probe_summary.json").write_text(json.dumps(SUMMARY, indent=2) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
