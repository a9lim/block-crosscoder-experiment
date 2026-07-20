"""Figures for the 0.9.6 tier-B 4b pilot (run on the Mac).

Inputs (mirrored to data/analysis/ from jobe):
  ring_tests_pilot4b.json, depth_scalar_pilot4b.json,
  block_codes_<run>_pilot4b.npz, calendar_probe_acts_pilot4b.npz,
  pilot4b_steps/<run>.jsonl + .report.json

Outputs figures/pilot4b/*.png plus computed frame-projection ring stats
in data/analysis/fig_pilot4b_summary.json (numbers cited in
docs/findings-phase096-pilot4b.md come from here).

  python scripts/analysis/fig_pilot4b.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _style as st
from tier_a_ring_tests import MONTHS, WEEKDAYS, perm_p, ring_stats

DATA = Path("data/analysis")
STEPS = DATA / "pilot4b_steps"
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
SITE_RAMP = [plt.cm.Blues(x) for x in np.linspace(0.35, 0.95, len(SITES))]

BSC_RUNS = {
    "primary 3e-4": "bsc_lam0.001_seed0_G4096_k32",
    "seed 1, 3e-4": "bsc_lam0.001_seed1_G4096_k32",
    "renorm 3e-4": "bsc_lam0.001_seed0_G4096_k32_renorm",
    "6e-4 (spiked)": "bsc_lam0.001_seed0_lr0.0006_G4096_k32",
    "6e-4 renorm (destr.)": "bsc_lam0.001_seed0_lr0.0006_G4096_k32_renorm",
    "1.2e-3 (destr.)": "bsc_lam0.001_seed0_lr0.0012_G4096_k32",
}
SCALAR_RUNS = {
    "scalar 3e-4": "scalar_lam0_seed0_G4096_k32",
    "scalar 6e-4 (spiked)": "scalar_lam0_seed0_lr0.0006_G4096_k32",
}
MONTH_WHEEL = [plt.cm.twilight(x) for x in np.linspace(0.04, 0.96, 12)]
DAY_WHEEL = [plt.cm.twilight(x) for x in np.linspace(0.04, 0.96, 7)]
SUMMARY: dict = {}


def codes_npz(run: str):
    return np.load(DATA / f"block_codes_{run}_pilot4b.npz")


def fig_ring_depth(depth) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(SITES))
    for family, names, color in (("month", MONTHS, st.CAT[0]),
                                 ("weekday", WEEKDAYS, st.CAT[3])):
        C = len(names)
        hits = [e["ring_hits"] for e in depth[family]]
        ax.plot(x, hits, "o-", color=color, label=f"{family} (max {C})")
        ax.axhline(C, color=color, lw=0.8, alpha=0.35, zorder=0)
    ax.set_xticks(x, [f"L{s}" for s in SITES])
    ax.set_ylim(0, 13)
    ax.set_xlabel("site (gemma-3-4b layer)")
    ax.set_ylabel("calendar-adjacent pairs in ring order")
    ax.set_title("Both calendar rings ride the 4b stream across the site list\n"
                 "(raw whitened class-mean top plane; month fades only late)")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "p4b_ring_depth.png")
    plt.close(fig)


def fig_capture_maps(rings, scal) -> None:
    rows = [(label, rings[run]["month"]["top1_map"])
            for label, run in BSC_RUNS.items() if run in rings]
    rows += [(label, scal["scalar"][run]["month"]["top1_map"])
             for label, run in SCALAR_RUNS.items() if run in scal["scalar"]]
    ids = [[m[mo] for mo in MONTHS] for _, m in rows]
    from collections import Counter

    counts = Counter(b for row in ids for b in row)
    notable = [b for b, c in counts.most_common(len(st.CAT)) if c >= 3]
    color_of = {b: st.CAT[i] for i, b in enumerate(notable)}

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(rows) + 1.6))
    for r, row in enumerate(ids):
        for c, b in enumerate(row):
            fc = color_of.get(b, st.BASELINE)
            ax.add_patch(plt.Rectangle((c, len(rows) - 1 - r), 0.96, 0.9,
                                       facecolor=fc, edgecolor=st.SURFACE))
            ax.text(c + 0.48, len(rows) - 0.55 - r, str(b), ha="center",
                    va="center", fontsize=7,
                    color="white" if b in color_of else st.INK2)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(12) + 0.48, MONTHS)
    ax.set_yticks(np.arange(len(rows)) + 0.45,
                  [label for label, _ in reversed(rows)])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Which unit claims each month (top-1 by class-mean score)\n"
                 "one shared color per recurring unit — capture is a lottery, "
                 "and the destroyed run's single block claims everything")
    fig.savefig(OUT / "p4b_capture_maps.png")
    plt.close(fig)


def fig_instability() -> None:
    show = {
        "3e-4 (clean)": ("bsc_lam0.001_seed0_G4096_k32", st.CAT[0]),
        "6e-4": ("bsc_lam0.001_seed0_lr0.0006_G4096_k32", st.CAT[3]),
        "6e-4 renorm": ("bsc_lam0.001_seed0_lr0.0006_G4096_k32_renorm",
                        st.CAT[5]),
        "1.2e-3": ("bsc_lam0.001_seed0_lr0.0012_G4096_k32", st.CAT[7]),
    }
    panels = [("rec", "recon loss", "log"),
              ("grad_norm", "main grad norm", "log"),
              ("grad_norm_aux", "AuxK grad norm", "log"),
              ("dead_frac_window", "dead fraction (window)", "linear")]
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 9.2), sharex=True)
    for label, (run, color) in show.items():
        rows = [json.loads(l) for l in (STEPS / f"{run}.jsonl").open()]
        steps = [r["step"] for r in rows]
        for ax, (key, _, _) in zip(axes, panels):
            y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            ax.plot(steps, np.where(y <= 0, np.nan, y) if key != panels[3][0]
                    else y, color=color, lw=1.4,
                    label=label if ax is axes[0] else None)
    for ax, (_, name, scale) in zip(axes, panels):
        ax.set_yscale(scale)
        ax.set_ylabel(name)
        ax.axvline(1000, color=st.MUTED, lw=0.8, ls=":", zorder=0)
    axes[0].annotate("warmup peak", xy=(1000, axes[0].get_ylim()[1]),
                     xytext=(4, -10), textcoords="offset points",
                     fontsize=8.5, color=st.MUTED)
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[-1].set_xlabel("step")
    axes[0].set_title("The warmup-peak instability: main-loss spike seeds it,\n"
                      "the AuxK revival cascade amplifies it — absent at 3e-4")
    fig.savefig(OUT / "p4b_instability.png")
    plt.close(fig)


def fig_allocation() -> None:
    show = {
        "BSC 3e-4": ("bsc_lam0.001_seed0_G4096_k32", st.CAT[0], "-"),
        "BSC seed 1": ("bsc_lam0.001_seed1_G4096_k32", st.CAT[0], "--"),
        "BSC renorm": ("bsc_lam0.001_seed0_G4096_k32_renorm", st.CAT[5], "-"),
        "scalar": ("scalar_lam0_seed0_G4096_k32", st.CAT[3], "-"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(SITES))
    for label, (run, color, ls) in show.items():
        rep = json.loads((STEPS / f"{run}.report.json").read_text())
        fvu = rep["eval"]["topk"]["fvu_per_site"]
        ax.plot(x, fvu, "o", ls=ls, color=color, label=label)
    ax.set_xticks(x, [f"L{s}" for s in SITES])
    ax.set_xlabel("site (gemma-3-4b layer)")
    ax.set_ylabel("eval FVU (top-k mode)")
    ax.set_title("The F7 allocation reversal replicates at 4b:\n"
                 "renorm spends its capacity shallow, baseline deep")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT / "p4b_allocation.png")
    plt.close(fig)


def _code_plane(ax, z_sel, block_ix, mask, c, names, wheel, title):
    zc = z_sel[mask][:, block_ix].astype(np.float32)
    C = len(names)
    means = np.stack([zc[c == k].mean(0) for k in range(C)])
    X = means - means.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    Pt = (zc - means.mean(0)) @ Vt[:2].T
    for k in range(C):
        pk = Pt[c == k]
        if len(pk) > 400:
            pk = pk[np.random.default_rng(0).choice(len(pk), 400, replace=False)]
        ax.scatter(pk[:, 0], pk[:, 1], s=5, color=wheel[k], alpha=0.18,
                   linewidths=0)
    loop = np.append(np.arange(C), 0)
    ax.plot(P[loop, 0], P[loop, 1], color=st.INK2, lw=1.0, alpha=0.7,
            zorder=3)
    for k in range(C):
        ax.scatter(*P[k], s=52, color=wheel[k], edgecolor=st.SURFACE,
                   linewidths=1.2, zorder=4)
        ax.annotate(names[k], P[k], xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, zorder=5)
    hits, top2 = ring_stats(means)
    p = perm_p(hits, C, 20_000)
    ax.set_title(f"{title}\nring {hits}/{C} (p {p:.1e}), "
                 f"top plane {top2:.0%} of class-mean var", fontsize=10)
    ax.set_xlabel("code-plane PC1")
    ax.set_ylabel("code-plane PC2")
    return hits, p


def fig_code_planes() -> None:
    z = codes_npz(BSC_RUNS["renorm 3e-4"])
    acts_meta = np.load(DATA / "calendar_probe_acts_pilot4b.npz")
    fam, cls = acts_meta["fam"], acts_meta["cls"]
    is_cap = z["is_cap"]
    blocks = z["blocks"].tolist()

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    m = (fam == 1) & is_cap
    _code_plane(ax, z["z_sel"], blocks.index(595), m, cls[m], MONTHS,
                MONTH_WHEEL, "renorm b595 — the month manifold at 4b\n"
                "(class means + labeled tokens in the code plane)")
    fig.savefig(OUT / "p4b_b595_ring.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    m = (fam == 0) & is_cap
    _code_plane(ax, z["z_sel"], blocks.index(862), m, cls[m], WEEKDAYS,
                DAY_WHEEL, "renorm b862 — the weekday ring at 4b")
    fig.savefig(OUT / "p4b_weekday_ring.png")
    plt.close(fig)


def fig_ring_in_frames(depth) -> None:
    z = codes_npz(BSC_RUNS["renorm 3e-4"])
    blocks = z["blocks"].tolist()
    za = np.load(DATA / "calendar_probe_acts_pilot4b.npz")
    fam, cls = za["fam"], za["cls"]
    is_cap = z["is_cap"]
    acts = za["acts"]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(SITES))
    SUMMARY["ring_in_frames"] = {}
    for family, names, block, color in (
            ("month", MONTHS, 595, st.CAT[0]),
            ("weekday", WEEKDAYS, 862, st.CAT[3])):
        C = len(names)
        m = (fam == (1 if family == "month" else 0)) & is_cap
        a, c = acts[m], cls[m]
        frames = z["frames"][:, blocks.index(block)]  # [S, b, d]
        raw = [e["ring_hits"] for e in depth[family]]
        fr_hits, fr_p = [], []
        for s in range(len(SITES)):
            proj = a[:, s] @ frames[s].T  # per-site scalar renorm cancels
            means = np.stack([proj[c == k].mean(0) for k in range(C)])
            h, _ = ring_stats(means)
            fr_hits.append(h)
            fr_p.append(perm_p(h, C, 20_000))
        ax.plot(x, raw, "o-", color=color, alpha=0.35,
                label=f"{family}: raw stream top plane")
        ax.plot(x, fr_hits, "o-", color=color,
                label=f"{family}: inside b{block}'s frames")
        SUMMARY["ring_in_frames"][family] = {
            "block": block, "raw_hits": raw, "frame_hits": fr_hits,
            "frame_p": fr_p}
    ax.set_xticks(x, [f"L{s}" for s in SITES])
    ax.set_ylim(0, 13)
    ax.set_xlabel("site (gemma-3-4b layer)")
    ax.set_ylabel("adjacent pairs in ring order")
    ax.set_title("Do the captured blocks' rotating frames carry the rings\n"
                 "at every depth (block-23-style), at pilot budget?")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.savefig(OUT / "p4b_ring_in_frames.png")
    plt.close(fig)


def fig_depth_planes() -> None:
    za = np.load(DATA / "calendar_probe_acts_pilot4b.npz")
    z = codes_npz(BSC_RUNS["renorm 3e-4"])
    fam, cls = za["fam"], za["cls"]
    m = (fam == 1) & z["is_cap"]
    a, c = za["acts"][m], cls[m]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.6))
    for s, ax in enumerate(axes.flat):
        means = np.stack([a[c == k, s].mean(0) for k in range(12)])
        X = means - means.mean(0)
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        P = X @ Vt[:2].T
        loop = np.append(np.arange(12), 0)
        ax.plot(P[loop, 0], P[loop, 1], color=st.INK2, lw=0.9, alpha=0.6)
        for k in range(12):
            ax.scatter(*P[k], s=34, color=MONTH_WHEEL[k],
                       edgecolor=st.SURFACE, linewidths=0.8, zorder=3)
        for k in (0, 3, 6, 9):
            ax.annotate(MONTHS[k], P[k], xytext=(3, 3),
                        textcoords="offset points", fontsize=7.5)
        hits, _ = ring_stats(means)
        ax.set_title(f"L{SITES[s]} — {hits}/12", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Month class means in each site's top plane (raw whitened "
                 "stream):\nthe ring is calendar-ordered nearly everywhere, "
                 "loosening at L27/L30", y=1.0)
    fig.savefig(OUT / "p4b_depth_planes.png")
    plt.close(fig)


def main() -> None:
    st.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    rings = json.loads((DATA / "ring_tests_pilot4b.json").read_text())
    scal = json.loads((DATA / "depth_scalar_pilot4b.json").read_text())
    depth = scal["depth"]

    fig_ring_depth(depth)
    fig_capture_maps(rings, scal)
    fig_instability()
    fig_allocation()
    fig_code_planes()
    fig_ring_in_frames(depth)
    fig_depth_planes()

    (DATA / "fig_pilot4b_summary.json").write_text(
        json.dumps(SUMMARY, indent=2) + "\n")
    print(json.dumps(SUMMARY, indent=2))
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
