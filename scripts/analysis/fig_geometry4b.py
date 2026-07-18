"""4b full-dictionary geometry figures (post-pilot analysis, 2026-07-18).

Consumes the pilot geometry npz (extract_geometry --runs, 8 sites), the
pilot evalstats npz (eval_activation_stats), the zoo class means, and
the named-block frame dumps; produces the PNG set:

  p4b_geo_share.png       per-block depth-energy share, primary vs
                          renorm vs scalar + depth-argmax histograms
  p4b_geo_rotation.png    adjacent-site frame rotation by depth (all
                          arms + shuffled-block null) and stream-vs-
                          frame tracking for the named captured blocks
  p4b_geo_dimensions.png  cross-site stacked spectral dimension (is a
                          block one rotating subspace or S fresh ones)
  p4b_geo_packing.png     co-activation Jaccard structure, clique
                          membership, code anisotropy per arm
  p4b_census.png          shape-space census: every sane-frequency
                          block by code PR / top-2 mass, named
                          manifolds highlighted

plus data/analysis/geometry4b_summary.json with the headline numbers.

  python scripts/analysis/fig_geometry4b.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr

import _style as st

st.apply()

DATA = Path("data/analysis")
OUT = Path("figures/pilot4b")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
S = len(SITES)
PAIRS = list(itertools.combinations(range(S), 2))
ADJ = [i for i, (a, b) in enumerate(PAIRS) if b - a == 1]
GAPS = [f"L{SITES[i]}-L{SITES[i+1]}" for i in range(S - 1)]
FREQ_BAND = (1e-4, 0.05)

ARMS = {  # geometry/evalstats name -> display
    "pilot": "primary",
    "pilot_seed1": "seed 1",
    "pilot_renorm": "renorm",
    "pilot_lr12e-4_destroyed": "destroyed (lr 1.2e-3)",
    "pilot_scalar": "scalar",
}
NAMED = {
    "primary": {
        "ring": {2982: "weekday", 1270: "month"},
        "line": {2146: "cardinal", 382: "ordinal"},
        "oddball": {1623: "astonishment"},
        "clique": [80, 242, 355, 651, 734, 1297, 1825, 2338, 2545, 2608,
                   2917, 3492, 3533, 3988, 257, 1103, 1694, 2927],
    },
    "renorm": {
        "ring": {595: "month", 862: "weekday"},
        "line": {3194: "cardinal", 1393: "ordinal"},
        "identity": {2407: "3/third", 1609: "4/fourth", 3234: "6/sixth",
                     1018: "7/seventh", 1808: "ten/tenth", 2820: "teens"},
        "oddball": {510: "Latin", 3227: "late-teens", 1219: "digit",
                    2324: "duration", 2987: "magnitude"},
        "clique": [416, 552, 819, 1825, 1987],
    },
}
# stream family -> (zoo means key, cyclic, [(arm, block)])
TRACK = [
    ("month", True, [("renorm", 595)]),
    ("weekday", True, [("renorm", 862)]),
    ("cardinal", False, [("renorm", 3194), ("primary", 2146)]),
    ("ordinal", False, [("primary", 382)]),
]


def geo(name: str):
    return np.load(DATA / f"geometry_{name}.npz")


def ev(name: str):
    return np.load(DATA / f"evalstats_{name}.npz")


def adj_rot(g, k: int = 2) -> np.ndarray:
    """[G, 7] mean top-k principal cosine per adjacent gap."""
    pc = g["pair_cos"].astype(np.float32)
    return pc[:, ADJ, : min(k, pc.shape[2])].mean(2)


def null_rot(g, k: int = 2) -> np.ndarray:
    nc = g["null_pair_cos"].astype(np.float32)
    return nc[:, ADJ, : min(k, nc.shape[2])].mean(2)


def stream_plane(X: np.ndarray, cyclic: bool) -> np.ndarray:
    X = X - X.mean(0)
    if cyclic:
        F = np.fft.fft(X, axis=0)
        u, v = np.real(F[1]), -np.imag(F[1])
        u /= np.linalg.norm(u)
        v -= u * (v @ u)
        v /= np.linalg.norm(v)
        return np.stack([u, v], 1)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:2].T


def plane_rot(planes: list[np.ndarray]) -> list[float]:
    return [float(np.mean(np.linalg.svd(planes[s].T @ planes[s + 1],
                                        compute_uv=False)[:2]))
            for s in range(len(planes) - 1)]


def frame_rot(frames: np.ndarray) -> list[float]:
    """frames [S, b, d] -> adjacent-gap mean top-2 principal cosine."""
    Q = [np.linalg.qr(frames[s].T)[0] for s in range(frames.shape[0])]
    return [float(np.mean(np.linalg.svd(Q[s].T @ Q[s + 1],
                                        compute_uv=False)[:2]))
            for s in range(len(Q) - 1)]


def cliques(z) -> tuple[list[list[int]], np.ndarray]:
    f, C = z["fire_count"], z["coact"]
    J = C / np.maximum(f[:, None] + f[None, :] - C, 1)
    np.fill_diagonal(J, 0)
    _, lab = connected_components(sp.csr_matrix(J > 0.9), directed=False)
    sizes = np.bincount(lab)
    return sorted((np.flatnonzero(lab == c).tolist()
                   for c in np.flatnonzero(sizes > 1)), key=len, reverse=True), J


def code_pr(z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals = np.linalg.eigvalsh(z["zz"].astype(np.float64))[:, ::-1]
    tot = np.maximum(evals.sum(1), 1e-30)
    pr = tot**2 / np.maximum((evals**2).sum(1), 1e-30)
    top2 = evals[:, :2].sum(1) / tot
    return pr, top2, z["fire_count"] / int(z["n_tokens"])


# ---------------------------------------------------------------- share --
def fig_share(summary):
    fig, axes = plt.subplots(
        2, 4, figsize=(13, 6.2),
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.32, "wspace": 0.3})
    tops = [("pilot", "primary"), ("pilot_renorm", "renorm"),
            ("pilot_scalar", "scalar")]
    for ax, (name, label) in zip(axes[0], tops):
        share = geo(name)["share"]
        order = np.argsort(share @ np.arange(S))
        im = ax.imshow(share[order], aspect="auto", cmap="Blues",
                       vmin=0, vmax=0.8, interpolation="nearest")
        ax.set_title(f"{label} (G={share.shape[0]})")
        ax.set_xticks(range(S), [f"L{s}" for s in SITES], fontsize=7)
        ax.set_ylabel("block (sorted by depth centroid)" if name == "pilot"
                      else None, fontsize=8)
        ax.grid(False)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.04, pad=0.02,
                 label="site energy share")
    axes[0, 3].axis("off")
    axes[0, 3].text(0, 0.75,
                    "Gram constraint:\nrows sum to 1.\n\n"
                    "Non-renorm arms park\nmost blocks' energy\npeak at L30;"
                    "\nrenorm spreads peaks\nacross all 8 sites.",
                    fontsize=8.5, color=st.INK2, va="top")
    bots = [("pilot", "primary"), ("pilot_renorm", "renorm"),
            ("pilot_scalar", "scalar"),
            ("pilot_lr12e-4_destroyed", "destroyed")]
    for ax, (name, label) in zip(axes[1], bots):
        share = geo(name)["share"]
        hist = np.bincount(share.argmax(1), minlength=S) / share.shape[0]
        ax.bar(range(S), hist, color=st.CAT[0] if "renorm" not in name
               else st.CAT[4], width=0.7)
        ax.set_xticks(range(S), [f"{s}" for s in SITES], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}: share-argmax", fontsize=8.5)
        summary["share_argmax_hist"][label] = hist.round(3).tolist()
    fig.suptitle("Where blocks put their decoder energy across depth "
                 "(gemma-3-4b pilot, 8 sites)", y=0.99)
    fig.savefig(OUT / "p4b_geo_share.png")
    plt.close(fig)


# ------------------------------------------------------------- rotation --
def fig_rotation(summary):
    zm = np.load(DATA / "zoo_means_zoo4b.npz")
    frames = {"primary": np.load(DATA / "frames_pilot_primary.npz"),
              "renorm": np.load(DATA / "frames_pilot_renorm.npz")}
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 2], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    x = np.arange(S - 1)
    for i, (name, label) in enumerate(ARMS.items()):
        g = geo(name)
        med = np.median(adj_rot(g), 0)
        ax.plot(x, med, marker="o", ms=3.5, color=st.CAT[i], label=label)
        summary["adjacent_rotation_median"][label] = med.round(3).tolist()
    nul = np.median(null_rot(geo("pilot")), 0)
    ax.plot(x, nul, ls=":", color=st.MUTED, label="shuffled-block null")
    ax.set_xticks(x, GAPS, rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("median adjacent-site principal cos (top-2)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Frame rotation by depth: the mid-stack shear zone")

    sub = gs[1].subgridspec(1, 5, wspace=0.08)
    panels = [(fam, cyc, arm, blk) for fam, cyc, pairs in TRACK
              for arm, blk in pairs]
    dict_med = {a: np.median(adj_rot(geo(n)), 0)
                for n, a in (("pilot", "primary"), ("pilot_renorm", "renorm"))}
    for j, (fam, cyc, arm, blk) in enumerate(panels):
        ax = fig.add_subplot(sub[j])
        M = zm[f"{fam}_means"].transpose(1, 0, 2)
        stream = plane_rot([stream_plane(M[s], cyc) for s in range(S)])
        fz = frames[arm]
        fr = frame_rot(fz["frames"][:, fz["blocks"].tolist().index(blk)])
        r = pearsonr(stream, fr).statistic
        ax.plot(x, stream, color=st.CAT[1], marker="o", ms=3,
                label="stream manifold")
        ax.plot(x, fr, color=st.CAT[0], marker="s", ms=3,
                label="block frames")
        ax.plot(x, dict_med[arm], ls="--", color=st.BASELINE, lw=1.4,
                label="dictionary median")
        ax.set_ylim(0.35, 1.0)
        ax.set_xticks([0, 3, 6], [GAPS[0], GAPS[3], GAPS[6]],
                      fontsize=6, rotation=30, ha="right")
        ax.set_title(f"{fam} b{blk} ({arm})\nr={r:.2f}", fontsize=8)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("adjacent-site cos")
            ax.legend(fontsize=6.5, loc="lower left")
        summary["stream_frame_tracking"][f"{fam}_b{blk}_{arm}"] = {
            "stream": np.round(stream, 3).tolist(),
            "frames": np.round(fr, 3).tolist(), "pearson_r": round(r, 3)}
    fig.suptitle("The captured blocks' frames rotate with the stream's own "
                 "manifold rotation (trough L18-L21)", y=1.02)
    fig.savefig(OUT / "p4b_geo_rotation.png")
    plt.close(fig)


# ----------------------------------------------------------- dimensions --
def fig_dimensions(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    bsc = [("pilot", "primary"), ("pilot_seed1", "seed 1"),
           ("pilot_renorm", "renorm"),
           ("pilot_lr12e-4_destroyed", "destroyed")]
    for i, (name, label) in enumerate(bsc):
        ssv = geo(name)["stacked_svals"]
        pr = (ssv**2).sum(1)**2 / ((ssv**2)**2).sum(1)
        xs = np.sort(pr)
        axes[0].plot(xs, np.linspace(0, 1, len(xs)), color=st.CAT[i],
                     label=f"{label} (med {np.median(pr):.1f})")
        summary["stacked_pr_median"][label] = round(float(np.median(pr)), 2)
    axes[0].axvline(4, color=st.MUTED, ls=":", lw=1.2)
    axes[0].text(4.1, 0.03, "b=4: one rigid\nshared subspace",
                 fontsize=7.5, color=st.MUTED)
    axes[0].axvline(32, color=st.MUTED, ls=":", lw=1.2)
    axes[0].text(31.8, 0.03, "S·b=32: fresh\nper site", fontsize=7.5,
                 color=st.MUTED, ha="right")
    axes[0].set_xlabel("participation ratio of the cross-site stacked "
                       "spectrum [S·b, d]")
    axes[0].set_ylabel("CDF over blocks")
    axes[0].legend(fontsize=8)
    axes[0].set_title("BSC blocks: one slowly-rotating subspace "
                      "(collapse decoheres it)")

    ssv = geo("pilot_scalar")["stacked_svals"]
    pr = (ssv**2).sum(1)**2 / ((ssv**2)**2).sum(1)
    xs = np.sort(pr)
    axes[1].plot(xs, np.linspace(0, 1, len(xs)), color=st.CAT[5],
                 label=f"scalar (med {np.median(pr):.1f})")
    summary["stacked_pr_median"]["scalar"] = round(float(np.median(pr)), 2)
    axes[1].axvline(1, color=st.MUTED, ls=":", lw=1.2)
    axes[1].axvline(8, color=st.MUTED, ls=":", lw=1.2)
    axes[1].text(1.05, 0.9, "rigid direction", fontsize=7.5, color=st.MUTED)
    axes[1].text(7.9, 0.9, "fresh per site", fontsize=7.5, color=st.MUTED,
                 ha="right")
    axes[1].set_xlabel("participation ratio of the stacked spectrum [S, d]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Scalar features rotate too")
    fig.suptitle("Cross-site spectral footprint per dictionary unit", y=1.0)
    fig.savefig(OUT / "p4b_geo_dimensions.png")
    plt.close(fig)


# -------------------------------------------------------------- packing --
def fig_packing(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    arms = [("pilot", "primary"), ("pilot_seed1", "seed 1"),
            ("pilot_renorm", "renorm")]
    for i, (name, label) in enumerate(arms):
        z = ev(name)
        comps, J = cliques(z)
        iu = np.triu_indices_from(J, 1)
        jj = J[iu]
        jj = jj[jj > 0.05]
        axes[0].hist(jj, bins=np.linspace(0.05, 1, 40), histtype="step",
                     color=st.CAT[i], label=label, log=True)
        summary["cliques"][label] = [len(c) for c in comps]
        g = geo(name)
        share = g["share"]
        pr, top2, freq = code_pr(z)
        xs = np.sort(pr)
        axes[1].plot(xs, np.linspace(0, 1, len(xs)), color=st.CAT[i],
                     label=f"{label} (frac PR<1.5: {(pr < 1.5).mean():.1%})")
        summary["code_pr_median"][label] = round(float(np.median(pr)), 2)
        # clique depth profile vs dictionary
        mem = [b for c in comps for b in c]
        if mem:
            axes[2].plot(range(S), share[mem].mean(0), color=st.CAT[i],
                         marker="o", ms=3, label=f"{label} clique blocks")
    axes[2].plot(range(S), geo("pilot")["share"].mean(0), ls="--",
                 color=st.BASELINE, label="primary all blocks")
    axes[2].plot(range(S), geo("pilot_renorm")["share"].mean(0), ls=":",
                 color=st.BASELINE, label="renorm all blocks")
    axes[0].set_xlabel("co-activation Jaccard (pairs > 0.05)")
    axes[0].set_ylabel("pair count (log)")
    axes[0].axvline(0.9, color=st.MUTED, ls=":", lw=1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Co-firing pairs: primary grows cliques,\n"
                      "renorm suppresses them")
    axes[1].set_xlabel("code participation ratio (of b=4)")
    axes[1].set_ylabel("CDF over blocks")
    axes[1].legend(fontsize=7.5)
    axes[1].set_title("Code anisotropy: renorm uses its\n4 dims more fully")
    axes[2].set_xticks(range(S), [f"L{s}" for s in SITES], fontsize=7)
    axes[2].set_ylabel("mean site energy share")
    axes[2].legend(fontsize=7.5)
    axes[2].set_title("Clique blocks' depth allocation")
    fig.suptitle("Packing structure on the 1M-token eval split "
                 "(J>0.9 components)", y=1.08)
    fig.savefig(OUT / "p4b_geo_packing.png")
    plt.close(fig)


# --------------------------------------------------------------- census --
def fig_census(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    style = {"ring": (st.CAT[0], "o"), "line": (st.CAT[1], "s"),
             "identity": (st.CAT[3], "D"), "oddball": (st.CAT[6], "^"),
             "clique": (st.CAT[7], "x")}
    for ax, (name, arm) in zip(axes, (("pilot", "primary"),
                                      ("pilot_renorm", "renorm"))):
        pr, top2, freq = code_pr(ev(name))
        sane = (freq >= FREQ_BAND[0]) & (freq <= FREQ_BAND[1])
        ax.scatter(pr[sane], top2[sane], s=4, color=st.BASELINE, alpha=0.35,
                   linewidths=0, label=f"all sane-freq ({sane.sum()})")
        for kind, entries in NAMED[arm].items():
            blocks = list(entries) if not isinstance(entries, dict) \
                else list(entries.keys())
            col, mk = style[kind]
            ax.scatter(pr[blocks], top2[blocks], s=28, color=col, marker=mk,
                       linewidths=1.2, label=kind)
            if isinstance(entries, dict):
                for b, lab in entries.items():
                    ax.annotate(f"b{b} {lab}", (pr[b], top2[b]), fontsize=6,
                                color=st.INK2, xytext=(3, 3),
                                textcoords="offset points")
        ax.set_xlabel("code participation ratio")
        ax.set_title(arm)
        summary["census_sane_blocks"][arm] = int(sane.sum())
    axes[0].set_ylabel("top-2 eigenvalue mass of the code second moment")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Shape-space census: named manifolds live on the planar "
                 "shelf (top-2 mass high, PR 2–3)", y=1.0)
    fig.savefig(OUT / "p4b_census.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {k: {} for k in
               ("share_argmax_hist", "adjacent_rotation_median",
                "stream_frame_tracking", "stacked_pr_median",
                "code_pr_median", "cliques", "census_sane_blocks")}
    fig_share(summary)
    print("share done", flush=True)
    fig_rotation(summary)
    print("rotation done", flush=True)
    fig_dimensions(summary)
    print("dimensions done", flush=True)
    fig_packing(summary)
    print("packing done", flush=True)
    fig_census(summary)
    print("census done", flush=True)
    (DATA / "geometry4b_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    print(f"-> {DATA / 'geometry4b_summary.json'}")


if __name__ == "__main__":
    main()
