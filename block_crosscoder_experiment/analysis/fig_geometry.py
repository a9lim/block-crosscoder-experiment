"""Winner dictionary-geometry summary figures.

Structural geometry of the current winner (``data/winner.json``) and
its matched primary-gauge counterpart, from the extract_geometry /
eval_activation_stats / dump_block_frames artifacts in the winner
analysis dir:

  site-share.png      per-block depth-energy share, winner vs primary,
                      + depth-argmax histograms
  frame-rotation.png  adjacent-site frame rotation by depth (both arms +
                      shuffled-block null) and stream-vs-frame tracking
                      for the qualified showcase blocks
  effective-dimensions.png  cross-site stacked spectral dimension (is a block
                      one rotating subspace or S fresh ones)
  packing.png         co-activation Jaccard structure, clique membership,
                      code anisotropy per arm
  block-census.png    shape-space census: every sane-frequency block by
                      code PR / top-2 mass, qualified showcase manifolds
                      highlighted

plus ``geometry_summary.json`` beside the source artifacts. Block identities
always come from the current winner's derived showcase metadata.
"""

from __future__ import annotations

import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr

from . import style as st
from .artifacts import analysis_dir, load_showcase, load_winner, summary_dir

st.apply()

W = load_winner()
DATA = analysis_dir(W)
OUT = summary_dir()
SITES = W["sites"]
S = len(SITES)
PAIRS = list(itertools.combinations(range(S), 2))
ADJ = [i for i, (a, b) in enumerate(PAIRS) if b - a == 1]
GAPS = [f"L{SITES[i]}-L{SITES[i+1]}" for i in range(S - 1)]
FREQ_BAND = (1e-4, 0.05)
CYCLIC = {"weekday", "month", "season", "compass"}

ARMS = {"winner": "renorm (winner)", "primary": "primary"}


def geo(name: str):
    return np.load(DATA / f"geometry_{name}.npz")


def ev(name: str):
    return np.load(DATA / f"evalstats_{name}.npz")


def adj_rot(g, k: int = 2) -> np.ndarray:
    """[G, S-1] mean top-k principal cosine per adjacent gap."""
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


def showcase_by_arm(show: dict) -> dict[str, dict[str, dict]]:
    """arm -> {family: qualified per-arm entry (block, order, ...)}."""
    out: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    for family, e in show["families"].items():
        for arm, pe in e.get("arms", {}).items():
            if arm in out and pe["qualified"]:
                out[arm][family] = pe
    return out


# ---------------------------------------------------------------- share --
def fig_share(summary):
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 4.4), gridspec_kw={"wspace": 0.3})
    for ax, (name, label) in zip(axes[:2], ARMS.items()):
        share = geo(name)["share"]
        order = np.argsort(share @ np.arange(S))
        im = ax.imshow(share[order], aspect="auto", cmap="Blues",
                       vmin=0, vmax=0.8, interpolation="nearest")
        ax.set_title(f"{label} (G={share.shape[0]})")
        ax.set_xticks(range(S), [f"L{s}" for s in SITES], fontsize=7)
        ax.set_ylabel("block (sorted by depth centroid)"
                      if name == "winner" else None, fontsize=8)
        ax.grid(False)
        hist = np.bincount(share.argmax(1), minlength=S) / share.shape[0]
        summary["share_argmax_hist"][label] = hist.round(3).tolist()
    fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.02,
                 label="site energy share")
    ax = axes[2]
    for (name, label), color in zip(ARMS.items(), (st.CAT[4], st.CAT[0])):
        share = geo(name)["share"]
        hist = np.bincount(share.argmax(1), minlength=S) / share.shape[0]
        ax.bar(np.arange(S) + (0.2 if name == "primary" else -0.2), hist,
               color=color, width=0.38, label=label)
    ax.set_xticks(range(S), [f"{s}" for s in SITES], fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("share-argmax by depth", fontsize=9)
    fig.suptitle("Where blocks put their decoder energy across depth "
                 "(Gram constraint: rows sum to 1)", y=0.99)
    fig.savefig(OUT / "site-share.png")
    plt.close(fig)


# ------------------------------------------------------------- rotation --
def fig_rotation(summary, show):
    zm = np.load(DATA / "zoo_means.npz")
    frames = {a: np.load(DATA / f"frames_{a}.npz") for a in ARMS}
    by_arm = showcase_by_arm(show)
    panels = [(fam, fam in CYCLIC, arm, e["block"])
              for arm in ARMS for fam, e in by_arm[arm].items()
              if f"{fam}_means" in zm or fam == "month"]
    panels = panels[:5]

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 2], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    x = np.arange(S - 1)
    for i, (name, label) in enumerate(ARMS.items()):
        g = geo(name)
        med = np.median(adj_rot(g), 0)
        ax.plot(x, med, marker="o", ms=3.5, color=st.CAT[i], label=label)
        summary["adjacent_rotation_median"][label] = med.round(3).tolist()
    nul = np.median(null_rot(geo("winner")), 0)
    ax.plot(x, nul, ls=":", color=st.MUTED, label="shuffled-block null")
    ax.set_xticks(x, GAPS, rotation=45, fontsize=7, ha="right")
    ax.set_ylabel("median adjacent-site principal cos (top-2)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Frame rotation by depth")

    if panels:
        sub = gs[1].subgridspec(1, len(panels), wspace=0.08)
        dict_med = {a: np.median(adj_rot(geo(a)), 0) for a in ARMS}
        # cap-only month means: the zoo means' May class is 88% modal 'may'
        za = np.load(DATA / "zoo_activations.npz")
        act_fams = json.loads(str(za["meta"]))["families"]
        month_cap = None
        if "month" in act_fams:
            zc_any = np.load(DATA / "zoo_codes_winner.npz")
            mcap = (za["fam"] == act_fams.index("month")) & zc_any["is_cap"]
            a, c = za["acts"][mcap], za["cls"][mcap]
            month_cap = np.stack([a[c == k].mean(0) for k in range(12)], 1)
        for j, (fam, cyc, arm, blk) in enumerate(panels):
            ax = fig.add_subplot(sub[j])
            M = month_cap if fam == "month" and month_cap is not None \
                else zm[f"{fam}_means"].transpose(1, 0, 2)
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
            ax.set_title(f"{fam} b{blk} ({ARMS[arm]})\nr={r:.2f}", fontsize=8)
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("adjacent-site cos")
                ax.legend(fontsize=6.5, loc="lower left")
            summary["stream_frame_tracking"][f"{fam}_b{blk}_{arm}"] = {
                "stream": np.round(stream, 3).tolist(),
                "frames": np.round(fr, 3).tolist(), "pearson_r": round(r, 3)}
    fig.suptitle("Do the captured blocks' frames rotate with the stream's "
                 "own manifold rotation?", y=1.02)
    fig.savefig(OUT / "frame-rotation.png")
    plt.close(fig)


# ----------------------------------------------------------- dimensions --
def fig_dimensions(summary):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (name, label) in enumerate(ARMS.items()):
        ssv = geo(name)["stacked_svals"]
        pr = (ssv**2).sum(1)**2 / ((ssv**2)**2).sum(1)
        xs = np.sort(pr)
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=st.CAT[i],
                label=f"{label} (med {np.median(pr):.1f})")
        summary["stacked_pr_median"][label] = round(float(np.median(pr)), 2)
    b = W["block_dim"]
    ax.axvline(b, color=st.MUTED, ls=":", lw=1.2)
    ax.text(b + 0.1, 0.03, f"b={b}: one rigid\nshared subspace",
            fontsize=7.5, color=st.MUTED)
    ax.axvline(S * b, color=st.MUTED, ls=":", lw=1.2)
    ax.text(S * b - 0.2, 0.03, f"S·b={S*b}: fresh\nper site", fontsize=7.5,
            color=st.MUTED, ha="right")
    ax.set_xlabel("participation ratio of the cross-site stacked "
                  "spectrum [S·b, d]")
    ax.set_ylabel("CDF over blocks")
    ax.legend(fontsize=8)
    ax.set_title("BSC blocks: one slowly-rotating subspace across depth")
    fig.savefig(OUT / "effective-dimensions.png")
    plt.close(fig)


# -------------------------------------------------------------- packing --
def fig_packing(summary):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for i, (name, label) in enumerate(ARMS.items()):
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
        mem = [b for c in comps for b in c]
        if mem:
            axes[2].plot(range(S), share[mem].mean(0), color=st.CAT[i],
                         marker="o", ms=3, label=f"{label} clique blocks")
        axes[2].plot(range(S), share.mean(0),
                     ls="--" if name == "primary" else ":",
                     color=st.BASELINE, label=f"{label} all blocks")
    axes[0].set_xlabel("co-activation Jaccard (pairs > 0.05)")
    axes[0].set_ylabel("pair count (log)")
    axes[0].axvline(0.9, color=st.MUTED, ls=":", lw=1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Co-firing pairs by arm")
    axes[1].set_xlabel("code participation ratio (of b=4)")
    axes[1].set_ylabel("CDF over blocks")
    axes[1].legend(fontsize=7.5)
    axes[1].set_title("Code anisotropy")
    axes[2].set_xticks(range(S), [f"L{s}" for s in SITES], fontsize=7)
    axes[2].set_ylabel("mean site energy share")
    axes[2].legend(fontsize=7.5)
    axes[2].set_title("Clique blocks' depth allocation")
    fig.suptitle("Packing structure on the 1M-token eval split "
                 "(J>0.9 components)", y=1.08)
    fig.savefig(OUT / "packing.png")
    plt.close(fig)


# --------------------------------------------------------------- census --
def fig_census(summary, show):
    by_arm = showcase_by_arm(show)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    kind_style = {"ring": (st.CAT[0], "o"), "geo": (st.CAT[3], "D"),
                  "line": (st.CAT[1], "s")}
    for ax, (name, label) in zip(axes, ARMS.items()):
        pr, top2, freq = code_pr(ev(name))
        sane = (freq >= FREQ_BAND[0]) & (freq <= FREQ_BAND[1])
        ax.scatter(pr[sane], top2[sane], s=4, color=st.BASELINE, alpha=0.35,
                   linewidths=0, label=f"all sane-freq ({sane.sum()})")
        seen_kinds = set()
        for fam, e in by_arm[name].items():
            kind = e["order"]["kind"]
            col, mk = kind_style.get(kind, (st.CAT[6], "^"))
            ax.scatter(pr[e["block"]], top2[e["block"]], s=28, color=col,
                       marker=mk, linewidths=1.2,
                       label=kind if kind not in seen_kinds else None)
            seen_kinds.add(kind)
            ax.annotate(f"b{e['block']} {fam}", (pr[e["block"]],
                        top2[e["block"]]), fontsize=6, color=st.INK2,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("code participation ratio")
        ax.set_title(label)
        summary["census_sane_blocks"][label] = int(sane.sum())
    axes[0].set_ylabel("top-2 eigenvalue mass of the code second moment")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Shape-space census: qualified showcase manifolds in "
                 "code-shape space", y=1.0)
    fig.savefig(OUT / "block-census.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    show = load_showcase(W)
    summary = {k: {} for k in
               ("share_argmax_hist", "adjacent_rotation_median",
                "stream_frame_tracking", "stacked_pr_median",
                "code_pr_median", "cliques", "census_sane_blocks")}
    fig_share(summary)
    print("share done", flush=True)
    fig_rotation(summary, show)
    print("rotation done", flush=True)
    fig_dimensions(summary)
    print("dimensions done", flush=True)
    fig_packing(summary)
    print("packing done", flush=True)
    fig_census(summary, show)
    print("census done", flush=True)
    (DATA / "geometry_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    print(f"-> {DATA / 'geometry_summary.json'}")


if __name__ == "__main__":
    main()
