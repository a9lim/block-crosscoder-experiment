"""Figures: 0.9.5 calibration landscape + store whitener spectra.

Reads data/analysis/{phase095,phase09}/  and writes figures/interim/.

  python scripts/analysis/fig_calibration.py
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
P95 = ROOT / "data/analysis/phase095"
OUT = ROOT / "figures/interim"
OUT.mkdir(parents=True, exist_ok=True)

BSC, SCALAR = st.CAT[0], st.CAT[5]  # blue / orange


def load_reports() -> list[dict]:
    out = []
    for rj in P95.glob("*/report.json"):
        r = json.loads(rj.read_text())
        r["_name"] = rj.parent.name
        out.append(r)
    return out


def fig_lr_response(reports: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for arm, color in (("bsc", BSC), ("scalar", SCALAR)):
        for sched, ls, m in (("cosine", "-", "o"), ("linear_fifth", "--", "s")):
            pts = sorted(
                (r["lr"], r["eval"]["topk"]["fvu_pooled"])
                for r in reports
                if r["arm"] == arm and r.get("schedule") == sched
                and r.get("seed") == 0 and not r.get("site_renorm")
                and r.get("encoder_wd", 0) == 0
                and r["model_cfg"]["n_blocks"] == (1024 if arm == "bsc" else 4096)
                and r["model_cfg"]["block_dim"] == (4 if arm == "bsc" else 1)
            )
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, ls, marker=m, ms=5, color=color,
                    label=f"{'BSC' if arm=='bsc' else 'scalar'} · {sched}")
        # seed-1 runs as open markers
        s1 = [
            (r["lr"], r["eval"]["topk"]["fvu_pooled"])
            for r in reports
            if r["arm"] == arm and r.get("seed") == 1
        ]
        if s1:
            x, y = zip(*s1)
            ax.plot(x, y, "o", ms=8, mfc="none", mec=color, mew=1.4,
                    label=f"{'BSC' if arm=='bsc' else 'scalar'} · seed 1")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("pooled FVU (eval, topk)")
    ax.set_title("0.9.5 calibration: lr response on the 1b store — optimum 1.2e-3, cliff at 2.4e-3")
    ax.axvline(1.2e-3, color=st.BASELINE, lw=1, zorder=0)
    ax.annotate("ratified\nPhase-1 lr", xy=(1.2e-3, ax.get_ylim()[1]),
                xytext=(4, -6), textcoords="offset points",
                fontsize=8, color=st.INK2, va="top")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "cal_lr_response.png")
    plt.close(fig)


def fig_training_curves() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    runs = [
        ("bsc_lam0.001_seed0_lr0.0001", "1e-4"),
        ("bsc_lam0.001_seed0_lr0.0002", "2e-4"),
        ("bsc_lam0.001_seed0", "3e-4"),
        ("bsc_lam0.001_seed0_lr0.0006", "6e-4"),
        ("bsc_lam0.001_seed0_lr0.0012", "1.2e-3"),
        ("bsc_lam0.001_seed0_lr0.0024", "2.4e-3"),
    ]
    for (name, label), color in zip(runs, st.BLUE_RAMP):
        path = P95 / name / "steps.jsonl"
        if not path.exists():
            continue
        steps, rec = [], []
        for line in path.read_text().splitlines():
            d = json.loads(line)
            steps.append(d["step"])
            rec.append(d["rec"])
        color = st.CAT[7] if label == "2.4e-3" else color
        ax.plot(steps, rec, color=color, lw=1.6,
                label=f"lr {label}", zorder=3 if label == "2.4e-3" else 2)
    ax.annotate("2.4e-3 spike, step ~1.5k", xy=(1500, 4), xytext=(24, 6),
                textcoords="offset points", fontsize=9, color=st.CAT[7])
    ax.set_yscale("log")
    ax.set_xlabel("step (batch 1024 tokens)")
    ax.set_ylabel("reconstruction loss (whitened MSE)")
    ax.set_title("BSC cosine lr ladder — 2.4e-3 destabilizes mid-run and never fully recovers")
    ax.set_xlim(0, 4100)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.savefig(OUT / "cal_training_curves.png")
    plt.close(fig)


def fig_renorm_allocation(reports: list[dict]) -> None:
    base = next(r for r in reports if r["_name"] == "bsc_lam0.001_seed0")
    ren = next(r for r in reports if r["_name"] == "bsc_lam0.001_seed0_renorm")
    fvu_b = base["eval"]["topk"]["fvu_per_site"]
    fvu_r = ren["eval"]["topk"]["fvu_per_site"]
    gb = np.load(ROOT / "data/analysis/npz/geometry_base_lr3e-4.npz")
    gr = np.load(ROOT / "data/analysis/npz/geometry_renorm_lr3e-4.npz")
    sh_b, sh_r = gb["share"].mean(0), gr["share"].mean(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    x = np.arange(6)
    for ax, yb, yr, ylab, title in (
        (ax1, fvu_b, fvu_r, "per-site FVU",
         "site renorm reverses the FVU allocation"),
        (ax2, sh_b, sh_r, "mean decoder energy share",
         "…and re-levels the dictionary's depth budget"),
    ):
        ax.plot(x, yb, "o-", color=st.CAT[0], label="baseline (λ=1e-3, lr 3e-4)")
        ax.plot(x, yr, "o-", color=st.CAT[5], label="site-renorm arm")
        for xi, (yb_i, yr_i) in enumerate(zip(yb, yr)):
            ax.plot([xi, xi], [yb_i, yr_i], color=st.GRID, lw=1, zorder=0)
        ax.set_xticks(x, [f"L{s}" for s in st.SITES])
        ax.set_xlabel("site (gemma-3-1b layer)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    ax1.legend(fontsize=8, loc="lower left")
    fig.suptitle("F7 site-renorm arm vs baseline (matched config)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "cal_renorm_allocation.png")
    plt.close(fig)


def fig_whitener() -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    from block_crosscoder_experiment.store import Whitener

    w = Whitener.load(ROOT / "data/analysis/phase09/whitener.pt")
    eigs = w.eigenvalues.double()  # [S, d] ascending, of Σ+λI
    lam = w.ridge.double()
    d = eigs.shape[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    idx = np.arange(1, d + 1)
    for s, layer in enumerate(st.SITES):
        e = eigs[s].flip(0).numpy()  # descending
        raw = np.clip(e - float(lam[s]), 1e-12, None)  # spectrum of Σ
        ax1.plot(idx, raw, color=st.BLUE_RAMP[s], lw=1.5)
        ax1.annotate(f"L{layer}", xy=(idx[-1], raw[-1]), xytext=(4, 0),
                     textcoords="offset points", fontsize=8,
                     color=st.BLUE_RAMP[s], va="center")
        retained = (e - float(lam[s])) / e
        ax2.plot(idx, retained, color=st.BLUE_RAMP[s], lw=1.5)
        pr = float((raw.sum() ** 2) / (raw**2).sum())
        ax2.annotate(f"L{layer} · PR {pr:.0f}", xy=(0.02, 0.30 - 0.045 * s),
                     xycoords="axes fraction", fontsize=8,
                     color=st.BLUE_RAMP[s])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("eigenvalue rank")
    ax1.set_ylabel("covariance eigenvalue")
    ax1.set_title("residual-stream covariance spectra by depth")
    ax2.set_xscale("log")
    ax2.set_xlabel("eigenvalue rank")
    ax2.set_ylabel("retained fraction after shrinkage")
    ax2.set_title("shrinkage whitener: retained variance (e−λ)/e")
    scalars = w.site_rms_scalars().numpy()
    ax2.annotate(
        "F7 renorm scalars\n(1/√mean-retained):\n"
        + ", ".join(f"{v:.2f}" for v in scalars),
        xy=(0.98, 0.97), xycoords="axes fraction", fontsize=8.5,
        color=st.INK2, va="top", ha="right",
    )
    fig.suptitle("gemma-3-1b whitener (13M-token fit, 6 sites)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "whitener_spectra.png")
    plt.close(fig)


def fig_dead_dynamics() -> None:
    path = P95 / "bsc_lam0.001_seed0_G4096_k32" / "steps.jsonl"
    steps, dead, ema = [], [], []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        steps.append(d["step"])
        dead.append(d["dead_frac_window"])
        ema.append(d.get("ema_min_score", np.nan))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(steps, np.array(dead) * 100, color=st.CAT[0])
    ax.set_xlabel("step")
    ax.set_ylabel("dead-block fraction in window (%)")
    ax.set_title("G=4096, k=32 stress arm: dead dynamics engage — final mortality 0.098%")
    fig.savefig(OUT / "cal_dead_dynamics.png")
    plt.close(fig)


def main() -> None:
    reports = load_reports()
    fig_lr_response(reports)
    fig_training_curves()
    fig_renorm_allocation(reports)
    fig_whitener()
    fig_dead_dynamics()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
