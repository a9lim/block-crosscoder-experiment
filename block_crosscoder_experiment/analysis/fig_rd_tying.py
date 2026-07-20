"""Tying-price figure: joint frontier vs 8x single-site systems, log-x.

The joint arms transmit one support set + one amplitude vector for all
8 sites; the single-site factorial cells pay per site. This plots the
q=4 R-D positions of the joint frontier (whitened-gauge pooled FVU;
the renorm arm in its own gauge, as everywhere) against the two
8-model system points from ``single_site_placement.json``, drawn in
both poolings (filled = whitened, open = renorm/uniform). Log-x makes
the ~7.9x rate cut legible as a fixed offset.

  bsc fig-rd-tying
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "evidence"
OUT = ROOT / "figures" / "summary" / "tying-rate.png"

ARM_COLORS = {
    "bsc": "#1f77b4",
    "renorm": "#2ca02c",
    "scalar": "#d62728",
    "bsf": "#9467bd",
    "sae": "#8c564b",
}
Q = "4"

JOINT = {  # file -> (arm, label)
    "f_bsc_lam0_k16": ("bsc", "k16"),
    "f_bsc_lam0_k32": ("bsc", "k32"),
    "f_bsc_lam0_k64": ("bsc", "k64"),
    "f_bsc_lam0_k32_renorm": ("renorm", "k32"),
    "f_scalar_lam0_k16": ("scalar", "k16"),
    "rd_scalar": ("scalar", "k32"),
    "f_scalar_lam0_k64": ("scalar", "k64"),
    # renorm k16/k64 appended by the extension chain when they land
    "f_renorm_lam0_k16": ("renorm", "k16"),
    "f_renorm_lam0_k64": ("renorm", "k64"),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    curves: dict[str, list[tuple[float, float]]] = {}
    for name, (arm, klabel) in JOINT.items():
        path = DATA / f"{name}.json"
        if not path.exists():
            continue
        pt = json.loads(path.read_text())["results"]["points"][Q]
        x, y = pt["rate_bits_per_token"], pt["fvu_pooled"]
        curves.setdefault(arm, []).append((x, y))
        ax.plot(x, y, "o", color=ARM_COLORS[arm], ms=6, zorder=3)
        ax.annotate(klabel, (x, y), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color=ARM_COLORS[arm])
    for arm, pts in curves.items():
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "--", color=ARM_COLORS[arm], lw=0.9, alpha=0.6,
                label=f"{arm} (joint)", zorder=2)

    placement = json.loads((DATA / "single_site_placement.json").read_text())
    for fam, joint_ref in (("bsf", "f_bsc_lam0_k32_renorm"), ("sae", "rd_scalar")):
        pt = placement[f"{fam}_system"]["points"][Q]
        x = pt["rate_bits_per_token"]
        c = ARM_COLORS[fam]
        ax.plot(x, pt["fvu_pooled_whitened"], "D", color=c, ms=8, zorder=3,
                label=f"{fam} 8x single-site (whitened pool)")
        ax.plot(x, pt["fvu_pooled_renorm"], "D", mfc="none", mec=c, ms=8,
                zorder=3, label=f"{fam} 8x single-site (renorm pool)")
        jp = json.loads((DATA / f"{joint_ref}.json").read_text())
        jpt = jp["results"]["points"][Q]
        y_arrow = (pt["fvu_pooled_renorm"] + jpt["fvu_pooled"]) / 2
        ax.annotate(
            "", (jpt["rate_bits_per_token"], y_arrow), (x, y_arrow),
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.7),
        )
        cut = x / jpt["rate_bits_per_token"]
        ax.text((x * jpt["rate_bits_per_token"]) ** 0.5, y_arrow,
                f"tying: {cut:.1f}x", fontsize=8, color=c,
                ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("rate (bits/token, support + amplitude, log scale)")
    ax.set_ylabel("pooled FVU (threshold mode, q=4)")
    ax.set_title("The price of untying — joint arms vs 8x independent "
                 "per-site models")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
