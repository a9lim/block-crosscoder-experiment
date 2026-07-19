"""H3 preview figure: the rate-distortion plane from codec payloads.

Globs ``rd_*.json`` (written by ``validate_rd_codec.py``), plots pooled
FVU vs bits/token with bootstrap CI bars, one color per arm, one marker
per q, dashed per-arm connection across rate points. Seeds are drawn
individually (never averaged) per the tranche-4 battery rule.

  python scripts/analysis/plot_rd_frontier.py \
      --inputs data/phase099/rd_*.json --out figures/phase099/rd_frontier.png
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARM_COLORS = {
    "bsc": "#1f77b4",
    "renorm": "#2ca02c",
    "scalar": "#d62728",
    "bsf": "#9467bd",
    "sae": "#8c564b",
}
Q_MARKERS = {4: "o", 6: "s", 8: "^"}


def arm_of(payload: dict, path: str) -> tuple[str, str]:
    """(arm, label) from the checkpoint path + renorm flag."""
    ckpt = payload.get("ckpt", path)
    renorm = payload.get("site_renorm", False)
    m = re.search(r"(bsc|scalar|bsf|sae)_lam([0-9.e-]+)_seed(\d+)", ckpt)
    if not m:
        return ("bsc", Path(path).stem)
    base, lam, seed = m.groups()
    arm = "renorm" if (base == "bsc" and renorm) else base
    km = re.search(r"_k(\d+)", ckpt)
    k = km.group(1) if km else "?"
    label = f"{arm} λ={lam} k={k} s{seed}"
    return (arm, label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--qs", type=int, nargs="*", default=[4, 6],
                    help="q levels to draw (q=8 adds nothing; default 4,6)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    paths = sorted(set(sum((glob.glob(p) for p in args.inputs), [])))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    seen_arms: dict[str, list] = {}

    for p in paths:
        payload = json.loads(Path(p).read_text())
        arm, label = arm_of(payload, p)
        color = ARM_COLORS.get(arm, "#7f7f7f")
        pts = []
        for qk, pt in sorted(payload["results"]["points"].items()):
            q = int(qk.lstrip("q"))
            if q not in args.qs:
                continue
            rate = pt["rate_bits_per_token"]
            fvu = pt["fvu_pooled"]
            lo, hi = pt["fvu_ci95"]
            ax.errorbar(rate, fvu, yerr=[[fvu - lo], [hi - fvu]],
                        fmt=Q_MARKERS.get(q, "x"), color=color, ms=6,
                        capsize=2, lw=1, zorder=3)
            pts.append((rate, fvu))
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, "--", color=color, lw=0.8, alpha=0.6, zorder=2)
            seen_arms.setdefault(arm, []).append((label, sorted(pts)))
            # label the cheapest point of each payload with its k/seed
            x0, y0 = sorted(pts)[0]
            short = re.sub(r"^[a-z]+ ", "", label)
            ax.annotate(short, (x0, y0), textcoords="offset points",
                        xytext=(4, -10), fontsize=7, color=color)

    handles = [Line2D([], [], color=ARM_COLORS[a], marker="o", ls="--",
                      label=a) for a in seen_arms]
    handles += [Line2D([], [], color="gray", marker=m, ls="",
                           label=f"q={q}") for q, m in Q_MARKERS.items()
                if q in args.qs]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xlabel("rate (bits/token, support + amplitude)")
    ax.set_ylabel("pooled FVU (threshold mode)")
    ax.set_title("R-D plane — pilot store, preregistered codec "
                 "(CIs: 979-sequence bootstrap)")
    ax.grid(alpha=0.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"-> {args.out}  ({len(paths)} payloads, "
          f"{sum(len(v) for v in seen_arms.values())} curves)")


if __name__ == "__main__":
    main()
