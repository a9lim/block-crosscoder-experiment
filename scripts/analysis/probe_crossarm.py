"""Cross-arm correspondence: do the primary and renorm dictionaries carve
the same manifold coordinates?

Open item from the pilot findings (cross-arm b1270/b595 span alignment,
b2146/b3194 for the cardinal line). Two levels, both on the paired zoo
probe (identical 82k tokens through both checkpoints):

1. Code maps (token level): for each family's best block per arm, fit a
   linear map primary-code -> renorm-code on a train split and score
   held-out R^2, against two permutation nulls — full shuffle (breaks
   token+class correspondence) and within-class shuffle (breaks token
   correspondence, keeps class structure). Phase-0.5 logic, now across
   *dictionaries* instead of across depths.

2. Span alignment (weight level, needs frames npz from
   dump_block_frames): per-site principal cosines between the two arms'
   block decoder subspaces, against the dictionary-wide null read from
   the geometry npz.

Winner-scoped: reads zoo_block_tests.json + zoo_codes_{winner,primary}.npz
from the winner analysis dir (probe_families with --runs winner=... primary=...).

  python scripts/analysis/probe_crossarm.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from winner import analysis_dir

DATA = analysis_dir()
CAP_FAMS = {"weekday", "month"}  # calendar families are cap-restricted
N_PERM = 2000
TRAIN_FRAC = 0.8


def r2(y: np.ndarray, yhat: np.ndarray, mu: np.ndarray) -> float:
    sse = float(((y - yhat) ** 2).sum())
    sst = float(((y - mu) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-30)


def code_map(x: np.ndarray, y: np.ndarray, cls: np.ndarray, rng) -> dict:
    """Linear map x->y (with intercept), held-out R^2 + permutation nulls."""
    n = len(x)
    idx = rng.permutation(n)
    n_tr = int(TRAIN_FRAC * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    X = np.concatenate([x, np.ones((n, 1))], 1)
    W, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    mu = y[tr].mean(0)
    yhat = X[te] @ W
    obs = r2(y[te], yhat, mu)

    null_full, null_cls = [], []
    y_te, c_te = y[te], cls[te]
    for _ in range(N_PERM):
        null_full.append(r2(y_te[rng.permutation(len(te))], yhat, mu))
        p = np.arange(len(te))
        for c in np.unique(c_te):
            m = np.flatnonzero(c_te == c)
            p[m] = m[rng.permutation(len(m))]
        null_cls.append(r2(y_te[p], yhat, mu))
    null_full, null_cls = np.array(null_full), np.array(null_cls)
    return {
        "n": int(n), "r2": round(obs, 4),
        "null_full_max": round(float(null_full.max()), 4),
        "p_full": round(float((null_full >= obs).mean() + 1 / N_PERM), 5),
        "null_cls_mean": round(float(null_cls.mean()), 4),
        "null_cls_q99": round(float(np.quantile(null_cls, 0.99)), 4),
        "p_cls": round(float((null_cls >= obs).mean() + 1 / N_PERM), 5),
    }


def span_alignment(fp: np.ndarray, fr: np.ndarray) -> list[list[float]]:
    """Per-site principal cosines between two [S, b, d] frame stacks."""
    out = []
    for s in range(fp.shape[0]):
        qa, _ = np.linalg.qr(fp[s].T)
        qb, _ = np.linalg.qr(fr[s].T)
        out.append(np.linalg.svd(qa.T @ qb, compute_uv=False).round(4).tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-primary", type=Path,
                    default=DATA / "frames_primary.npz")
    ap.add_argument("--frames-renorm", type=Path,
                    default=DATA / "frames_winner.npz")
    ap.add_argument("--geometry", type=Path,
                    default=DATA / "geometry_primary.npz",
                    help="dictionary-wide null scale for span alignment")
    ap.add_argument("--out", type=Path, default=DATA / "crossarm.json")
    args = ap.parse_args()

    zt = json.load(open(DATA / "zoo_block_tests.json"))
    zp = np.load(DATA / "zoo_codes_primary.npz")
    zr = np.load(DATA / "zoo_codes_winner.npz")
    assert np.array_equal(zp["token_ids"], zr["token_ids"])
    pb, rb = zp["blocks"].tolist(), zr["blocks"].tolist()
    fam, cls, is_cap = zp["fam"], zp["cls"], zp["is_cap"]
    fams = json.loads(str(zp["meta"]))["families"]
    rng = np.random.default_rng(0)

    report: dict = {"pairs": {}}
    for fi, name in enumerate(fams):
        bp = zt["primary"].get(name, {}).get("best_block")
        br = zt["winner"].get(name, {}).get("best_block")
        if bp is None or br is None or bp not in pb or br not in rb:
            continue
        m = fam == fi
        if name in CAP_FAMS:
            m &= is_cap.astype(bool)
        x = zp["z_sel"][m, pb.index(bp)].astype(np.float64)
        y = zr["z_sel"][m, rb.index(br)].astype(np.float64)
        entry = {
            "primary_block": int(bp), "renorm_block": int(br),
            "p_to_r": code_map(x, y, cls[m], rng),
            "r_to_p": code_map(y, x, cls[m], rng),
            "score_corr": round(float(np.corrcoef(
                np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))[0, 1]), 4),
        }
        report["pairs"][name] = entry
        print(name, f"b{bp}->b{br}", "R2", entry["p_to_r"]["r2"],
              "(cls-null q99", entry["p_to_r"]["null_cls_q99"], ")",
              flush=True)

    if args.frames_primary.exists() and args.frames_renorm.exists():
        fpz, frz = np.load(args.frames_primary), np.load(args.frames_renorm)
        fpb, frb = fpz["blocks"].tolist(), frz["blocks"].tolist()
        null_mean = None
        if args.geometry.exists():
            g = np.load(args.geometry)
            null_mean = round(float(g["null_pair_cos"].astype(np.float32).mean()), 4)
        report["span_null_pair_cos_mean"] = null_mean
        report["span"] = {}
        for name in report["pairs"]:
            bp = report["pairs"][name]["primary_block"]
            br = report["pairs"][name]["renorm_block"]
            if bp in fpb and br in frb:
                cos = span_alignment(fpz["frames"][:, fpb.index(bp)],
                                     frz["frames"][:, frb.index(br)])
                report["span"][name] = {
                    "per_site_cos": cos,
                    "mean_top2": round(float(np.mean(
                        [c[:2] for c in cos])), 4),
                }
                print(f"span {name} b{bp}~b{br} mean top-2 cos",
                      report["span"][name]["mean_top2"], flush=True)

    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
