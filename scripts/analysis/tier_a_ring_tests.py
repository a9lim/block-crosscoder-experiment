"""Tier-A consolidation ring tests (Phase 0.9.6, runbook decision map).

Encodes the SAVED calendar-probe activations (no re-scan) through every
tier-A checkpoint and replays the interim-findings ring methodology:

  - capitalized-only month/weekday labels (modal "may"/"march" pollution);
  - per-class top-1 / top-2 block maps by class-mean selection score
    (the consolidation statistic: 12/12 = one block claims every month);
  - ring test on the consensus block's b-dim code: class means, top
    plane, calendar-adjacent hit count, 20k-draw permutation null
    (p floors at 1/20001);
  - the weekday analogue (7 classes) as the honest null companion.

CPU by default so it can run beside GPU training. Output: one JSON +
printed table per run directory found under --out-root.

  python scripts/analysis/tier_a_ring_tests.py \
      --out-root /data/runs/bcc-phase096 \
      --acts /data/runs/bcc-analysis/calendar_probe_acts.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.store import Whitener

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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
    return hits, top2


def perm_p(obs_hits: int, n_classes: int, n_perms: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    ge = 0
    idx = np.arange(n_classes)
    for _ in range(n_perms):
        pos = rng.permutation(n_classes)
        d = np.abs(np.diff(np.concatenate([pos[idx], pos[idx[:1]]])))
        d = np.minimum(d, n_classes - d)
        if int((d == 1).sum()) >= obs_hits:
            ge += 1
    return (1 + ge) / (n_perms + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/runs/bcc-phase096"))
    ap.add_argument("--acts", type=Path,
                    default=Path("/data/runs/bcc-analysis/calendar_probe_acts.npz"))
    ap.add_argument("--store", type=Path,
                    default=Path("/data/stores/bcc-phase09/gemma3_1b_6site_fineweb"))
    ap.add_argument("--tokenizer", default="google/gemma-3-1b-pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--perms", type=int, default=20_000)
    ap.add_argument("--out", type=Path,
                    default=Path("/data/runs/bcc-analysis/tier_a_ring_tests.json"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these run-dir names")
    args = ap.parse_args()

    za = np.load(args.acts)
    meta = json.loads(str(za["meta"]))
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    cap = set()
    for family in ("month", "weekday"):
        for t in meta["label_maps"][family]:
            if tk.decode([int(t)]).strip()[:1].isupper():
                cap.add(int(t))
    is_cap = np.isin(za["token_ids"], list(cap))
    fam, cls = za["fam"], za["cls"]
    acts = torch.from_numpy(za["acts"])  # [n, S, d] float32, whitened

    renorm_scalars = Whitener.load(args.store / "whitener.pt").site_rms_scalars()

    results: dict[str, dict] = {}
    run_dirs = sorted(p for p in args.out_root.iterdir()
                      if (p / "latest.pt").exists() and (p / "report.json").exists())
    if args.only:
        run_dirs = [p for p in run_dirs if p.name in args.only]
    for root in run_dirs:
        report = json.loads((root / "report.json").read_text())
        if report["arm"] != "bsc":
            continue
        ckpt = torch.load(root / "latest.pt", map_location="cpu",
                          weights_only=False)
        mc = ckpt["model_cfg"]
        cfg = BSCConfig(
            n_blocks=mc["n_blocks"], block_dim=mc["block_dim"],
            n_sites=mc["n_sites"], d_model=mc["d_model"], k=mc["k"],
        )
        bsc = BlockCrosscoder(cfg, device=args.device)
        bsc.load_state_dict(ckpt["model"])
        bsc.eval()

        zs, ps = [], []
        with torch.no_grad():
            for i in range(0, acts.shape[0], 8192):
                xb = acts[i : i + 8192].to(args.device, torch.float32)
                if report.get("site_renorm"):
                    xb = xb * renorm_scalars.to(args.device).view(1, -1, 1)
                z = bsc.encode(xb)
                zs.append(z.cpu())
                ps.append(bsc.scores(z).cpu())
        z_lab = torch.cat(zs).numpy()
        p_lab = torch.cat(ps).numpy()

        entry: dict = {"config": mc, "epochs": report.get("epochs", 2),
                       "seed": report.get("seed", 0),
                       "lr": report.get("lr"),
                       "site_renorm": bool(report.get("site_renorm")),
                       "fvu_pooled": report["eval"]["topk"]["fvu_pooled"]}
        for fi, family, names in ((1, "month", MONTHS), (0, "weekday", WEEKDAYS)):
            m = (fam == fi) & is_cap
            C = len(names)
            cm = np.stack([p_lab[m & (cls == k)].mean(0) for k in range(C)])
            top1 = cm.argmax(1)                     # [C] block ids
            top2 = np.argsort(cm, 1)[:, -2:]        # [C, 2]
            blocks, counts = np.unique(top1, return_counts=True)
            best = int(blocks[counts.argmax()])
            top1_n = int(counts.max())
            top2_n = int((top2 == best).any(1).sum())
            zc = z_lab[m][:, best]
            c_ = cls[m]
            means = np.stack([zc[c_ == k].mean(0) for k in range(C)])
            hits, top2v = ring_stats(means)
            p = perm_p(hits, C, args.perms)
            entry[family] = {
                "best_block": best,
                "top1_claimed": top1_n, "top2_claimed": top2_n,
                "top1_map": {names[k]: int(top1[k]) for k in range(C)},
                "ring_hits": hits, "ring_max": C,
                "top_plane_var": round(top2v, 3),
                "perm_p": p,
            }
        results[root.name] = entry
        mo, wd = entry["month"], entry["weekday"]
        print(
            f"{root.name}: month best b{mo['best_block']} "
            f"top1 {mo['top1_claimed']}/12 top2 {mo['top2_claimed']}/12 "
            f"ring {mo['ring_hits']}/12 (p {mo['perm_p']:.2e}, "
            f"plane {mo['top_plane_var']:.0%}) | weekday b{wd['best_block']} "
            f"ring {wd['ring_hits']}/7 (p {wd['perm_p']:.2f})",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
