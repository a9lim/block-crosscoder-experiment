"""Depth-availability + scalar-individuation tests (Phase 0.9.6 tier B).

Companion to tier_a_ring_tests.py, covering what that script skips:

  - per-site RAW whitened-stream ring (capitalized-only class means,
    top-plane adjacency, 20k-perm null) — the depth-availability read
    (Phase 0.5 expectation: the 4b ring band sits early);
  - the scalar arms: top-1-feature-per-class collapse count (interim
    §B individuation statistic) plus the population ring over the
    top-2C selective features' response space. Screen = family/overall
    mean-score ratio (the interim fallback screen — no background
    encode needed; validated against the z-score screen at 1b).

  python scripts/analysis/depth_scalar_tests.py \
      --out-root /data/runs/bcc-pilot4b \
      --acts /data/runs/bcc-analysis/calendar_probe_acts_pilot4b.npz \
      --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
      --tokenizer google/gemma-3-4b-pt --device cuda \
      --out /data/runs/bcc-analysis/depth_scalar_pilot4b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.store import Whitener

from probe_ring_consolidation import MONTHS, WEEKDAYS, perm_p, ring_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/runs/bcc-pilot4b"))
    ap.add_argument("--acts", type=Path)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--perms", type=int, default=20_000)
    ap.add_argument("--out", type=Path,
                    default=Path("/data/runs/bcc-analysis/depth_scalar_pilot4b.json"))
    args = ap.parse_args()

    za = np.load(args.acts)
    meta = json.loads(str(za["meta"]))
    sites = meta["sites"]
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

    results: dict = {"depth": {}, "scalar": {}}

    # --- depth availability: raw whitened stream, per site ---
    for family, names in (("month", MONTHS), ("weekday", WEEKDAYS)):
        fi = meta["families"].index(family)
        m = (fam == fi) & is_cap
        C = len(names)
        a = za["acts"][m]
        c = cls[m]
        per_site = []
        for s in range(len(sites)):
            cm = np.stack([a[c == k, s].mean(0) for k in range(C)])
            hits, top2 = ring_stats(cm)
            per_site.append({
                "site": sites[s], "ring_hits": hits, "ring_max": C,
                "top_plane_var": round(top2, 3),
                "perm_p": perm_p(hits, C, args.perms),
            })
        results["depth"][family] = per_site
        print(f"depth {family}: " + " ".join(
            f"L{e['site']}:{e['ring_hits']}/{C}" for e in per_site), flush=True)

    renorm_scalars = Whitener.load(args.store / "whitener.pt").site_rms_scalars()

    # --- scalar arms: individuation + population ring ---
    run_dirs = sorted(p for p in args.out_root.iterdir()
                      if (p / "latest.pt").exists() and (p / "report.json").exists())
    for root in run_dirs:
        report = json.loads((root / "report.json").read_text())
        if report["arm"] != "scalar":
            continue
        ckpt = torch.load(root / "latest.pt", map_location="cpu",
                          weights_only=False)
        mc = ckpt["model_cfg"]
        cfg = BSCConfig(
            n_blocks=mc["n_blocks"], block_dim=mc["block_dim"],
            n_sites=mc["n_sites"], d_model=mc["d_model"], k=mc["k"],
        )
        sca = BlockCrosscoder(cfg, device=args.device)
        sca.load_state_dict(ckpt["model"])
        sca.eval()

        ps = []
        with torch.no_grad():
            for i in range(0, acts.shape[0], 8192):
                xb = acts[i : i + 8192].to(args.device, torch.float32)
                if report.get("site_renorm"):
                    xb = xb * renorm_scalars.to(args.device).view(1, -1, 1)
                ps.append(sca.scores(sca.encode(xb)).cpu())
        p_lab = torch.cat(ps).numpy()

        entry: dict = {"config": mc, "lr": report.get("lr"),
                       "fvu_pooled": report["eval"]["topk"]["fvu_pooled"]}
        overall = p_lab[is_cap].mean(0) + 1e-9
        for family, names in (("month", MONTHS), ("weekday", WEEKDAYS)):
            fi = meta["families"].index(family)
            m = (fam == fi) & is_cap
            C = len(names)
            cm = np.stack([p_lab[m & (cls == k)].mean(0) for k in range(C)])
            top1 = cm.argmax(1)
            distinct = np.unique(top1)
            sel = cm.mean(0) / overall            # family/overall ratio screen
            top_feats = np.argsort(sel)[::-1][: 2 * C]
            hits, top2v = ring_stats(cm[:, top_feats])
            entry[family] = {
                "top1_map": {names[k]: int(top1[k]) for k in range(C)},
                "distinct_top1": int(len(distinct)),
                "population_feats": [int(f) for f in top_feats],
                "ring_hits": hits, "ring_max": C,
                "top_plane_var": round(top2v, 3),
                "perm_p": perm_p(hits, C, args.perms),
            }
        results["scalar"][root.name] = entry
        mo = entry["month"]
        print(
            f"{root.name}: month distinct top-1 {mo['distinct_top1']}/12, "
            f"population ring {mo['ring_hits']}/12 (p {mo['perm_p']:.2e}, "
            f"plane {mo['top_plane_var']:.0%})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
