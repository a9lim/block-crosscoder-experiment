"""Family-general block capture tests over a generalized-probe npz.

The tier-A ring-test methodology extended to the manifold zoo
(0.9.6 analysis pass): for every family in the probe's meta —

  - consolidation: per-class top-1 block by class-mean selection score,
    distinct-count (the individuation statistic);
  - order: the best block's b-dim code-plane class means — cyclic
    families get the adjacency ring statistic, linear families the
    Spearman |rho| of class order along PC1 — each with a 20k-draw
    permutation null;
  - capitalized-only filtering applies to weekday/month exactly as in
    tier A (modal "may" pollution); other families keep all tokens
    (lowercase is their canonical form — known polysemy is worn).

Two GPU passes per run (the zoo acts don't fit encoded in host RAM):
pass 1 accumulates class-mean scores, pass 2 collects the selected
blocks' codes and writes a compact Mac-side export.

  python scripts/analysis/zoo_block_tests.py \
      --acts /data/runs/bcc-analysis/calendar_probe_acts_zoo4b.npz \
      --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
      --device cuda --tag _zoo4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.phase0.labels import FAMILIES
from block_crosscoder_experiment.store import Whitener

from tier_a_ring_tests import perm_p, ring_stats

CAP_FAMILIES = {"weekday", "month"}
CYCLIC = {"weekday", "month", "season", "compass"}
RUNS = {
    "renorm": "/data/runs/bcc-pilot4b/bsc_lam0.001_seed0_G4096_k32_renorm",
    "primary": "/data/runs/bcc-pilot4b/bsc_lam0.001_seed0_G4096_k32",
}


def spearman_p(order_vals: np.ndarray, n_perms: int, seed: int = 0):
    """|Spearman rho| of class order along a 1-D projection + perm null."""
    C = len(order_vals)
    ranks = np.argsort(np.argsort(order_vals))
    idx = np.arange(C)

    def rho(r):
        return abs(np.corrcoef(idx, r)[0, 1])

    obs = rho(ranks)
    rng = np.random.default_rng(seed)
    ge = sum(rho(rng.permutation(ranks)) >= obs for _ in range(n_perms))
    return float(obs), (1 + ge) / (n_perms + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--perms", type=int, default=20_000)
    ap.add_argument("--runs", nargs="*", default=None, metavar="NAME=PATH")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data/runs/bcc-analysis"))
    args = ap.parse_args()
    runs = RUNS if args.runs is None else dict(
        pair.split("=", 1) for pair in args.runs
    )

    za = np.load(args.acts)
    meta = json.loads(str(za["meta"]))
    families = meta["families"]
    fam, cls = za["fam"], za["cls"]
    tok_ids = za["token_ids"]
    acts = torch.from_numpy(za["acts"])  # [n, S, d]

    # means for the 3D stream stacks (tiny; Mac-side figures)
    np.savez_compressed(
        args.out_dir / f"zoo_means{args.tag}.npz",
        **{f"{f}_means": za[f"{f}_means"] for f in families},
        meta=json.dumps(meta),
    )

    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    cap = set()
    for family in CAP_FAMILIES & set(families):
        for t in meta["label_maps"][family]:
            if tk.decode([int(t)]).strip()[:1].isupper():
                cap.add(int(t))
    is_cap = np.isin(tok_ids, list(cap))

    def family_mask(fi: int, family: str) -> np.ndarray:
        m = fam == fi
        return m & is_cap if family in CAP_FAMILIES else m

    renorm_scalars = Whitener.load(args.store / "whitener.pt").site_rms_scalars()
    results: dict = {}
    for name, root in runs.items():
        root = Path(root)
        report = json.loads((root / "report.json").read_text())
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
        G = mc["n_blocks"]

        def encode_chunks():
            with torch.no_grad():
                for i in range(0, acts.shape[0], 8192):
                    xb = acts[i : i + 8192].to(args.device, torch.float32)
                    if report.get("site_renorm"):
                        xb = xb * renorm_scalars.to(args.device).view(1, -1, 1)
                    yield i, bsc.encode(xb)

        # pass 1: class-mean scores, never materializing [n, G]
        sums = {f: np.zeros((len(FAMILIES[f]), G), np.float64)
                for f in families}
        counts = {f: np.zeros(len(FAMILIES[f]), np.int64) for f in families}
        for i, z in encode_chunks():
            p = bsc.scores(z).cpu().numpy()
            for fi, family in enumerate(families):
                m = family_mask(fi, family)[i : i + p.shape[0]]
                c = cls[i : i + p.shape[0]][m]
                np.add.at(sums[family], c, p[m])
                np.add.at(counts[family], c, 1)

        entry: dict = {"fvu_pooled": report["eval"]["topk"]["fvu_pooled"]}
        best_blocks: set[int] = set()
        cms = {}
        for family in families:
            cm = sums[family] / np.maximum(counts[family][:, None], 1)
            cms[family] = cm
            top1 = cm.argmax(1)
            blocks, n = np.unique(top1, return_counts=True)
            best = int(blocks[n.argmax()])
            best_blocks.update(int(b) for b in top1)
            entry[family] = {
                "class_counts": counts[family].tolist(),
                "top1_map": {FAMILIES[family][k]: int(top1[k])
                             for k in range(len(top1))},
                "distinct_top1": int(len(blocks)),
                "best_block": best, "top1_claimed": int(n.max()),
            }

        # pass 2: codes for every top-1 block, order stats for the best
        sel = sorted(best_blocks)
        sel_t = torch.tensor(sel, device=args.device)
        z_sel = np.empty((acts.shape[0], len(sel), mc["block_dim"]), np.float16)
        for i, z in encode_chunks():
            z_sel[i : i + z.shape[0]] = z[:, sel_t].cpu().numpy()

        for fi, family in enumerate(families):
            e = entry[family]
            m = family_mask(fi, family)
            C = len(FAMILIES[family])
            zc = z_sel[m][:, sel.index(e["best_block"])].astype(np.float32)
            c = cls[m]
            means = np.stack([
                zc[c == k].mean(0) if (c == k).any() else np.zeros(zc.shape[1])
                for k in range(C)
            ])
            if family in CYCLIC:
                hits, top2 = ring_stats(means)
                e["order"] = {"kind": "ring", "hits": hits, "max": C,
                              "top_plane_var": round(top2, 3),
                              "perm_p": perm_p(hits, C, args.perms)}
            else:
                X = means - means.mean(0)
                _, _, Vt = np.linalg.svd(X, full_matrices=False)
                rho, p = spearman_p(X @ Vt[0], args.perms)
                e["order"] = {"kind": "line", "spearman": round(rho, 3),
                              "perm_p": p}
            o = e["order"]
            stat = (f"ring {o['hits']}/{o['max']}" if o["kind"] == "ring"
                    else f"|rho| {o['spearman']}")
            print(f"{name} {family}: best b{e['best_block']} "
                  f"top1 {e['top1_claimed']}/{len(FAMILIES[family])} "
                  f"distinct {e['distinct_top1']} {stat} "
                  f"(p {o['perm_p']:.2e})", flush=True)

        np.savez_compressed(
            args.out_dir / f"zoo_codes_{name}{args.tag}.npz",
            blocks=np.array(sel), z_sel=z_sel,
            fam=fam, cls=cls, token_ids=tok_ids, is_cap=is_cap,
            **{f"{f}_cm": cms[f].astype(np.float32) for f in families},
            meta=json.dumps({"run": str(root), "model_cfg": mc,
                             "families": families}),
        )
        results[name] = entry

    out = args.out_dir / f"zoo_block_tests{args.tag}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
