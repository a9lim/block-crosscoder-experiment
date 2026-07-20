"""Export compact per-block probe codes + decoder frames for figures.

For each BSC run: encode the saved calendar-probe acts, pick the blocks
that matter for calendar structure (union of the month/weekday top-1
maps and the top-8 by family/overall mean-score ratio per family), and
save a small npz per run: selected-block codes for every labeled token,
class-mean score matrices, the selected blocks' per-site decoder frames,
and the capitalization mask. Mac-side figure scripts consume these
instead of the 1.5 GB acts npz + checkpoints.

  python scripts/analysis/export_block_codes.py \
      --out-root /data/runs/bcc-pilot4b \
      --acts /data/runs/bcc-analysis/calendar_probe_acts_pilot4b.npz \
      --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
      --tokenizer google/gemma-3-4b-pt \
      --out-dir /data/runs/bcc-analysis --tag _pilot4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.store import Whitener


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path("/data/runs/bcc-pilot4b"))
    ap.add_argument("--acts", type=Path)
    ap.add_argument("--store", type=Path)
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-per-family", type=int, default=8)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data/runs/bcc-analysis"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", nargs="*", default=None)
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
    acts = torch.from_numpy(za["acts"])

    renorm_scalars = Whitener.load(args.store / "whitener.pt").site_rms_scalars()

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

        sel: set[int] = set()
        cms = {}
        overall = p_lab[is_cap].mean(0) + 1e-9
        for fi, C in ((meta["families"].index("month"), 12),
                      (meta["families"].index("weekday"), 7)):
            m = (fam == fi) & is_cap
            cm = np.stack([p_lab[m & (cls == k)].mean(0) for k in range(C)])
            cms[fi] = cm
            sel.update(int(b) for b in cm.argmax(1))
            ratio = cm.mean(0) / overall
            sel.update(int(b) for b in np.argsort(ratio)[::-1][: args.top_per_family])
        blocks = sorted(sel)

        D = ckpt["model"]["D"]  # [S, G, b, d] fp32 master
        out = args.out_dir / f"block_codes_{root.name}{args.tag}.npz"
        np.savez_compressed(
            out,
            blocks=np.array(blocks),
            z_sel=z_lab[:, blocks].astype(np.float16),
            month_cm=cms[1].astype(np.float32),
            weekday_cm=cms[0].astype(np.float32),
            frames=D[:, blocks].numpy().astype(np.float32),
            theta=np.float32(float(bsc.theta)),
            is_cap=is_cap,
            site_renorm=np.bool_(bool(report.get("site_renorm"))),
            meta=json.dumps({"run": str(root), "model_cfg": mc,
                             "sites": meta["sites"]}),
        )
        print(f"{root.name}: {len(blocks)} blocks -> {out}", flush=True)


if __name__ == "__main__":
    main()
