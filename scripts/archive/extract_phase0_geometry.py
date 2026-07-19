"""Extract phase-0/0.5 ring geometry into one portable npz for figures.

Pulls, per gemma depth (9/17/22/29 @ 65k) and family (month/weekday):
  - supervised member features (top-1 per class) and their decoder rows
  - per-class -> feature assignment
  - member codes on the family-labeled tokens of the shared Pile stream
  - class ids of those tokens
  - cross-depth affine code maps (least squares on all labeled tokens;
    viz-grade — the audited held-out numbers live in cross_layer.json)

And for the GPT-2 positive control (layer 7, 16k): the top-affinity
cluster's member decoder rows + labeled codes for weekday/month/year.

Everything lands in /data/runs/bcc-analysis/phase0_geometry.npz, small
enough to scp to the Mac for 3D figure generation.

  python scripts/analysis/extract_phase0_geometry.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

RELEASE = "gemma-scope-2-4b-pt-res"
TOKENIZER = "google/gemma-3-4b-pt"
DEPTHS = {
    9: ("layer_9_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l9_65k_pile")),
    17: ("layer_17_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l17_65k_pile")),
    22: ("layer_22_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l22_65k_pile")),
    29: ("layer_29_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l29_65k_pile")),
}
FAMILIES = ("month", "weekday")
CONTROL_STORE = Path("/data/stores/bcc-phase0/gpt2_l7_owt")
CONTROL_RELEASE = "gpt2-small-res-jb"
CONTROL_SAE_ID = "blocks.7.hook_resid_pre"
CONTROL_FAMILIES = ("weekday", "month", "year")
OUT = Path("/data/runs/bcc-analysis/phase0_geometry.npz")


def main() -> None:
    from sae_lens import SAE
    from transformers import AutoTokenizer

    from block_crosscoder_experiment.phase0.harvest import CodeStore
    from block_crosscoder_experiment.phase0.labels import build_label_map, label_tokens

    out: dict[str, np.ndarray] = {}
    meta: dict = {"release": RELEASE, "depths": {}, "families": FAMILIES}

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    stores = {d: CodeStore(root) for d, (_, root) in DEPTHS.items()}
    ref_ids = stores[9].token_ids()

    label_ids = {}
    for family in FAMILIES:
        cls = label_tokens(ref_ids.long(), build_label_map(tok, family))
        labeled = (cls >= 0).nonzero().squeeze(1)
        label_ids[family] = (labeled, cls[labeled])
        out[f"{family}_token_cls"] = cls[labeled].numpy().astype(np.int16)

    codes_by_depth: dict[str, dict[int, torch.Tensor]] = {f: {} for f in FAMILIES}
    for depth, (sae_id, root) in DEPTHS.items():
        dec = SAE.from_pretrained(RELEASE, sae_id, device="cpu", dtype="float32").W_dec.detach()
        probe = json.loads((root / "target_run" / "supervised_ring.json").read_text())
        for family in FAMILIES:
            per_class = probe[family]["per_class"]
            top1 = {int(c): int(p[0][0]) for c, p in per_class.items() if p}
            members = sorted(set(top1.values()))
            m = torch.tensor(members, dtype=torch.long)
            labeled, _ = label_ids[family]
            z = stores[depth].select_members(m)[labeled]
            codes_by_depth[family][depth] = z
            out[f"d{depth}_{family}_members"] = np.array(members, dtype=np.int64)
            out[f"d{depth}_{family}_class_to_feat"] = np.array(
                [top1.get(c, -1) for c in range(max(top1) + 1)], dtype=np.int64
            )
            out[f"d{depth}_{family}_dec"] = dec[m].numpy().astype(np.float32)
            out[f"d{depth}_{family}_codes"] = z.numpy().astype(np.float32)
        meta["depths"][depth] = {"sae_id": sae_id, "d_model": int(dec.shape[1])}
        print(f"depth {depth}: extracted", flush=True)

    # Viz-grade cross-depth affine code maps (fit on all labeled tokens).
    for family in FAMILIES:
        ds = sorted(codes_by_depth[family])
        for i, a in enumerate(ds):
            for b in ds[i + 1 :]:
                za, zb = codes_by_depth[family][a], codes_by_depth[family][b]
                ones = torch.ones(za.shape[0], 1)
                m = torch.linalg.lstsq(
                    torch.cat([za, ones], 1), zb
                ).solution
                out[f"map_{family}_{a}_{b}"] = m.numpy().astype(np.float32)

    # GPT-2 positive control.
    dec = SAE.from_pretrained(CONTROL_RELEASE, CONTROL_SAE_ID, device="cpu").W_dec.detach().float()
    store = CodeStore(CONTROL_STORE)
    ids = store.token_ids()
    clusters = torch.load(CONTROL_STORE / "control_run" / "cluster_labels.pt",
                          weights_only=True)
    battery = json.loads(
        (CONTROL_STORE / "control_run" / "family_battery.json").read_text()
    )
    from transformers import AutoTokenizer as AT

    gtok = AT.from_pretrained("gpt2")
    for family in CONTROL_FAMILIES:
        top = battery[family]["top_affinity"][0]["cluster"]
        members = (clusters == top).nonzero().squeeze(1)
        cls = label_tokens(ids.long(), build_label_map(gtok, family))
        labeled = (cls >= 0).nonzero().squeeze(1)
        if labeled.numel() > 20000:  # year family is huge; subsample for viz
            keep = torch.randperm(labeled.numel(), generator=torch.Generator().manual_seed(0))[:20000]
            labeled = labeled[keep.sort().values]
        z = store.select_members(members)[labeled]
        out[f"ctl_{family}_members"] = members.numpy().astype(np.int64)
        out[f"ctl_{family}_dec"] = dec[members].numpy().astype(np.float32)
        out[f"ctl_{family}_codes"] = z.numpy().astype(np.float32)
        out[f"ctl_{family}_token_cls"] = cls[labeled].numpy().astype(np.int16)
        meta[f"ctl_{family}_cluster"] = int(top)
        print(f"control {family}: cluster {top}, {members.numel()} members, "
              f"{labeled.numel():,} labeled tokens", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, meta=json.dumps(meta), **out)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
