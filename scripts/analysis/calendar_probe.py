"""Exploratory calendar probe through the trained 0.9/0.9.5 crosscoders.

NOT a phase gate — an opportunistic H1 preview on the 1b rehearsal
artifacts (16M-token training, G=1024/4096; read soft). Question: do the
trained BSC blocks carry weekday/month structure at gemma-3-1b, and does
any block's b-dim code look ring-like?

Pipeline (jobe, CUDA):
  1. Stream fineweb-edu (pinned revision, first --skip-docs documents
     skipped so the probe slice is disjoint from the store's head),
     packed exactly like the 0.9 harvest (ctx 1024, BOS, positions 0/1
     dropped, concat-no-boundary).
  2. Label packed positions with the phase-0 single-token weekday/month
     maps; collect whitened activations at labeled positions plus an
     every-97th-position background reservoir.
  3. Encode labeled + background through each checkpoint: selection
     scores for all blocks; full code z (fp16) for labeled tokens.
  4. Save per-site class means for a first activation-space look.

  python scripts/analysis/calendar_probe.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.phase0.harvest import pack_token_rows
from block_crosscoder_experiment.phase0.labels import (FAMILIES, build_label_map, label_tokens)
from block_crosscoder_experiment.store import Whitener

MODEL = "google/gemma-3-1b-pt"
CORPUS = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
SITES = (7, 10, 13, 17, 20, 22)
CTX = 1024
DROP = 2
STORE = Path("/data/stores/bcc-phase09/gemma3_1b_6site_fineweb")
RUNS = {
    "winner": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_lr0.0012",
    "G4096_k32": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_G4096_k32",
    "scalar_winner": "/data/runs/bcc-phase095/scalar_lam0_seed0_lr0.0012",
    "renorm_lr3e-4": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_renorm",
}
BACKGROUND_STRIDE = 97
BACKGROUND_CAP = 60_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scan-tokens", type=int, default=8_000_000)
    ap.add_argument("--skip-docs", type=int, default=20_000)
    ap.add_argument("--batch-rows", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("/data/runs/bcc-analysis"))
    ap.add_argument(
        "--model", default=MODEL,
        help="0.9.6: google/gemma-3-4b-pt for the pilot store",
    )
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument(
        "--runs", nargs="*", default=None, metavar="NAME=PATH",
        help="override the default run map (name=path pairs)",
    )
    ap.add_argument(
        "--families", nargs="+", default=["weekday", "month"],
        help="label families from phase0.labels.FAMILIES; fam index in the "
        "saved npz follows this order",
    )
    ap.add_argument(
        "--tag", default="",
        help="suffix for output filenames (e.g. _pilot4b) so 1b artifacts "
        "are never clobbered",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = RUNS if args.runs is None else dict(
        pair.split("=", 1) for pair in args.runs
    )

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sae_lens import HookedSAETransformer

    whitener = Whitener.load(args.store / "whitener.pt")
    sites = whitener.sites  # site list rides with the store, not the script
    renorm_scalars = whitener.site_rms_scalars()

    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    d_model = int(model.cfg.d_model)
    if whitener.mean.shape != (len(sites), d_model):
        raise SystemExit("whitener shape does not match the model")
    hook_names = [f"blocks.{L}.hook_resid_post" for L in sites]
    stop_at = max(sites) + 1
    bos = model.tokenizer.bos_token_id

    label_maps = {
        fam: build_label_map(model.tokenizer, fam) for fam in args.families
    }
    corpus_sha = HfApi().dataset_info(CORPUS[0]).sha
    stream = load_dataset(
        CORPUS[0], name=CORPUS[1], split="train", streaming=True, revision=corpus_sha
    )

    def token_docs():
        for i, doc in enumerate(stream):
            if i < args.skip_docs:
                continue
            yield model.tokenizer.encode(doc["text"], add_special_tokens=False)

    tokens_per_row = CTX - DROP
    n_rows = -(-args.scan_tokens // tokens_per_row) + args.batch_rows + 2
    rows = pack_token_rows(token_docs(), ctx=CTX, bos_id=bos, n_rows=n_rows)

    w_gpu = whitener.W.to(args.device)
    mu_gpu = whitener.mean.to(args.device)

    lab_acts, lab_cls, lab_fam, lab_tok = [], [], [], []
    bg_acts = []
    scanned = 0
    bg_phase = 0

    @torch.no_grad()
    def process(toks: torch.Tensor) -> None:
        nonlocal scanned, bg_phase
        _, cache = model.run_with_cache(
            toks.to(args.device),
            names_filter=lambda name: name in hook_names,
            stop_at_layer=stop_at,
            return_type=None,
        )
        acts = torch.stack([cache[h] for h in hook_names], dim=2)  # [B,ctx,S,d]
        acts = acts[:, DROP:].reshape(-1, len(sites), d_model)
        ids = toks[:, DROP:].reshape(-1)
        xw = torch.einsum("sde,nse->nsd", w_gpu, acts.float() - mu_gpu)

        cls = torch.full_like(ids, -1)
        fam = torch.full_like(ids, -1)
        for fi, family in enumerate(args.families):
            c = label_tokens(ids.long(), label_maps[family])
            hit = c >= 0
            cls[hit] = c[hit]
            fam[hit] = fi
        hit = fam >= 0
        if int(hit.sum()):
            lab_acts.append(xw[hit].cpu())
            lab_cls.append(cls[hit].cpu())
            lab_fam.append(fam[hit].cpu())
            lab_tok.append(ids[hit].cpu())
        if sum(a.shape[0] for a in bg_acts) < BACKGROUND_CAP:
            sel = torch.arange(bg_phase, xw.shape[0], BACKGROUND_STRIDE)
            keep = sel[~hit[sel]]
            bg_acts.append(xw[keep].cpu())
            bg_phase = int((bg_phase + xw.shape[0]) % BACKGROUND_STRIDE)
        scanned += xw.shape[0]

    buf: list[torch.Tensor] = []
    for row in rows:
        buf.append(row)
        if len(buf) < args.batch_rows:
            continue
        process(torch.stack(buf))
        buf = []
        if scanned >= args.scan_tokens:
            break
        if scanned % 1_000_000 < args.batch_rows * tokens_per_row:
            n_lab = sum(a.shape[0] for a in lab_acts)
            print(f"  scanned {scanned:,} tokens, {n_lab:,} labeled", flush=True)

    acts = torch.cat(lab_acts) if lab_acts else torch.zeros(0, len(sites), d_model)
    cls = torch.cat(lab_cls)
    fam = torch.cat(lab_fam)
    tok_ids = torch.cat(lab_tok)
    bg = torch.cat(bg_acts)[:BACKGROUND_CAP]
    per_fam = ", ".join(
        f"{family} {int((fam == fi).sum()):,}"
        for fi, family in enumerate(args.families)
    )
    print(f"scan done: {scanned:,} tokens, {per_fam}, "
          f"background {bg.shape[0]:,}", flush=True)

    # Per-site class means (activation space, whitened coords).
    means = {}
    for fi, family in enumerate(args.families):
        n_cls = len(FAMILIES[family])
        m = torch.zeros(n_cls, len(sites), d_model)
        for cix in range(n_cls):
            sel = (fam == fi) & (cls == cix)
            if int(sel.sum()):
                m[cix] = acts[sel].mean(0)
        means[family] = m

    np.savez_compressed(
        args.out / f"calendar_probe_acts{args.tag}.npz",
        acts=acts.numpy().astype(np.float32),
        cls=cls.numpy(),
        fam=fam.numpy(),
        token_ids=tok_ids.numpy(),
        **{f"{family}_means": m.numpy() for family, m in means.items()},
        bg_mean=bg.mean(0).numpy(),
        bg_var=bg.var(0).numpy(),
        meta=json.dumps(
            {
                "model": args.model, "corpus": CORPUS, "corpus_revision": corpus_sha,
                "skip_docs": args.skip_docs, "scanned_tokens": scanned,
                "sites": list(sites), "whitener_hash": whitener.hash,
                "families": list(args.families),
                "label_maps": {
                    f: {str(k): v for k, v in m.items()}
                    for f, m in label_maps.items()
                },
            }
        ),
    )

    # Encode through each trained checkpoint.
    for name, root in runs.items():
        root = Path(root)
        if not (root / "latest.pt").exists():
            print(f"{name}: missing, skipped")
            continue
        ckpt = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
        report = json.loads((root / "report.json").read_text())
        mc = ckpt["model_cfg"]
        cfg = BSCConfig(
            n_blocks=mc["n_blocks"], block_dim=mc["block_dim"],
            n_sites=mc["n_sites"], d_model=mc["d_model"], k=mc["k"],
        )
        bsc = BlockCrosscoder(cfg, device=args.device)
        bsc.load_state_dict(ckpt["model"])
        bsc.eval()

        def encode_all(x: torch.Tensor, chunk: int = 16384):
            zs, ps = [], []
            with torch.no_grad():
                for i in range(0, x.shape[0], chunk):
                    xb = x[i : i + chunk].to(args.device, torch.float32)
                    if report.get("site_renorm"):
                        xb = xb * renorm_scalars.to(args.device).view(1, -1, 1)
                    z = bsc.encode(xb)
                    zs.append(z.half().cpu())
                    ps.append(bsc.scores(z).half().cpu())
            return torch.cat(zs), torch.cat(ps)

        z_lab, p_lab = encode_all(acts)
        _, p_bg = encode_all(bg)
        np.savez_compressed(
            args.out / f"calendar_probe_codes_{name}{args.tag}.npz",
            z_lab=z_lab.numpy(),
            p_lab=p_lab.numpy(),
            p_bg=p_bg.numpy(),
            theta=np.float32(float(bsc.theta)),
            meta=json.dumps({"run": str(root), "model_cfg": mc}),
        )
        print(f"{name}: encoded {z_lab.shape[0]:,} labeled + {p_bg.shape[0]:,} bg",
              flush=True)


if __name__ == "__main__":
    main()
