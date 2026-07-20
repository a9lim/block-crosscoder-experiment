"""Capture labeled residual-stream activations for the descriptive zoo.

This is not a phase gate. The zoo is burned descriptive evidence and is
regenerated only to characterize the promoted checkpoint. Selection and
confirmation use the separately sealed Phase-1 panel.

Pipeline (jobe, CUDA):
  1. Stream fineweb-edu (pinned revision, first --skip-docs documents
     skipped so the probe slice is disjoint from the store's head),
     packed exactly like the store (ctx 1024, BOS, positions 0/1
     dropped, concat-no-boundary).
  2. Label packed positions with the phase-0 single-token weekday/month
     maps; collect whitened activations at labeled positions plus an
     every-97th-position background reservoir.
  3. Save labeled activations and a background reservoir once. Downstream
     probes encode them through the promoted winner and its controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.discovery.harvest import pack_token_rows
from block_crosscoder_experiment.analysis.catalog import (
    ZOO_FAMILIES,
    build_label_map,
    label_tokens,
)
from block_crosscoder_experiment.store import Whitener

CORPUS = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
CTX = 1024
DROP = 2
BACKGROUND_STRIDE = 97
BACKGROUND_CAP = 60_000


def main() -> None:
    from .artifacts import analysis_dir, load_winner

    winner = load_winner()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scan-tokens", type=int, default=8_000_000)
    ap.add_argument("--skip-docs", type=int, default=20_000)
    ap.add_argument("--batch-rows", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument(
        "--per-class-cap", type=int, default=0,
        help="keep at most this many labeled tokens per (family, class); "
        "0 = unlimited. Needed for plentiful families (digits/cardinals): "
        "uncapped zoo scans OOM host RAM at the final concat",
    )
    ap.add_argument(
        "--families", nargs="+", default=list(ZOO_FAMILIES),
        help="label families from analysis.catalog.FAMILIES; fam index in the "
        "saved npz follows this order",
    )
    ap.add_argument(
        "--tag", default="",
        help="optional suffix for parallel diagnostic captures",
    )
    args = ap.parse_args()
    args.out = args.out or analysis_dir(winner)
    args.model = args.model or winner["model"]
    args.store = args.store or Path(winner["store"])
    args.out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sae_lens import HookedSAETransformer

    whitener = Whitener.load(args.store / "whitener.pt")
    sites = whitener.sites  # site list rides with the store, not the script

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
    kept = {}  # (fam, cls) -> kept count, for --per-class-cap

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
        hit_all = hit.clone()  # pre-cap mask: background must stay family-free
        if args.per_class_cap and int(hit.sum()):
            # first-N cap per (family, class); labeled counts per batch are
            # small, so the python loop is cheap
            for j in hit.nonzero(as_tuple=True)[0].tolist():
                key = (int(fam[j]), int(cls[j]))
                if kept.get(key, 0) >= args.per_class_cap:
                    hit[j] = False
                else:
                    kept[key] = kept.get(key, 0) + 1
        if int(hit.sum()):
            lab_acts.append(xw[hit].cpu())
            lab_cls.append(cls[hit].cpu())
            lab_fam.append(fam[hit].cpu())
            lab_tok.append(ids[hit].cpu())
        if sum(a.shape[0] for a in bg_acts) < BACKGROUND_CAP:
            sel = torch.arange(bg_phase, xw.shape[0], BACKGROUND_STRIDE)
            keep = sel[~hit_all[sel]]
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

    np.savez_compressed(
        args.out / f"zoo_activations{args.tag}.npz",
        acts=acts.numpy().astype(np.float32),
        cls=cls.numpy(),
        fam=fam.numpy(),
        token_ids=tok_ids.numpy(),
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

    print(f"-> {args.out / f'zoo_activations{args.tag}.npz'}", flush=True)


if __name__ == "__main__":
    main()
