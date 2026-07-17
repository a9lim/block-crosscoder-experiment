"""Harvest the D13 4b pilot store: gemma-3-4b, 8 sites, FineWeb-Edu.

The design-mandated >=3M-token exact-config pilot (design v2.3.2, D13)
fits on jobe's existing /data disk (~290 GB stored vs ~600 GB usable
above the ShardWriter floor) — only the 53M-token production store needs
the 4 TB NVMe. Same machinery as the 0.9 harvest, production model and
site band:

  sites = 8 evenly spaced in the 25-90% band of 34 layers
        = (9, 12, 15, 18, 21, 24, 27, 30)
  (design line: "8 in 25-90% band (~ layers 9-30, resolved at harvest,
  frozen in config)". Step-3 spacing lands exactly on the band edges and
  brackets the Phase-0 probed depths 9/17/22/29. The production harvest
  should freeze the SAME list — flagged for a9 ratification in the
  0.9.6 runbook.)

Defaults: 2M whitener slice (accumulated, never stored; halves/quarters
stability printed — if 4b needs the production 5M slice, this run is the
evidence), 2M calibration + 1M eval + 4M train stored whitened bf16
(4 x 8 x 2560 x 2B ~ 287 GB), first 100k calibration tokens also raw.
fp16 forbidden throughout; free-space abort before every shard write.

  nohup python -u scripts/harvest_pilot4b_store.py \
      > /data/runs/bcc-pilot4b/harvest.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL = "google/gemma-3-4b-pt"
CORPUS = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
SITES = (9, 12, 15, 18, 21, 24, 27, 30)  # 25-90% band of 34, step 3
CTX = 1024
DROP_POSITIONS = 2  # BOS + position 1
RAW_VALIDATION_TOKENS = 100_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whitener-tokens", type=int, default=2_000_000)
    parser.add_argument("--calib-tokens", type=int, default=2_000_000)
    parser.add_argument("--eval-tokens", type=int, default=1_000_000)
    parser.add_argument("--train-tokens", type=int, default=4_000_000)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument(
        "--out", type=Path,
        default=Path("/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(
        f"config: model={MODEL} sites={SITES} corpus={CORPUS} ctx={CTX} "
        f"whitener={args.whitener_tokens:,} calib={args.calib_tokens:,} "
        f"eval={args.eval_tokens:,} train={args.train_tokens:,} out={args.out}",
        flush=True,
    )

    import torch
    from datasets import load_dataset
    from sae_lens import HookedSAETransformer

    from block_crosscoder_experiment.phase0.harvest import pack_token_rows
    from block_crosscoder_experiment.store import (
        ShardWriter,
        WhitenerAccumulator,
    )

    model = HookedSAETransformer.from_pretrained_no_processing(
        MODEL, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    d_model = int(model.cfg.d_model)
    n_layers = int(model.cfg.n_layers)
    if max(SITES) >= n_layers:
        raise SystemExit(f"site list {SITES} exceeds n_layers={n_layers}")
    print(f"model loaded: d_model={d_model} n_layers={n_layers}", flush=True)
    hook_names = [f"blocks.{L}.hook_resid_post" for L in SITES]
    stop_at = max(SITES) + 1
    bos = model.tokenizer.bos_token_id

    from huggingface_hub import HfApi

    corpus_sha = HfApi().dataset_info(CORPUS[0]).sha
    stream = load_dataset(
        CORPUS[0], name=CORPUS[1], split="train", streaming=True,
        revision=corpus_sha,
    )

    def token_docs():
        for doc in stream:
            yield model.tokenizer.encode(doc["text"], add_special_tokens=False)

    tokens_per_row = CTX - DROP_POSITIONS
    total_tokens = (
        args.whitener_tokens + args.calib_tokens + args.eval_tokens + args.train_tokens
    )
    n_rows = -(-total_tokens // tokens_per_row) + 5 * args.batch_rows + 2
    rows = pack_token_rows(token_docs(), ctx=CTX, bos_id=bos, n_rows=n_rows)

    corpus_meta = {
        "model": MODEL,
        "corpus": CORPUS[0],
        "corpus_config": CORPUS[1],
        "corpus_revision": corpus_sha,
        "corpus_split": "train",
        "context_size": CTX,
        "prepend_bos": True,
        "dropped_positions": DROP_POSITIONS,
        "sites": list(SITES),
        "hook_names": hook_names,
        "model_dtype": "bfloat16",
        "pack_convention": "concat-no-boundary",
        "pilot": "D13 exact-config pilot store (pre-NVMe, /data resident)",
    }

    @torch.no_grad()
    def act_batches():
        buf: list[torch.Tensor] = []
        for row in rows:
            buf.append(row)
            if len(buf) < args.batch_rows:
                continue
            toks = torch.stack(buf).to(args.device)
            buf = []
            _, cache = model.run_with_cache(
                toks,
                names_filter=lambda name: name in hook_names,
                stop_at_layer=stop_at,
                return_type=None,
            )
            acts = torch.stack(
                [cache[h] for h in hook_names], dim=2
            )  # [B, ctx, S, d] bf16
            yield acts[:, DROP_POSITIONS:].reshape(-1, len(SITES), d_model)

    batches = act_batches()

    # ---- Stage 1: whitener slice (accumulated, never stored) -------------
    print("=== stage 1: whitener slice ===", flush=True)
    quarters = [
        WhitenerAccumulator(len(SITES), d_model, device=args.device) for _ in range(4)
    ]
    seen = 0
    t0 = time.time()
    while seen < args.whitener_tokens:
        acts = next(batches)
        take = min(acts.shape[0], args.whitener_tokens - seen)
        q = min(3, seen * 4 // args.whitener_tokens)
        quarters[q].update(acts[:take].float())
        seen += take
        if seen % 500_000 < acts.shape[0]:
            print(f"  whitener {seen:,}/{args.whitener_tokens:,} tokens "
                  f"({seen / (time.time() - t0):,.0f} tok/s)", flush=True)

    halves = [quarters[0].merge(quarters[1]), quarters[2].merge(quarters[3])]
    full = halves[0].merge(halves[1])
    whitener = full.finalize(sites=SITES, meta=corpus_meta)
    for label, accs in (("half", halves), ("quarter", quarters)):
        for i, acc in enumerate(accs):
            w_i = acc.finalize(sites=SITES, meta=corpus_meta)
            rel = (
                (w_i.W - whitener.W).flatten(1).norm(dim=1)
                / whitener.W.flatten(1).norm(dim=1)
            )
            print(
                f"  stability {label} {i}: rel ΔW per site "
                f"{[round(float(r), 4) for r in rel]}",
                flush=True,
            )
    args.out.mkdir(parents=True, exist_ok=True)
    whitener.save(args.out / "whitener.pt")
    print(f"  whitener hash {whitener.hash[:16]}… -> {args.out / 'whitener.pt'}",
          flush=True)

    # ---- Stage 2: splits, written whitened bf16 --------------------------
    def writer_for(split: str) -> ShardWriter:
        return ShardWriter(
            args.out, split,
            whitener_hash=whitener.hash,
            sites=SITES, d_model=d_model, meta=corpus_meta,
        )

    stages = [
        ("calibration", args.calib_tokens),
        ("eval", args.eval_tokens),
        ("train", args.train_tokens),
    ]
    w_gpu = whitener.W.to(args.device)
    mu_gpu = whitener.mean.to(args.device)
    raw_writer = writer_for("raw_validation")
    raw_writer.whitener_hash = "raw:" + whitener.hash
    raw_remaining = RAW_VALIDATION_TOKENS
    heldout_m2 = torch.zeros(
        len(SITES), d_model, d_model, dtype=torch.float64, device=args.device
    )
    heldout_n = 0

    for split, quota in stages:
        writer = writer_for(split)
        written = 0
        t0 = time.time()
        io_wait = 0.0
        while written < quota:
            acts = next(batches)
            take = min(acts.shape[0], quota - written)
            raw = acts[:take]
            xw = torch.einsum("sde,nse->nsd", w_gpu, raw.float() - mu_gpu)
            if split == "calibration":
                if raw_remaining > 0:
                    n_raw = min(raw_remaining, take)
                    raw_writer.add(raw[:n_raw].cpu())
                    raw_remaining -= n_raw
                if heldout_n < 200_000:
                    heldout_m2 += torch.einsum(
                        "nsd,nse->sde", xw.double(), xw.double()
                    )
                    heldout_n += take
            t_io = time.time()
            writer.add(xw.to(torch.bfloat16).cpu())
            io_wait += time.time() - t_io
            written += take
            if written % 1_000_000 < take:
                dt = time.time() - t0
                print(
                    f"  {split} {written:,}/{quota:,} tokens "
                    f"({written / dt:,.0f} tok/s, io-wait {io_wait / dt:.0%})",
                    flush=True,
                )
        manifest = writer.close()
        print(f"  {split}: {manifest['n_tokens']:,} tokens in "
              f"{len(manifest['shards'])} shards", flush=True)
    raw_manifest = raw_writer.close()
    print(f"  raw_validation: {raw_manifest['n_tokens']:,} tokens", flush=True)

    # ---- held-out transformed-second-moment validation (D9, S4) ----------
    m2 = (heldout_m2 / max(heldout_n, 1)).cpu()
    dev_vs_pred, mean_dev_vs_one = [], []
    for s in range(len(SITES)):
        held = torch.linalg.eigvalsh(m2[s]).flip(0)
        reg = whitener.eigenvalues[s].double().flip(0)
        lam = float(whitener.ridge[s])
        predicted = (reg - lam) / reg
        dev_vs_pred.append(float((held - predicted).abs().mean()))
        mean_dev_vs_one.append(float(held.sub(1).abs().mean()))
    print(f"held-out whitened spectrum, mean |eig - predicted| per site: "
          f"{[round(v, 4) for v in dev_vs_pred]}", flush=True)
    print(f"held-out whitened spectrum, mean |eig - 1| per site "
          f"(ridge softness): {[round(v, 3) for v in mean_dev_vs_one]}",
          flush=True)

    report = {
        "whitener_hash": whitener.hash,
        "heldout_eig_dev_vs_predicted": dev_vs_pred,
        "heldout_eig_dev_vs_identity": mean_dev_vs_one,
        "heldout_tokens": heldout_n,
        "meta": corpus_meta,
    }
    (args.out / "harvest_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
