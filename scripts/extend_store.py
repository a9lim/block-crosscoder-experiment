"""E6 (runbook-phase099 tranches 1/6): extend the 4b pilot store's train
split with fresh, corpus-disjoint tokens under the FROZEN pilot whitener.

Mechanism: the pilot harvest consumed a deterministic stream — pinned
corpus revision, streaming order, tokenize, pack_token_rows — so the
extension replays the same stream and fast-forwards past an upper bound
on the rows the pilot consumed. Row r of the replay is bit-identical to
row r of the pilot's stream, so skipping rows [0, skip) guarantees
token-level disjointness with everything stored (whitener slice
included). The pilot's own row budget was

    n_rows = ceil(total_tokens / (CTX - DROP_POSITIONS)) + 5*batch_rows + 2

evaluated at total = 2M whitener + 2M calib + 1M eval + 6M train = 11M
with batch_rows=8 -> 10,805 — an upper bound on rows actually pulled
(measured consumption is full batches: ceil(11M/8176)*8 = 10,768). The
default --skip-rows adds a +64-row margin on top of the bound. Skipping
extra rows only moves deeper into a 10B-token corpus; skipping too few
would duplicate — so the margin is one-sided by construction.

The whitener is loaded, never refit (--load-whitener semantics); the
corpus revision comes from whitener.meta, never re-resolved. A held-out
transformed-second-moment check on the first 200k extension tokens
reports whitener drift vs the shrinkage prediction (the extension sits
~11M tokens deeper in the corpus than the fit slice).

Output: <store>/train_ext/ shards + a merged <store>/train12m/split.json
whose shard entries point into ../train and ../train_ext — readable by
StoreReader(store, "train12m") with no reader changes (the tranche-6
epochs-vs-fresh factorial's 12M x 2ep arm).

  nohup python -u scripts/extend_store.py \
      > /data/runs/bcc-phase099/extend_store.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL = "google/gemma-3-4b-pt"
CORPUS = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
SITES = (9, 12, 15, 18, 21, 24, 27, 30)
CTX = 1024
DROP_POSITIONS = 2

# The pilot harvest's stream-consumption upper bound (docstring math).
PILOT_TOTAL_TOKENS = 11_000_000
PILOT_BATCH_ROWS = 8
TOKENS_PER_ROW = CTX - DROP_POSITIONS
PILOT_N_ROWS = -(-PILOT_TOTAL_TOKENS // TOKENS_PER_ROW) + 5 * PILOT_BATCH_ROWS + 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext-tokens", type=int, default=6_000_000)
    parser.add_argument(
        "--skip-rows", type=int, default=PILOT_N_ROWS + 64,
        help="rows of the replayed stream to drain before harvesting "
        f"(default: pilot bound {PILOT_N_ROWS} + 64 margin)",
    )
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument(
        "--store", type=Path,
        default=Path("/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb"),
    )
    parser.add_argument("--split", default="train_ext")
    parser.add_argument("--merged-split", default="train12m")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from block_crosscoder_experiment.phase0.harvest import pack_token_rows
    from block_crosscoder_experiment.store import MANIFEST_NAME, ShardWriter, Whitener

    whitener = Whitener.load(args.store / "whitener.pt")
    meta = whitener.meta
    if meta.get("model") != MODEL or tuple(whitener.sites) != SITES:
        raise SystemExit(
            f"whitener meta mismatch: model={meta.get('model')} "
            f"sites={whitener.sites} — this script extends the D13 pilot store"
        )
    corpus_sha = meta["corpus_revision"]
    print(
        f"config: frozen whitener {whitener.hash[:16]}… corpus_rev={corpus_sha[:12]} "
        f"skip_rows={args.skip_rows} ext={args.ext_tokens:,} split={args.split}",
        flush=True,
    )

    train_manifest = json.loads(
        (args.store / "train" / MANIFEST_NAME).read_text()
    )
    if train_manifest["whitener_hash"] != whitener.hash:
        raise SystemExit("train split was written under a different whitener")

    from datasets import load_dataset
    from sae_lens import HookedSAETransformer

    model = HookedSAETransformer.from_pretrained_no_processing(
        MODEL, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    d_model = int(model.cfg.d_model)
    hook_names = [f"blocks.{L}.hook_resid_post" for L in SITES]
    stop_at = max(SITES) + 1
    bos = model.tokenizer.bos_token_id

    stream = load_dataset(
        CORPUS[0], name=CORPUS[1], split="train", streaming=True,
        revision=corpus_sha,
    )

    def token_docs():
        for doc in stream:
            yield model.tokenizer.encode(doc["text"], add_special_tokens=False)

    ext_rows = -(-args.ext_tokens // TOKENS_PER_ROW) + 5 * args.batch_rows + 2
    rows = pack_token_rows(
        token_docs(), ctx=CTX, bos_id=bos,
        n_rows=args.skip_rows + ext_rows,
    )

    # ---- fast-forward: drain the pilot's rows, model-free ----------------
    t0 = time.time()
    for i, _ in enumerate(rows):
        if i + 1 >= args.skip_rows:
            break
        if (i + 1) % 2000 == 0:
            print(f"  skip {i + 1:,}/{args.skip_rows:,} rows "
                  f"({(i + 1) / (time.time() - t0):,.0f} rows/s)", flush=True)
    print(f"fast-forwarded {args.skip_rows:,} rows in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)

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
            acts = torch.stack([cache[h] for h in hook_names], dim=2)
            yield acts[:, DROP_POSITIONS:].reshape(-1, len(SITES), d_model)

    ext_meta = dict(meta) | {
        "extension": "E6 corpus-disjoint train extension",
        "skip_rows": args.skip_rows,
        "parent_train_tokens": train_manifest["n_tokens"],
    }
    writer = ShardWriter(
        args.store, args.split,
        whitener_hash=whitener.hash, sites=SITES, d_model=d_model, meta=ext_meta,
    )
    w_gpu = whitener.W.to(args.device)
    mu_gpu = whitener.mean.to(args.device)
    heldout_m2 = torch.zeros(
        len(SITES), d_model, d_model, dtype=torch.float64, device=args.device
    )
    heldout_n = 0
    written = 0
    t0 = time.time()
    batches = act_batches()
    while written < args.ext_tokens:
        acts = next(batches)
        take = min(acts.shape[0], args.ext_tokens - written)
        xw = torch.einsum("sde,nse->nsd", w_gpu, acts[:take].float() - mu_gpu)
        if heldout_n < 200_000:
            heldout_m2 += torch.einsum("nsd,nse->sde", xw.double(), xw.double())
            heldout_n += take
        writer.add(xw.to(torch.bfloat16).cpu())
        written += take
        if written % 1_000_000 < take:
            print(f"  {args.split} {written:,}/{args.ext_tokens:,} tokens "
                  f"({written / (time.time() - t0):,.0f} tok/s)", flush=True)
    ext_manifest = writer.close()
    print(f"  {args.split}: {ext_manifest['n_tokens']:,} tokens in "
          f"{len(ext_manifest['shards'])} shards", flush=True)

    # ---- whitener drift on the extension (D9-style, frozen W) ------------
    m2 = (heldout_m2 / max(heldout_n, 1)).cpu()
    dev = []
    for s in range(len(SITES)):
        held = torch.linalg.eigvalsh(m2[s]).flip(0)
        reg = whitener.eigenvalues[s].double().flip(0)
        lam = float(whitener.ridge[s])
        dev.append(float((held - (reg - lam) / reg).abs().mean()))
    print(f"extension held-out spectrum, mean |eig - predicted| per site: "
          f"{[round(v, 4) for v in dev]}", flush=True)

    # ---- merged manifest: train + train_ext as one logical split ---------
    merged_dir = args.store / args.merged_split
    merged_dir.mkdir(exist_ok=True)
    merged = {
        "split": args.merged_split,
        "whitener_hash": whitener.hash,
        "sites": list(SITES),
        "d_model": d_model,
        "n_tokens": train_manifest["n_tokens"] + ext_manifest["n_tokens"],
        "shards": (
            [{"file": f"../train/{s['file']}", "n_tokens": s["n_tokens"]}
             for s in train_manifest["shards"]]
            + [{"file": f"../{args.split}/{s['file']}", "n_tokens": s["n_tokens"]}
               for s in ext_manifest["shards"]]
        ),
        "meta": ext_meta | {"merged_from": ["train", args.split]},
    }
    (merged_dir / MANIFEST_NAME).write_text(json.dumps(merged, indent=2) + "\n")
    print(f"merged split {args.merged_split}: {merged['n_tokens']:,} tokens "
          f"({len(merged['shards'])} shards) -> {merged_dir / MANIFEST_NAME}",
          flush=True)

    report = {
        "whitener_hash": whitener.hash,
        "skip_rows": args.skip_rows,
        "ext_tokens": ext_manifest["n_tokens"],
        "heldout_eig_dev_vs_predicted": dev,
        "heldout_tokens": heldout_n,
    }
    (args.store / f"{args.split}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"-> {args.store / args.split}", flush=True)


if __name__ == "__main__":
    main()
