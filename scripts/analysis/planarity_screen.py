"""Unsupervised manifold screen: find blocks whose codes live on planes,
then decode what they fire on.

The discovery-direction complement to the supervised calendar probe
(0.9.6 analysis pass): block 23 was *confirmed* by a named family; this
screen is how such blocks get *found* without naming one.

Stage `rank` (CPU, needs evalstats_<run>.npz from eval_activation_stats):
  Per block, eigendecompose the eval code second moment zz [b,b] and
  rank candidates by three complementary signatures, taking the union:
    - planar:    top-2 eigenvalue mass with a non-degenerate second axis
                 (lam2/lam1 high — a ring/plane, not a magnitude line)
    - midband:   code PR in [1.7, 3.0] — block 23's regime
    - all blocks pass a frequency sanity band [1e-4, 0.05] (excludes
      dead-ish tails and always-on syntax blocks)

Stage `decode` (jobe, CUDA): stream the pinned corpus (same packing as
the harvest, --skip-docs past the store slice), encode the checkpoint,
and keep the top-K activating (token, context) pairs per candidate
block. Output JSON is meant to be *read* — each candidate block's
contexts either name themselves or they don't.

  python scripts/analysis/planarity_screen.py --stage rank \
      --evalstats /data/runs/bcc-analysis/evalstats_winner.npz
  python scripts/analysis/planarity_screen.py --stage decode \
      --checkpoint /data/runs/bcc-phase095/bsc_lam0.001_seed0_lr0.0012 \
      --model google/gemma-3-1b-pt \
      --store /data/stores/bcc-phase09/gemma3_1b_6site_fineweb
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np

FREQ_BAND = (1e-4, 0.05)
PR_BAND = (1.7, 3.0)
TOP_N_PER_SIGNATURE = 64
TOP_K_CONTEXTS = 24
CTX_WINDOW = 10  # tokens of left context to decode


def rank(evalstats: Path, out: Path) -> list[int]:
    z = np.load(evalstats)
    zz = z["zz"].astype(np.float64)
    freq = z["fire_count"] / int(z["n_tokens"])
    ev = np.linalg.eigvalsh(zz)[:, ::-1]  # descending
    tot = np.maximum(ev.sum(1), 1e-30)
    top2 = (ev[:, 0] + ev[:, 1]) / tot
    ratio21 = ev[:, 1] / np.maximum(ev[:, 0], 1e-30)
    pr = (ev.sum(1) ** 2) / np.maximum((ev**2).sum(1), 1e-30)
    sane = (freq >= FREQ_BAND[0]) & (freq <= FREQ_BAND[1])

    planar_score = np.where(sane, top2 * ratio21, -1.0)
    midband = sane & (pr >= PR_BAND[0]) & (pr <= PR_BAND[1])
    midband_score = np.where(midband, ratio21, -1.0)

    cands: dict[int, dict] = {}
    for name, score in (("planar", planar_score), ("midband", midband_score)):
        for b in np.argsort(score)[::-1][:TOP_N_PER_SIGNATURE]:
            if score[b] <= 0:
                continue
            e = cands.setdefault(int(b), {
                "block": int(b), "freq": round(float(freq[b]), 5),
                "eig_frac": [round(float(v), 4) for v in ev[b] / tot[b]],
                "code_pr": round(float(pr[b]), 3), "signatures": [],
            })
            e["signatures"].append(name)
    ranked = sorted(
        cands.values(), key=lambda e: -(e["eig_frac"][0] + e["eig_frac"][1])
    )
    out.write_text(json.dumps(ranked, indent=1) + "\n")
    print(f"{len(ranked)} candidates -> {out}")
    return [e["block"] for e in ranked]


def decode(args) -> None:
    import torch

    from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
    from block_crosscoder_experiment.phase0.harvest import pack_token_rows
    from block_crosscoder_experiment.store import Whitener

    cand = [e["block"] for e in json.loads(args.candidates.read_text())]
    if not cand:
        raise SystemExit("no candidates — run --stage rank first")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sae_lens import HookedSAETransformer

    whitener = Whitener.load(args.store / "whitener.pt")
    sites = whitener.sites
    renorm = whitener.site_rms_scalars() if args.site_renorm else None

    model = HookedSAETransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    d_model = int(model.cfg.d_model)
    hook_names = [f"blocks.{L}.hook_resid_post" for L in sites]
    stop_at = max(sites) + 1
    bos = model.tokenizer.bos_token_id

    ckpt = torch.load(args.checkpoint / "latest.pt", map_location="cpu",
                      weights_only=False)
    mc = ckpt["model_cfg"]
    cfg = BSCConfig(
        n_blocks=mc["n_blocks"], block_dim=mc["block_dim"],
        n_sites=mc["n_sites"], d_model=mc["d_model"], k=mc["k"],
    )
    bsc = BlockCrosscoder(cfg, device=args.device)
    bsc.load_state_dict(ckpt["model"])
    bsc.eval()
    cand_t = torch.tensor(cand, device=args.device)

    corpus_sha = HfApi().dataset_info("HuggingFaceFW/fineweb-edu").sha
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        streaming=True, revision=corpus_sha,
    )

    def token_docs():
        for i, doc in enumerate(stream):
            if i < args.skip_docs:
                continue
            yield model.tokenizer.encode(doc["text"], add_special_tokens=False)

    ctx, drop = 1024, 2
    n_rows = -(-args.scan_tokens // (ctx - drop)) + args.batch_rows + 2
    rows = pack_token_rows(token_docs(), ctx=ctx, bos_id=bos, n_rows=n_rows)
    w_gpu = whitener.W.to(args.device)
    mu_gpu = whitener.mean.to(args.device)

    heaps: dict[int, list] = {b: [] for b in cand}
    scanned = 0
    counter = 0
    buf: list[torch.Tensor] = []

    @torch.no_grad()
    def process(toks: torch.Tensor) -> None:
        nonlocal scanned, counter
        _, cache = model.run_with_cache(
            toks.to(args.device),
            names_filter=lambda name: name in hook_names,
            stop_at_layer=stop_at, return_type=None,
        )
        acts = torch.stack([cache[h] for h in hook_names], dim=2)[:, drop:]
        B, T = acts.shape[0], acts.shape[1]
        xw = torch.einsum(
            "sde,nse->nsd", w_gpu,
            acts.reshape(-1, len(sites), d_model).float() - mu_gpu,
        )
        if renorm is not None:
            xw = xw * renorm.to(args.device).view(1, -1, 1)
        p = bsc.scores(bsc.encode(xw))[:, cand_t]  # [B*T, |cand|]
        top = p.topk(min(8, p.shape[0]), dim=0)
        vals, idxs = top.values.cpu(), top.indices.cpu()
        for j, b in enumerate(cand):
            for v, flat in zip(vals[:, j].tolist(), idxs[:, j].tolist()):
                if v <= float(bsc.theta):
                    continue
                r, pos = divmod(flat, T)
                lo = max(0, pos + drop - CTX_WINDOW)
                window = toks[r, lo : pos + drop + 1].tolist()
                counter += 1
                item = (v, counter, model.tokenizer.decode(window[-1:]),
                        model.tokenizer.decode(window))
                if len(heaps[b]) < TOP_K_CONTEXTS:
                    heapq.heappush(heaps[b], item)
                else:
                    heapq.heappushpop(heaps[b], item)
        scanned += B * T

    for row in rows:
        buf.append(row)
        if len(buf) < args.batch_rows:
            continue
        process(torch.stack(buf))
        buf = []
        if scanned >= args.scan_tokens:
            break
        if scanned % 1_000_000 < args.batch_rows * (ctx - drop):
            print(f"  scanned {scanned:,} tokens", flush=True)

    report = {
        "checkpoint": str(args.checkpoint), "model": args.model,
        "scanned_tokens": scanned, "theta": float(bsc.theta),
        "blocks": {
            str(b): [
                {"score": round(v, 3), "token": tok, "context": ctx_}
                for v, _, tok, ctx_ in sorted(heaps[b], reverse=True)
            ]
            for b in cand
        },
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("rank", "decode"), required=True)
    ap.add_argument("--evalstats", type=Path,
                    help="rank: evalstats_<run>.npz path")
    ap.add_argument("--candidates", type=Path,
                    default=Path("/data/runs/bcc-analysis/planarity_candidates.json"))
    ap.add_argument("--checkpoint", type=Path, help="decode: run dir with latest.pt")
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--store", type=Path,
                    default=Path("/data/stores/bcc-phase09/gemma3_1b_6site_fineweb"))
    ap.add_argument("--site-renorm", action="store_true",
                    help="decode: apply F7 renorm scalars (renorm-arm checkpoints)")
    ap.add_argument("--out", type=Path,
                    default=Path("/data/runs/bcc-analysis/planarity_contexts.json"))
    ap.add_argument("--scan-tokens", type=int, default=2_000_000)
    ap.add_argument("--skip-docs", type=int, default=20_000)
    ap.add_argument("--batch-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.stage == "rank":
        if args.evalstats is None:
            raise SystemExit("--stage rank needs --evalstats")
        rank(args.evalstats, args.candidates)
    else:
        if args.checkpoint is None:
            raise SystemExit("--stage decode needs --checkpoint")
        decode(args)


if __name__ == "__main__":
    main()
