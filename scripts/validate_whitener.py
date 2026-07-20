"""Tranche 7 (runbook-phase099): production-harvest whitener derisk.

One streaming GPU pass over fresh, corpus-disjoint tokens (the replay
stream fast-forwarded past everything the pilot harvest AND the E6
extension consumed) — nothing stored. Three measurements ride the pass:

1. **Whitener stability vs slice size up to the planned 5M production
   slice.** Sufficient statistics accumulate in 500k-token sub-slices
   (fp64, batch-granular per D9, offloaded to host at each boundary);
   post-pass they merge into prefix fits {0.5M, 1M, 2M, 2.5M, 5M} and
   five independent 1M fits. Metric per site: rel ΔW vs the full 5M fit
   (the harvest's own halves/quarters ladder — the pilot's depth-graded
   drift L27 0.026 / L30 0.031 at 2M is this number), plus the D9
   held-out transformed-spectrum deviation on 200k held-out tokens
   (computed from raw held-out sufficient statistics, so every prefix
   whitener is evaluated on the same tokens without storing them).
2. **Late-layer tail statistics on fresh tokens.** Per site: abs-max,
   threshold exceedance counts (raw: fp16-overflow documentation;
   whitened under the frozen pilot whitener: store-path headroom),
   per-channel running max (rogue channels named), and the streamed
   bf16 cast error of whitened values (mean/max relative — the store
   dtype's actual cost).
3. **Renorm scalars on independent slices** (F7 gauge stability): the
   site_rms_scalars of the five independent 1M fits, spread per site.

  nohup python -u scripts/validate_whitener.py \
      > /data/runs/bcc-phase099/tranche7_whitener.log 2>&1 &
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
TOKENS_PER_ROW = CTX - DROP_POSITIONS

# Stream-consumption upper bounds (extend_store.py docstring math): the
# pilot pulled <= 10,805 rows and E6 skipped 10,805+64 then pulled
# <= ceil(6M/1022) + 5*8 + 2 rows. Skipping past both bounds + margin
# guarantees token-level disjointness from the store AND the extension.
PILOT_SKIP = 10_805 + 64
EXT_ROWS_BOUND = -(-6_000_000 // TOKENS_PER_ROW) + 5 * 8 + 2
DEFAULT_SKIP = PILOT_SKIP + EXT_ROWS_BOUND + 64

RAW_THRESHOLDS = (1e3, 1e4, 65504.0, 1e5)  # 65504 = fp16 max
WHT_THRESHOLDS = (8.0, 16.0, 32.0, 64.0, 128.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-tokens", type=int, default=5_000_000)
    parser.add_argument("--sub-tokens", type=int, default=500_000)
    parser.add_argument("--heldout-tokens", type=int, default=200_000)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument(
        "--skip-rows", type=int, default=DEFAULT_SKIP,
        help=f"replay rows to drain first (default {DEFAULT_SKIP}: "
        "pilot bound + E6 bound + margin)",
    )
    parser.add_argument(
        "--store", type=Path,
        default=Path("/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb"),
        help="pilot store (frozen whitener for whitened-tail stats)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/data/runs/bcc-phase099/tranche7_whitener.json"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.slice_tokens % args.sub_tokens:
        raise SystemExit("--slice-tokens must be a multiple of --sub-tokens")
    n_sub = args.slice_tokens // args.sub_tokens

    import torch

    from block_crosscoder_experiment.phase0.harvest import pack_token_rows
    from block_crosscoder_experiment.store import Whitener, WhitenerAccumulator

    pilot = Whitener.load(args.store / "whitener.pt")
    if tuple(pilot.sites) != SITES or pilot.meta.get("model") != MODEL:
        raise SystemExit("pilot whitener meta mismatch — wrong store?")
    corpus_sha = pilot.meta["corpus_revision"]
    print(
        f"config: slice={args.slice_tokens:,} x{n_sub} subs "
        f"heldout={args.heldout_tokens:,} skip_rows={args.skip_rows} "
        f"pilot={pilot.hash[:16]}… corpus_rev={corpus_sha[:12]}",
        flush=True,
    )

    from datasets import load_dataset
    from sae_lens import HookedSAETransformer

    model = HookedSAETransformer.from_pretrained_no_processing(
        MODEL, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    d = int(model.cfg.d_model)
    S = len(SITES)
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

    total_tokens = args.slice_tokens + args.heldout_tokens
    need_rows = -(-total_tokens // TOKENS_PER_ROW) + 5 * args.batch_rows + 2
    rows = pack_token_rows(
        token_docs(), ctx=CTX, bos_id=bos, n_rows=args.skip_rows + need_rows
    )

    t0 = time.time()
    for i, _ in enumerate(rows):
        if i + 1 >= args.skip_rows:
            break
        if (i + 1) % 4000 == 0:
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
            yield acts[:, DROP_POSITIONS:].reshape(-1, S, d)

    dev = args.device
    w_gpu = pilot.W.to(dev)
    mu_gpu = pilot.mean.to(dev)

    # Tail accumulators (whole pass, fp32/fp64 on GPU).
    raw_chan_max = torch.zeros(S, d, device=dev)
    wht_chan_max = torch.zeros(S, d, device=dev)
    raw_exceed = torch.zeros(S, len(RAW_THRESHOLDS), dtype=torch.float64, device=dev)
    wht_exceed = torch.zeros(S, len(WHT_THRESHOLDS), dtype=torch.float64, device=dev)
    raw_sumsq = torch.zeros(S, dtype=torch.float64, device=dev)
    bf16_relerr_sum = torch.zeros(S, dtype=torch.float64, device=dev)
    bf16_relerr_max = torch.zeros(S, device=dev)
    tail_n = 0

    def update_tails(acts_f32: torch.Tensor) -> None:
        nonlocal tail_n
        a = acts_f32.abs()
        raw_chan_max.copy_(torch.maximum(raw_chan_max, a.amax(dim=0)))
        for j, t in enumerate(RAW_THRESHOLDS):
            raw_exceed[:, j] += (a > t).sum(dim=(0, 2)).double()
        raw_sumsq.add_(acts_f32.double().pow(2).sum(dim=(0, 2)))
        xw = torch.einsum("sde,nse->nsd", w_gpu, acts_f32 - mu_gpu)
        wa = xw.abs()
        wht_chan_max.copy_(torch.maximum(wht_chan_max, wa.amax(dim=0)))
        for j, t in enumerate(WHT_THRESHOLDS):
            wht_exceed[:, j] += (wa > t).sum(dim=(0, 2)).double()
        rel = (xw.to(torch.bfloat16).float() - xw).abs() / wa.clamp_min(1e-6)
        bf16_relerr_sum.add_(rel.double().sum(dim=(0, 2)))
        bf16_relerr_max.copy_(torch.maximum(bf16_relerr_max, rel.amax(dim=(0, 2))))
        tail_n += acts_f32.shape[0]

    # Streaming: one active fp64 sub-slice accumulator, offloaded at each
    # 500k boundary (host keeps [n_sub] x (n, sum, outer)); held-out raw
    # sufficient statistics reuse the same accumulator class.
    subs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    active = WhitenerAccumulator(S, d, device=dev)
    seen = 0
    t0 = time.time()
    batches = act_batches()
    while seen < args.slice_tokens:
        acts = next(batches).float()
        update_tails(acts)
        off = 0
        while off < acts.shape[0] and seen < args.slice_tokens:
            boundary = (seen // args.sub_tokens + 1) * args.sub_tokens
            take = min(acts.shape[0] - off, boundary - seen,
                       args.slice_tokens - seen)
            active.update(acts[off : off + take])
            off += take
            seen += take
            if seen % args.sub_tokens == 0:
                subs.append((active.n, active.sum.cpu(), active.outer.cpu()))
                active = WhitenerAccumulator(S, d, device=dev)
                print(f"  sub-slice {len(subs)}/{n_sub} done "
                      f"({seen:,} tokens, {seen / (time.time() - t0):,.0f} tok/s)",
                      flush=True)

    held = WhitenerAccumulator(S, d, device=dev)
    hseen = 0
    while hseen < args.heldout_tokens:
        acts = next(batches).float()
        take = min(acts.shape[0], args.heldout_tokens - hseen)
        held.update(acts[:take])
        update_tails(acts[:take])
        hseen += take
    held_stats = (held.n, held.sum.cpu(), held.outer.cpu())
    stream_tok_s = (seen + hseen) / (time.time() - t0)
    print(f"streamed {seen + hseen:,} tokens at {stream_tok_s:,.0f} tok/s",
          flush=True)

    # ---- post-pass fits on host ------------------------------------------
    def acc_from(stats_list) -> WhitenerAccumulator:
        acc = WhitenerAccumulator(S, d, device="cpu")
        for n, s_, o_ in stats_list:
            acc.n += n
            acc.sum += s_
            acc.outer += o_
        return acc

    meta = {"tranche7": "whitener stability", "corpus_revision": corpus_sha}

    def fit(stats_list, label: str) -> Whitener:
        t = time.time()
        w = acc_from(stats_list).finalize(sites=SITES, meta=meta)
        print(f"  fit {label}: {w.n_fit_tokens:,} tokens "
              f"({time.time() - t:.0f}s)", flush=True)
        return w

    prefix_sizes = [1, 2, 4, 5, n_sub]  # 0.5M 1M 2M 2.5M 5M at 500k subs
    prefixes = {k: fit(subs[:k], f"prefix {k * args.sub_tokens:,}")
                for k in prefix_sizes}
    full = prefixes[n_sub]
    indep = [fit(subs[i : i + 2], f"indep 1M #{i // 2}")
             for i in range(0, n_sub, 2)]

    def rel_dw(w: Whitener) -> list[float]:
        r = ((w.W - full.W).flatten(1).norm(dim=1)
             / full.W.flatten(1).norm(dim=1))
        return [round(float(v), 4) for v in r]

    def rel_dmu(w: Whitener) -> list[float]:
        r = (w.mean - full.mean).norm(dim=1) / full.mean.norm(dim=1)
        return [round(float(v), 4) for v in r]

    def heldout_dev(w: Whitener) -> list[float]:
        n_h, s_h, o_h = held_stats
        mean_h = (s_h / n_h).double()
        mu = w.mean.double()
        out = []
        for s in range(S):
            c = (o_h[s] / n_h
                 - torch.outer(mean_h[s], mu[s]) - torch.outer(mu[s], mean_h[s])
                 + torch.outer(mu[s], mu[s]))
            m2 = w.W[s].double() @ c @ w.W[s].double().T
            held_e = torch.linalg.eigvalsh(m2).flip(0)
            reg = w.eigenvalues[s].double().flip(0)
            lam = float(w.ridge[s])
            out.append(round(float((held_e - (reg - lam) / reg).abs().mean()), 4))
        return out

    report: dict = {
        "sites": list(SITES),
        "slice_tokens": args.slice_tokens,
        "heldout_tokens": hseen,
        "skip_rows": args.skip_rows,
        "pilot_whitener_hash": pilot.hash,
        "stream_tok_s": round(stream_tok_s),
        "stability_rel_dW_vs_5m": {},
        "stability_rel_dmu_vs_5m": {},
        "heldout_eig_dev": {},
        "renorm_scalars": {},
        "tails": {},
    }
    for k in prefix_sizes[:-1]:
        toks = k * args.sub_tokens
        report["stability_rel_dW_vs_5m"][f"{toks}"] = rel_dw(prefixes[k])
        report["stability_rel_dmu_vs_5m"][f"{toks}"] = rel_dmu(prefixes[k])
        print(f"stability prefix {toks:,}: rel ΔW per site "
              f"{report['stability_rel_dW_vs_5m'][str(toks)]}", flush=True)
    report["stability_rel_dW_vs_5m"]["indep_1m"] = [rel_dw(w) for w in indep]
    report["stability_rel_dW_vs_5m"]["pilot_2m"] = rel_dw(pilot)
    print(f"stability pilot(2M, store slice): rel ΔW per site "
          f"{report['stability_rel_dW_vs_5m']['pilot_2m']}", flush=True)

    for k in prefix_sizes:
        toks = k * args.sub_tokens
        report["heldout_eig_dev"][f"{toks}"] = heldout_dev(prefixes[k])
        print(f"heldout dev prefix {toks:,}: {report['heldout_eig_dev'][str(toks)]}",
              flush=True)
    report["heldout_eig_dev"]["pilot_2m"] = heldout_dev(pilot)
    print(f"heldout dev pilot(2M): {report['heldout_eig_dev']['pilot_2m']}",
          flush=True)

    scal_full = full.site_rms_scalars()
    scal_indep = torch.stack([w.site_rms_scalars() for w in indep])
    scal_pilot = pilot.site_rms_scalars()
    report["renorm_scalars"] = {
        "full_5m": [round(float(v), 5) for v in scal_full],
        "pilot_2m": [round(float(v), 5) for v in scal_pilot],
        "indep_1m_mean": [round(float(v), 5) for v in scal_indep.mean(0)],
        "indep_1m_cv": [
            round(float(v), 5)
            for v in scal_indep.std(0, unbiased=True) / scal_indep.mean(0)
        ],
        "indep_1m_max_rel_dev_vs_5m": [
            round(float(v), 5)
            for v in ((scal_indep - scal_full).abs() / scal_full).amax(0)
        ],
    }
    print(f"renorm scalars: 5M {report['renorm_scalars']['full_5m']}", flush=True)
    print(f"renorm scalars: indep-1M CV {report['renorm_scalars']['indep_1m_cv']}",
          flush=True)

    def top_channels(chan_max: torch.Tensor, k: int = 3) -> list[list[list[float]]]:
        vals, idx = chan_max.cpu().topk(k, dim=1)
        return [[[int(i), round(float(v), 1)] for i, v in zip(ii, vv)]
                for ii, vv in zip(idx, vals)]

    n_vals = tail_n * d
    report["tails"] = {
        "tokens": tail_n,
        "raw_abs_max": [round(float(v), 1) for v in raw_chan_max.amax(1)],
        "raw_rms": [round(float(v), 3)
                    for v in (raw_sumsq.cpu() / n_vals).sqrt()],
        "raw_exceed_per_1e6_vals": {
            str(t): [round(float(v), 3) for v in raw_exceed[:, j].cpu() / n_vals * 1e6]
            for j, t in enumerate(RAW_THRESHOLDS)
        },
        "raw_top_channels": top_channels(raw_chan_max),
        "whitened_abs_max": [round(float(v), 2) for v in wht_chan_max.amax(1)],
        "whitened_exceed_per_1e6_vals": {
            str(t): [round(float(v), 3) for v in wht_exceed[:, j].cpu() / n_vals * 1e6]
            for j, t in enumerate(WHT_THRESHOLDS)
        },
        "whitened_top_channels": top_channels(wht_chan_max),
        "bf16_relerr_mean": [round(float(v), 6)
                             for v in bf16_relerr_sum.cpu() / n_vals],
        "bf16_relerr_max": [round(float(v), 4) for v in bf16_relerr_max.cpu()],
    }
    print(f"tails: raw abs max per site {report['tails']['raw_abs_max']}", flush=True)
    print(f"tails: whitened abs max per site {report['tails']['whitened_abs_max']}",
          flush=True)
    print(f"tails: bf16 rel err mean {report['tails']['bf16_relerr_mean']}",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {args.out}", flush=True)
    print("tranche7 whitener validation done", flush=True)


if __name__ == "__main__":
    main()
