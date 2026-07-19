"""Harvest gemma-3-4b layer-22 SAE codes over the Pile (Phase-0 target).

Runs on jobe (CUDA). Conditions from the SAE's own metadata (verified at
phase start): gemma-scope-2-4b-pt-res @ blocks.22.hook_resid_post
(layer_22_width_16k_l0_medium — the provenance-pinned config), ctx 1024,
BOS prepended, monology/pile-uncopyrighted. Model runs bf16 (fp16 is
banned in the harvest path; f32 doesn't fit beside the codes on 24 GB),
codes stored f32.

Defaults reproduce the 16k run exactly; the width_65k reroute passes
--sae-id and --out (the Pile stream is deterministic, so both harvests
see identical token rows).

  python scripts/harvest_phase0_gemma.py --n-tokens 4000000
  python scripts/harvest_phase0_gemma.py --sae-id layer_22_width_65k_l0_medium \
      --out /data/stores/bcc-phase0/gemma3_4b_l22_65k_pile
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

RELEASE = "gemma-scope-2-4b-pt-res"
SAE_ID = "layer_22_width_16k_l0_medium"
CORPUS = "monology/pile-uncopyrighted"
MODEL = "google/gemma-3-4b-pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tokens", type=int, default=4_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("/data/stores/bcc-phase0/gemma3_4b_l22_pile")
    )
    parser.add_argument("--sae-id", default=SAE_ID)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"config: release={RELEASE} sae_id={args.sae_id} out={args.out}", flush=True)

    import torch
    from datasets import load_dataset
    from sae_lens import SAE, HookedSAETransformer

    from block_crosscoder_experiment.phase0.harvest import (
        decoder_hash,
        harvest_codes,
        pack_token_rows,
    )

    sae = SAE.from_pretrained(RELEASE, args.sae_id, device=args.device)
    hook_name = sae.cfg.metadata.hook_name
    ctx = int(sae.cfg.metadata.context_size)
    model = HookedSAETransformer.from_pretrained_no_processing(
        MODEL, dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    bos = model.tokenizer.bos_token_id
    print(f"harvesting {args.n_tokens:,} tokens @ ctx {ctx} from {CORPUS}")
    print(f"hook {hook_name}, model bf16, SAE {sae.cfg.dtype}")

    stream = load_dataset(CORPUS, split="train", streaming=True)

    def token_docs():
        for doc in stream:
            # gemma's tokenizer injects BOS on encode; rows add their own.
            yield model.tokenizer.encode(doc["text"], add_special_tokens=False)

    n_rows = args.n_tokens // (ctx - 1)
    rows = pack_token_rows(token_docs(), ctx=ctx, bos_id=bos, n_rows=n_rows)

    start = time.time()
    store = harvest_codes(
        model,
        sae,
        rows,
        hook_name=hook_name,
        out_root=args.out,
        batch_size=args.batch_size,
        meta={
            "release": RELEASE,
            "sae_id": args.sae_id,
            "model": MODEL,
            "corpus": CORPUS,
            "context_size": ctx,
            "prepend_bos": True,
            "bos_dropped": True,
            "model_dtype": "bfloat16",
            "W_dec_sha256": decoder_hash(sae),
        },
        device=args.device,
    )
    dt = time.time() - start
    counts = store.firing_counts()
    print(
        f"done in {dt/60:.1f} min: {store.n_tokens:,} tokens, "
        f"nnz {store.meta['nnz']:,} (mean L0 {store.meta['nnz']/store.n_tokens:.1f}), "
        f"zero-code fraction {store.meta['zero_code_fraction']:.2e}"
    )
    print(
        f"features ever firing: {int(counts.gt(0).sum()):,}/{counts.shape[0]:,}  "
        f"-> {store.root}"
    )


if __name__ == "__main__":
    main()
