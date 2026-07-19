"""Harvest GPT-2-small layer-7 SAE codes over OpenWebText (Phase-0 control).

Runs on jobe (CUDA). Reproduces the Bloom/Engels harvest conditions the
provenance gate verified: gpt2-small-res-jb @ blocks.7.hook_resid_pre,
ctx 128, BOS prepended, Skylion007/openwebtext, model loaded with the SAE's
own model_from_pretrained_kwargs. Codes land in a CodeStore under
/data/stores/bcc-phase0/ with full identity metadata.

  python scripts/harvest_phase0_control.py --n-tokens 4000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"
CORPUS = "Skylion007/openwebtext"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tokens", type=int, default=4_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out", type=Path, default=Path("/data/stores/bcc-phase0/gpt2_l7_owt")
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from datasets import load_dataset
    from sae_lens import SAE, HookedSAETransformer

    from block_crosscoder_experiment.phase0.harvest import (
        decoder_hash,
        harvest_codes,
        pack_token_rows,
    )

    sae = SAE.from_pretrained(RELEASE, SAE_ID, device=args.device)
    kwargs = sae.cfg.metadata.model_from_pretrained_kwargs or {}
    model = HookedSAETransformer.from_pretrained_no_processing(
        "gpt2", **kwargs
    ).to(args.device)
    model.eval()

    ctx = int(sae.cfg.metadata.context_size)
    bos = model.tokenizer.bos_token_id
    print(f"harvesting {args.n_tokens:,} tokens @ ctx {ctx} from {CORPUS}")

    stream = load_dataset(CORPUS, split="train", streaming=True)

    def token_docs():
        for doc in stream:
            yield model.tokenizer.encode(doc["text"])

    tokens_per_row = ctx - 1  # BOS dropped at harvest
    n_rows = args.n_tokens // tokens_per_row
    rows = pack_token_rows(token_docs(), ctx=ctx, bos_id=bos, n_rows=n_rows)

    start = time.time()
    store = harvest_codes(
        model,
        sae,
        rows,
        hook_name=SAE_ID,
        out_root=args.out,
        batch_size=args.batch_size,
        meta={
            "release": RELEASE,
            "sae_id": SAE_ID,
            "corpus": CORPUS,
            "context_size": ctx,
            "prepend_bos": True,
            "bos_dropped": True,
            "W_dec_sha256": decoder_hash(sae),
            "model_from_pretrained_kwargs": kwargs,
        },
        device=args.device,
    )
    dt = time.time() - start
    counts = store.firing_counts()
    active = counts.gt(0)
    print(
        f"done in {dt/60:.1f} min: {store.n_tokens:,} tokens, "
        f"nnz {store.meta['nnz']:,} (mean L0 {store.meta['nnz']/store.n_tokens:.1f}), "
        f"zero-code fraction {store.meta['zero_code_fraction']:.2e}"
    )
    print(
        f"features ever firing: {int(active.sum()):,}/{counts.shape[0]:,}  "
        f"-> {store.root}"
    )


if __name__ == "__main__":
    main()
