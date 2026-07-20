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
import re
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.discovery.harvest import pack_token_rows
from block_crosscoder_experiment.analysis.catalog import (
    CAP_ONLY,
    FAMILIES,
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
    ap.add_argument(
        "--background-acts-cap", type=int, default=10_000,
        help="retain this many transformed background examples as bf16 bits "
        "for operational firing/selectivity estimates; 0 stores moments only",
    )
    ap.add_argument(
        "--target-docs-per-class", type=int, default=0,
        help="natural-context rare-family capture: prefilter raw documents and "
        "accept them until every requested class appears in this many docs; "
        "0 keeps the ordinary representative stream. This changes prevalence "
        "and may not supply a background-specificity estimate.",
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

    prefilter_counts = {
        (family, class_id): 0
        for family in args.families
        for class_id in range(len(FAMILIES[family]))
    }
    prefilter_fold_counts = {key: [0, 0, 0] for key in prefilter_counts}
    prefilter_patterns = {
        (family, class_id): re.compile(
            rf"(?<!\w)(?:{'|'.join(re.escape(form) for form in (
                (word,) if family in CAP_ONLY else (word, word.lower())
            ))})(?!\w)"
        )
        for family in args.families
        for class_id, word in enumerate(FAMILIES[family])
    }

    def token_docs():
        for i, doc in enumerate(stream):
            if i < args.skip_docs:
                continue
            text = doc["text"]
            if args.target_docs_per_class:
                hits = set()
                for key, pattern in prefilter_patterns.items():
                    if pattern.search(text):
                        hits.add(key)
                needed = {
                    key for key in hits
                    if prefilter_counts[key] < args.target_docs_per_class
                }
                if not needed:
                    continue
                fold = min(
                    range(3),
                    key=lambda candidate: (
                        sum(prefilter_fold_counts[key][candidate] for key in needed),
                        candidate,
                    ),
                )
                for key in needed:
                    prefilter_counts[key] += 1
                    prefilter_fold_counts[key][fold] += 1
            else:
                fold = None
            yield model.tokenizer.encode(text, add_special_tokens=False), i, fold
            if args.target_docs_per_class and all(
                count >= args.target_docs_per_class
                for count in prefilter_counts.values()
            ):
                return

    tokens_per_row = CTX - DROP
    n_rows = -(-args.scan_tokens // tokens_per_row) + args.batch_rows + 2
    if args.target_docs_per_class:
        def targeted_rows():
            token_buffers = [[], [], []]
            document_buffers = [[], [], []]
            produced = 0
            for tokens, document_id, fold in token_docs():
                if fold is None:
                    raise AssertionError("targeted document lacks a split assignment")
                token_buffers[fold].extend(tokens)
                document_buffers[fold].extend([document_id] * len(tokens))
                while len(token_buffers[fold]) >= CTX - 1 and produced < n_rows:
                    row = [bos] + token_buffers[fold][: CTX - 1]
                    documents = [-1] + document_buffers[fold][: CTX - 1]
                    del token_buffers[fold][: CTX - 1]
                    del document_buffers[fold][: CTX - 1]
                    produced += 1
                    yield (
                        torch.tensor(row, dtype=torch.long),
                        torch.tensor(documents, dtype=torch.long),
                        fold,
                    )
                if produced >= n_rows:
                    return

        rows = targeted_rows()
    else:
        ordinary_docs = (tokens for tokens, _, _ in token_docs())
        rows = (
            (row, None, None)
            for row in pack_token_rows(
                ordinary_docs, ctx=CTX, bos_id=bos, n_rows=n_rows
            )
        )

    w_gpu = whitener.W.to(args.device)
    mu_gpu = whitener.mean.to(args.device)

    lab_acts, lab_cls, lab_fam, lab_tok, lab_seq = [], [], [], [], []
    lab_doc, lab_fold = [], []
    bg_acts = []
    bg_seq = []
    bg_doc, bg_fold = [], []
    scanned = 0
    bg_phase = 0
    kept = {}  # (fam, cls) -> kept count, for --per-class-cap

    @torch.no_grad()
    def process(
        toks: torch.Tensor,
        source_documents: torch.Tensor | None = None,
        source_folds: torch.Tensor | None = None,
    ) -> None:
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
        # Packed rows are the resampling unit available after the corpus has
        # been concatenated without document boundaries.  Preserve them so
        # tuning uses sequence-disjoint splits and bootstraps rather than a
        # random token split.
        row_base = scanned // tokens_per_row
        seq = torch.arange(
            row_base, row_base + toks.shape[0], device=ids.device
        ).repeat_interleave(tokens_per_row)
        documents = (
            source_documents[:, DROP:].reshape(-1)
            if source_documents is not None else torch.full_like(ids, -1)
        )
        folds = (
            source_folds.repeat_interleave(tokens_per_row)
            if source_folds is not None else torch.full_like(ids, -1)
        )
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
            lab_seq.append(seq[hit].cpu())
            lab_doc.append(documents[hit].cpu())
            lab_fold.append(folds[hit].cpu())
        if sum(a.shape[0] for a in bg_acts) < BACKGROUND_CAP:
            sel = torch.arange(bg_phase, xw.shape[0], BACKGROUND_STRIDE)
            keep = sel[~hit_all[sel]]
            bg_acts.append(xw[keep].cpu())
            bg_seq.append(seq[keep].cpu())
            bg_doc.append(documents[keep].cpu())
            bg_fold.append(folds[keep].cpu())
            bg_phase = int((bg_phase + xw.shape[0]) % BACKGROUND_STRIDE)
        scanned += xw.shape[0]

    buf: list[torch.Tensor] = []
    doc_buf: list[torch.Tensor] = []
    fold_buf: list[int] = []
    for row, row_documents, row_fold in rows:
        buf.append(row)
        if row_documents is not None:
            doc_buf.append(row_documents)
            fold_buf.append(int(row_fold))
        if len(buf) < args.batch_rows:
            continue
        process(
            torch.stack(buf),
            torch.stack(doc_buf) if doc_buf else None,
            torch.tensor(fold_buf) if fold_buf else None,
        )
        buf = []
        doc_buf = []
        fold_buf = []
        if scanned >= args.scan_tokens:
            break
        if scanned % 1_000_000 < args.batch_rows * tokens_per_row:
            n_lab = sum(a.shape[0] for a in lab_acts)
            print(f"  scanned {scanned:,} tokens, {n_lab:,} labeled", flush=True)
    if buf and scanned < args.scan_tokens:
        process(
            torch.stack(buf),
            torch.stack(doc_buf) if doc_buf else None,
            torch.tensor(fold_buf) if fold_buf else None,
        )

    acts = torch.cat(lab_acts) if lab_acts else torch.zeros(0, len(sites), d_model)
    cls = torch.cat(lab_cls)
    fam = torch.cat(lab_fam)
    tok_ids = torch.cat(lab_tok)
    sequence_ids = torch.cat(lab_seq)
    bg = torch.cat(bg_acts)[:BACKGROUND_CAP]
    background_sequence_ids = torch.cat(bg_seq)[:BACKGROUND_CAP]
    document_ids = torch.cat(lab_doc)
    fold_ids = torch.cat(lab_fold)
    background_document_ids = torch.cat(bg_doc)[:BACKGROUND_CAP]
    background_fold_ids = torch.cat(bg_fold)[:BACKGROUND_CAP]
    per_fam = ", ".join(
        f"{family} {int((fam == fi).sum()):,}"
        for fi, family in enumerate(args.families)
    )
    print(f"scan done: {scanned:,} tokens, {per_fam}, "
          f"background {bg.shape[0]:,}", flush=True)

    payload = dict(
        acts=acts.numpy().astype(np.float32),
        cls=cls.numpy(),
        fam=fam.numpy(),
        token_ids=tok_ids.numpy(),
        sequence_ids=sequence_ids.numpy(),
        bg_mean=bg.mean(0).numpy(),
        bg_var=bg.var(0).numpy(),
        # NumPy has no native bfloat16 dtype.  Store the exact bf16 payload as
        # uint16 and stamp the encoding in metadata; eval_manifolds restores it
        # with a dtype view.  Transformed bf16 is the validated analysis/store
        # precision, unlike prohibited fp16.
        bg_acts_bf16=(
            bg[: args.background_acts_cap]
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
        ),
        background_sequence_ids=(
            background_sequence_ids[: args.background_acts_cap].numpy()
        ),
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
                "split_unit": (
                    "source_document_preassigned_fold"
                    if args.target_docs_per_class else "packed_sequence"
                ),
                "background_acts_encoding": "bfloat16_uint16_view",
                "background_acts_count": min(args.background_acts_cap, len(bg)),
                "document_sampling": (
                    "family_hit_targeted"
                    if args.target_docs_per_class else "representative_stream"
                ),
                "target_docs_per_class": args.target_docs_per_class,
                "prefilter_doc_counts": {
                    f"{family}:{class_id}": count
                    for (family, class_id), count in prefilter_counts.items()
                },
                "prefilter_fold_counts": {
                    f"{family}:{class_id}": counts
                    for (family, class_id), counts in prefilter_fold_counts.items()
                },
            }
        ),
    )
    if args.target_docs_per_class:
        payload.update(
            document_ids=document_ids.numpy(),
            fold_ids=fold_ids.numpy(),
            background_document_ids=(
                background_document_ids[: args.background_acts_cap].numpy()
            ),
            background_fold_ids=(
                background_fold_ids[: args.background_acts_cap].numpy()
            ),
        )
    np.savez_compressed(args.out / f"zoo_activations{args.tag}.npz", **payload)

    print(f"-> {args.out / f'zoo_activations{args.tag}.npz'}", flush=True)


if __name__ == "__main__":
    main()
