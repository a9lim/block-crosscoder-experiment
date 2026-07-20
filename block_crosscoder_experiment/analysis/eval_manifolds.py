"""Evaluate held-out operational fidelity of cyclic BSC manifolds.

This is the tuning counterpart to :mod:`probe_families`.  It deliberately
does not render a figure or touch the remaining sealed Phase-1 panel. On split A it
selects a family-responsive block and fits the fixed semantic harmonic; split
B supplies the reported tuning metrics.  Split C remains unopened unless
``--open-holdout`` is explicitly passed after configuration freeze.

Legacy Phase-0 activation artifacts lack sequence ids and raw background
examples.  They remain valuable for retrospective screening, but the output
is stamped ``sample_stratified`` and specificity is left null.  New captures
from ``bsc capture-zoo`` carry sequence ids and an optional bf16 background
reservoir.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.analysis.catalog import FAMILIES, MANIFOLD_SPECS
from block_crosscoder_experiment.analysis.manifold_metrics import (
    class_means,
    cyclic_train_eval_metrics,
    fit_cycle_harmonic,
    native_roundness,
    score_cycle_centroids,
    sequence_folds,
    stratified_folds,
)
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder
from block_crosscoder_experiment.store import Whitener

CAP_FAMILIES = {"weekday", "month", "zodiac"}


def _background_tensor(artifact) -> torch.Tensor | None:
    if "bg_acts" in artifact:
        value = artifact["bg_acts"]
        return None if len(value) == 0 else torch.from_numpy(value.astype(np.float32))
    if "bg_acts_bf16" in artifact:
        value = artifact["bg_acts_bf16"]
        if len(value) == 0:
            return None
        raw = torch.from_numpy(value.copy())
        return raw.view(torch.bfloat16).float()
    return None


def _require_equal(label: str, left, right) -> None:
    if left != right:
        raise SystemExit(f"{label} mismatch: {left!r} != {right!r}")


def _load_model(root: Path, device: str) -> tuple[BlockCrosscoder, dict, dict]:
    report = json.loads((root / "report.json").read_text())
    checkpoint = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    mc = checkpoint["model_cfg"]
    cfg = BSCConfig(
        n_blocks=mc["n_blocks"],
        block_dim=mc["block_dim"],
        n_sites=mc["n_sites"],
        d_model=mc["d_model"],
        k=mc["k"],
        lambda_regularizer=mc.get(
            "lambda_regularizer", mc.get("lambda_rank", 0.0)
        ),
        eig_floor=mc.get("eig_floor", 1e-6),
        sv_eps=mc.get("sv_eps", 1e-8),
        seed=mc.get("seed", report.get("seed", 0)),
    )
    model = BlockCrosscoder(cfg, device=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    del checkpoint
    return model, report, mc


def _top_candidates(mean_scores: np.ndarray, per_class: int) -> np.ndarray:
    width = min(per_class, mean_scores.shape[1])
    index = np.argpartition(-mean_scores, width - 1, axis=1)[:, :width]
    return np.unique(index)


def _recall(active_counts: np.ndarray, counts: np.ndarray, block: int) -> dict:
    per_class = active_counts[:, block] / np.maximum(counts, 1)
    return {
        "mean": float(per_class.mean()),
        "min": float(per_class.min()),
        "per_class": per_class.tolist(),
    }


def main() -> None:
    from .artifacts import analysis_dir, load_winner

    winner = load_winner()
    default_analysis = analysis_dir(winner)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, default=default_analysis / "zoo_activations.npz")
    parser.add_argument(
        "--background-acts", type=Path, default=None,
        help="optional representative capture supplying only the background "
        "reservoir; required for specificity when --acts used targeted-doc sampling",
    )
    parser.add_argument("--store", type=Path, default=Path(winner["store"]))
    parser.add_argument("--tokenizer", default=winner["model"])
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--candidate-blocks-per-class", type=int, default=8)
    parser.add_argument("--min-class-recall", type=float, default=0.25)
    parser.add_argument(
        "--min-top1-fraction", type=float, default=0.25,
        help="selection-split responsiveness gate: the candidate must be the "
        "mean-score top block for at least this fraction of classes",
    )
    parser.add_argument("--min-background-lift", type=float, default=2.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--open-holdout",
        action="store_true",
        help="also score split C; use only after the tuning configuration is frozen",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = dict(pair.split("=", 1) for pair in args.runs)
    artifact = np.load(args.acts)
    meta = json.loads(str(artifact["meta"]))
    artifact_families = meta["families"]
    families = args.families or [
        family for family in artifact_families
        if family in MANIFOLD_SPECS
        and MANIFOLD_SPECS[family].topology == "ring"
    ]
    unknown = set(families) - set(artifact_families)
    if unknown:
        raise SystemExit(f"families absent from activation artifact: {sorted(unknown)}")

    acts = torch.from_numpy(artifact["acts"].astype(np.float32, copy=False))
    fam = artifact["fam"]
    cls = artifact["cls"]
    token_ids = artifact["token_ids"]
    sequence_ids = artifact["sequence_ids"] if "sequence_ids" in artifact else None
    assigned_folds = artifact["fold_ids"] if "fold_ids" in artifact else None
    background_artifact = (
        np.load(args.background_acts) if args.background_acts is not None else artifact
    )
    background_meta = json.loads(str(background_artifact["meta"]))
    background = _background_tensor(background_artifact)
    if (
        args.background_acts is None
        and meta.get("document_sampling") == "family_hit_targeted"
    ):
        background = None
        print(
            "targeted-document artifact: background specificity disabled; "
            "pass --background-acts from a representative capture",
            flush=True,
        )
    _require_equal("artifact model", meta.get("model"), args.tokenizer)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    capital_tokens: set[int] = set()
    for family in CAP_FAMILIES & set(families):
        for token in meta["label_maps"][family]:
            if tokenizer.decode([int(token)]).strip()[:1].isupper():
                capital_tokens.add(int(token))
    is_cap = np.isin(token_ids, list(capital_tokens))

    family_rows: dict[str, np.ndarray] = {}
    family_labels: dict[str, np.ndarray] = {}
    family_folds: dict[str, np.ndarray] = {}
    split_kind = (
        "source_document_preassigned"
        if assigned_folds is not None
        else "sequence_disjoint"
        if sequence_ids is not None
        else "sample_stratified"
    )
    for family in families:
        fi = artifact_families.index(family)
        n_classes = MANIFOLD_SPECS[family].fit_count or len(FAMILIES[family])
        mask = (fam == fi) & (cls < n_classes)
        if family in CAP_FAMILIES:
            mask &= is_cap
        rows = np.flatnonzero(mask)
        labels = cls[rows].astype(np.int64)
        if assigned_folds is not None:
            folds = assigned_folds[rows].astype(np.int8)
            for fold in range(3):
                if set(labels[folds == fold]) != set(range(n_classes)):
                    raise ValueError(
                        f"{family}: preassigned fold {fold} is missing a class"
                    )
        elif sequence_ids is None:
            folds = stratified_folds(labels, seed=args.split_seed)
        else:
            folds = sequence_folds(
                sequence_ids[rows], labels, n_folds=3, seed=args.split_seed
            )
        family_rows[family] = rows
        family_labels[family] = labels
        family_folds[family] = folds

    whitener = Whitener.load(args.store / "whitener.pt")
    _require_equal("artifact whitener hash", meta.get("whitener_hash"), whitener.hash)
    _require_equal("artifact sites", list(meta.get("sites", [])), list(whitener.sites))
    if acts.ndim != 3:
        raise SystemExit(f"artifact acts must be [N,S,d], got {tuple(acts.shape)}")
    _require_equal("artifact site dimension", int(acts.shape[1]), len(whitener.sites))
    _require_equal("artifact model dimension", int(acts.shape[2]), int(whitener.mean.shape[1]))
    if background is not None:
        _require_equal(
            "background whitener hash", background_meta.get("whitener_hash"), whitener.hash
        )
        _require_equal(
            "background sites", list(background_meta.get("sites", [])), list(whitener.sites)
        )
        _require_equal("background model", background_meta.get("model"), args.tokenizer)
        _require_equal(
            "background shape",
            tuple(background.shape[1:]),
            (len(whitener.sites), int(whitener.mean.shape[1])),
        )
    renorm_scalars = whitener.site_rms_scalars().view(1, -1, 1)
    output = {
        "protocol": {
            "acts": str(args.acts),
            "split_kind": split_kind,
            "split_seed": args.split_seed,
            "selection_fold": "A",
            "tuning_fold": "B",
            "holdout_fold": "C" if args.open_holdout else "sealed",
            "candidate_blocks_per_class": args.candidate_blocks_per_class,
            "families": families,
            "document_sampling": meta.get("document_sampling", "legacy_unknown"),
            "background_acts": (
                str(args.background_acts or args.acts) if background is not None else None
            ),
            "background_document_sampling": (
                background_meta.get("document_sampling", "legacy_unknown")
                if background is not None else None
            ),
            "burned_development_panel": True,
        },
        "runs": {},
    }

    for name, path in runs.items():
        root = Path(path)
        print(f"\n[{name}] loading {root}", flush=True)
        model, report, mc = _load_model(root, args.device)
        _require_equal("run whitener hash", report.get("whitener_hash"), whitener.hash)
        # Pilot reports predate the explicit `sites` field; the frozen
        # whitener hash already commits to the ordered site list. New reports
        # carry both and must agree independently.
        if report.get("sites") is not None:
            _require_equal("run sites", list(report["sites"]), list(meta["sites"]))
        _require_equal("run site dimension", int(mc["n_sites"]), len(meta["sites"]))
        _require_equal("run model dimension", int(mc["d_model"]), int(acts.shape[2]))
        if not torch.isfinite(model.theta):
            raise RuntimeError(f"{root}: checkpoint has no calibrated theta")
        theta = float(model.theta)
        scale = None
        if report.get("site_renorm_at_load", report.get("site_renorm", False)):
            scale = renorm_scalars

        G, b = model.cfg.n_blocks, model.cfg.block_dim
        accum = {}
        for family in families:
            n_classes = MANIFOLD_SPECS[family].fit_count or len(FAMILIES[family])
            accum[family] = {
                "score": np.zeros((n_classes, G), dtype=np.float64),
                "oper": np.zeros((n_classes, G, b), dtype=np.float64),
                "active": np.zeros((n_classes, G), dtype=np.int64),
                "counts": np.zeros(n_classes, dtype=np.int64),
            }

        with torch.no_grad():
            for start in range(0, len(acts), args.chunk_rows):
                stop = min(start + args.chunk_rows, len(acts))
                xb = acts[start:stop]
                if scale is not None:
                    xb = xb * scale
                z = model.encode(xb.to(args.device))
                scores = model.scores(z)
                active = scores > model.theta
                for family in families:
                    rows = family_rows[family]
                    left = np.searchsorted(rows, start)
                    right = np.searchsorted(rows, stop)
                    if left == right:
                        continue
                    local = np.arange(left, right)
                    keep = family_folds[family][local] == 0
                    if not np.any(keep):
                        continue
                    global_rows = rows[local[keep]]
                    chunk_rows = torch.as_tensor(global_rows - start, device=args.device)
                    labels = family_labels[family][local[keep]]
                    zf = z[chunk_rows]
                    pf = scores[chunk_rows]
                    af = active[chunk_rows]
                    for class_id in np.unique(labels):
                        member_np = labels == class_id
                        member = torch.as_tensor(member_np, device=args.device)
                        a = accum[family]
                        a["score"][class_id] += pf[member].sum(0).double().cpu().numpy()
                        a["oper"][class_id] += (
                            zf[member] * af[member].unsqueeze(-1)
                        ).sum(0).double().cpu().numpy()
                        a["active"][class_id] += af[member].sum(0).cpu().numpy()
                        a["counts"][class_id] += int(member_np.sum())
                del z, scores, active

        background_rate = None
        if background is not None:
            bg_active = np.zeros(G, dtype=np.int64)
            with torch.no_grad():
                for start in range(0, len(background), args.chunk_rows):
                    xb = background[start : start + args.chunk_rows]
                    if scale is not None:
                        xb = xb * scale
                    z = model.encode(xb.to(args.device))
                    bg_active += (model.scores(z) > model.theta).sum(0).cpu().numpy()
            background_rate = bg_active / max(len(background), 1)

        selected: dict[str, int] = {}
        selection_rows: dict[str, dict] = {}
        for family in families:
            a = accum[family]
            counts = a["counts"]
            mean_score = a["score"] / np.maximum(counts[:, None], 1)
            top1 = mean_score.argmax(axis=1)
            ids, top1_counts = np.unique(top1, return_counts=True)
            modal = int(ids[top1_counts.argmax()])
            candidates = _top_candidates(mean_score, args.candidate_blocks_per_class)
            oper_mu = a["oper"] / np.maximum(counts[:, None, None], 1)

            candidate_rows = []
            for block in candidates:
                fit = fit_cycle_harmonic(oper_mu[:, block])
                shape = score_cycle_centroids(oper_mu[:, block], fit)
                recall = _recall(a["active"], counts, int(block))
                family_rate = recall["mean"]
                bg_rate = None if background_rate is None else float(background_rate[block])
                lift = None if bg_rate is None else family_rate / max(bg_rate, 1e-9)
                top1_claimed = int((top1 == block).sum())
                eligible = recall["min"] >= args.min_class_recall and (
                    lift is None or lift >= args.min_background_lift
                ) and top1_claimed >= max(1, int(np.ceil(
                    args.min_top1_fraction * len(counts)
                )))
                candidate_rows.append(
                    {
                        "block": int(block),
                        "eligible": bool(eligible),
                        "shape_r2_A": shape["shape_r2"],
                        "roundness_A": native_roundness(fit),
                        # A ring is only as clear as its weaker property: the
                        # semantic harmonic must fit and its native embedding
                        # must use two comparably scaled directions.  Maximin
                        # avoids an arbitrary weighted sum and prevents a
                        # perfectly ordered line or a round but unordered
                        # cloud from winning the selection split.
                        "topology_floor_A": min(
                            max(shape["shape_r2"], 0.0), native_roundness(fit)
                        ),
                        "recall_A": recall,
                        "background_rate": bg_rate,
                        "background_lift": lift,
                        "top1_claimed_A": top1_claimed,
                    }
                )
            ranked = sorted(
                candidate_rows,
                key=lambda row: (
                    row["eligible"],
                    row["topology_floor_A"],
                    row["shape_r2_A"],
                    row["roundness_A"],
                    row["recall_A"]["mean"],
                    row["top1_claimed_A"],
                ),
                reverse=True,
            )
            choice = ranked[0]
            selected[family] = choice["block"]
            selection_rows[family] = {
                "selected_block": choice["block"],
                "modal_top1_block": modal,
                "modal_top1_claimed_A": int(top1_counts.max()),
                "n_classes": int(len(counts)),
                "candidate_count": int(len(candidates)),
                "eligibility_relaxed": not any(row["eligible"] for row in candidate_rows),
                "selected": choice,
            }

        union = sorted(set(selected.values()))
        union_tensor = torch.tensor(union, device=args.device)
        all_codes = np.empty((len(acts), len(union), b), dtype=np.float32)
        all_active = np.empty((len(acts), len(union)), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(acts), args.chunk_rows):
                stop = min(start + args.chunk_rows, len(acts))
                xb = acts[start:stop]
                if scale is not None:
                    xb = xb * scale
                z = model.encode(xb.to(args.device))[:, union_tensor]
                all_codes[start:stop] = z.float().cpu().numpy()
                all_active[start:stop] = (model.scores(z) > model.theta).cpu().numpy()

        run_result = {
            "path": str(root),
            "config": {
                "G": G,
                "b": b,
                "k": float(mc["k"]),
                "seed": report.get("seed"),
                "lambda_regularizer": report.get(
                    "lam", mc.get("lambda_regularizer", mc.get("lambda_rank"))
                ),
                "lr": report.get("lr"),
                "epochs": report.get("epochs"),
                "optimizer_tokens": int(report.get("total_steps", 0) * 4096),
                "site_renorm": bool(report.get("site_renorm")),
            },
            "constraints": {
                "fvu_pooled_topk": report["eval"]["topk"]["fvu_pooled"],
                "dead_fraction": report.get("dead_frac_final_window"),
                "skip_rate": report.get("skip_rate", 0.0),
                "theta": theta,
            },
            "families": {},
        }
        for family in families:
            block = selected[family]
            column = union.index(block)
            rows = family_rows[family]
            labels = family_labels[family]
            folds = family_folds[family]
            codes = all_codes[rows, column]
            active = all_active[rows, column]
            operational = codes * active[:, None]
            n_classes = MANIFOLD_SPECS[family].fit_count or len(FAMILIES[family])
            decoder_frames = model.D[:, block].detach().float().cpu().numpy()

            def evaluate(eval_fold: int) -> dict:
                train = folds == 0
                evaluate_on = folds == eval_fold
                dense = cyclic_train_eval_metrics(
                    codes[train], labels[train], codes[evaluate_on], labels[evaluate_on],
                    n_classes,
                )
                oper = cyclic_train_eval_metrics(
                    operational[train], labels[train], operational[evaluate_on],
                    labels[evaluate_on], n_classes,
                )
                eval_recall = np.array([
                    active[evaluate_on][labels[evaluate_on] == c].mean()
                    for c in range(n_classes)
                ])
                train_mu, _ = class_means(
                    operational[train], labels[train], n_classes
                )
                eval_mu, _ = class_means(
                    operational[evaluate_on], labels[evaluate_on], n_classes
                )
                site_metrics = []
                for site, frame in zip(meta["sites"], decoder_frames):
                    fit = fit_cycle_harmonic(train_mu @ frame)
                    site_metrics.append({
                        "site": int(site),
                        **score_cycle_centroids(eval_mu @ frame, fit),
                    })
                bg_rate = selection_rows[family]["selected"]["background_rate"]
                family_rate = float(eval_recall.mean())
                return {
                    "dense": dense,
                    "operational": oper,
                    "activation_recall": {
                        "mean": family_rate,
                        "min": float(eval_recall.min()),
                        "per_class": eval_recall.tolist(),
                    },
                    "background_rate": bg_rate,
                    "background_lift": (
                        None if bg_rate is None else family_rate / max(bg_rate, 1e-9)
                    ),
                    "site_contribution": site_metrics,
                    "n_tokens": int(evaluate_on.sum()),
                }

            family_result = {
                **selection_rows[family],
                "B": evaluate(1),
                "C": evaluate(2) if args.open_holdout else "sealed",
            }
            run_result["families"][family] = family_result
            bmetrics = family_result["B"]["operational"]
            print(
                f"  {family:<8} b{block:<5} recall "
                f"{family_result['B']['activation_recall']['mean']:.3f} "
                f"shape {bmetrics['shape_r2']:.3f} round "
                f"{bmetrics['roundness_eval']:.3f} token-cos "
                f"{bmetrics['token_cosine']:.3f}",
                flush=True,
            )

        output["runs"][name] = run_result
        del model, all_codes, all_active, accum
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
