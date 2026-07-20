"""Gate cyclic tuning targets on sequence-held-out residual-stream geometry.

This answers a question upstream of crosscoder tuning: does the source model
itself carry the declared semantic cycle? Split A fits only the fixed first
harmonic to class centroids; split B reports its fidelity and native
roundness at every captured site. Split C is untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from block_crosscoder_experiment.analysis.catalog import FAMILIES, MANIFOLD_SPECS
from block_crosscoder_experiment.analysis.manifold_metrics import (
    class_means,
    cycle_permutation_null,
    fit_cycle_harmonic,
    score_cycle_centroids,
    sequence_folds,
)

CAP_FAMILIES = {"weekday", "month", "zodiac"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--min-shape-r2", type=float, default=0.25)
    parser.add_argument("--min-roundness", type=float, default=0.5)
    parser.add_argument("--min-chord-corr", type=float, default=0.5)
    parser.add_argument("--min-passing-sites", type=int, default=3)
    parser.add_argument("--max-permutation-p", type=float, default=0.01)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    artifact = np.load(args.acts)
    meta = json.loads(str(artifact["meta"]))
    if "sequence_ids" not in artifact and "fold_ids" not in artifact:
        raise SystemExit("raw-stream gate requires a sequence-aware capture")
    artifact_families = list(meta["families"])
    families = args.families or [
        family for family in artifact_families
        if family in MANIFOLD_SPECS
        and MANIFOLD_SPECS[family].topology == "ring"
    ]
    missing = set(families) - set(artifact_families)
    if missing:
        raise SystemExit(f"families absent from activation artifact: {sorted(missing)}")

    acts = artifact["acts"]
    fam = artifact["fam"]
    cls = artifact["cls"]
    token_ids = artifact["token_ids"]
    sequence_ids = artifact["sequence_ids"]
    assigned_folds = artifact["fold_ids"] if "fold_ids" in artifact else None
    capital_tokens: dict[str, set[int]] = {}
    if CAP_FAMILIES & set(families):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(meta["model"])
        for family in CAP_FAMILIES & set(families):
            capital_tokens[family] = {
                int(token)
                for token in meta["label_maps"][family]
                if tokenizer.decode([int(token)]).strip()[:1].isupper()
            }
    result = {
        "protocol": {
            "acts": str(args.acts),
            "split_kind": (
                "source_document_preassigned"
                if assigned_folds is not None else "sequence_disjoint"
            ),
            "selection_fold": "A",
            "evaluation_fold": "B",
            "holdout_fold": "C_sealed",
            "surface_filter": "capitalized_calendar_and_zodiac",
            "split_seed": args.split_seed,
            "gate": {
                "min_shape_r2": args.min_shape_r2,
                "min_roundness": args.min_roundness,
                "min_chord_corr": args.min_chord_corr,
                "min_passing_sites": args.min_passing_sites,
                "max_permutation_p": args.max_permutation_p,
                "permutations": args.permutations,
            },
        },
        "families": {},
    }

    for family in families:
        fi = artifact_families.index(family)
        spec = MANIFOLD_SPECS[family]
        n_classes = spec.fit_count or len(FAMILIES[family])
        mask = (fam == fi) & (cls < n_classes)
        if family in capital_tokens:
            mask &= np.isin(token_ids, list(capital_tokens[family]))
        rows = np.flatnonzero(mask)
        labels = cls[rows].astype(np.int64)
        if assigned_folds is not None:
            folds = assigned_folds[rows].astype(np.int8)
            for fold in range(3):
                if set(labels[folds == fold]) != set(range(n_classes)):
                    raise ValueError(
                        f"{family}: preassigned fold {fold} is missing a class"
                    )
        else:
            folds = sequence_folds(
                sequence_ids[rows], labels, n_folds=3, seed=args.split_seed
            )
        sites = []
        for site_index, site in enumerate(meta["sites"]):
            train_mu, train_counts = class_means(
                acts[rows[folds == 0], site_index], labels[folds == 0], n_classes
            )
            eval_mu, eval_counts = class_means(
                acts[rows[folds == 1], site_index], labels[folds == 1], n_classes
            )
            fit = fit_cycle_harmonic(train_mu)
            metrics = score_cycle_centroids(eval_mu, fit)
            null = cycle_permutation_null(
                train_mu,
                eval_mu,
                n_permutations=args.permutations,
                seed=args.split_seed + site_index,
            )
            passed = (
                metrics["shape_r2"] >= args.min_shape_r2
                and metrics["roundness_eval"] >= args.min_roundness
                and metrics["chord_corr"] >= args.min_chord_corr
                and null["permutation_p_topology"] <= args.max_permutation_p
            )
            sites.append({
                "site": int(site),
                "passes": bool(passed),
                "train_count_min": int(train_counts.min()),
                "eval_count_min": int(eval_counts.min()),
                **metrics,
                **null,
            })
        passing = sum(site["passes"] for site in sites)
        family_result = {
            "n_classes": n_classes,
            "n_tokens": int(len(rows)),
            "passing_sites": passing,
            "enters_tuning_objective": passing >= args.min_passing_sites,
            "sites": sites,
        }
        result["families"][family] = family_result
        verdict = "PASS" if family_result["enters_tuning_objective"] else "FAIL"
        print(f"{family}: {verdict} ({passing}/{len(sites)} sites)", flush=True)
        for site in sites:
            print(
                f"  L{site['site']:>2}: shape {site['shape_r2']:.3f} "
                f"round {site['roundness_eval']:.3f} "
                f"chord {site['chord_corr']:.3f} "
                f"p {site['permutation_p_topology']:.4f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
