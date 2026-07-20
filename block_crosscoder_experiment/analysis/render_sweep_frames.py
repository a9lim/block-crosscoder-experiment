"""Render temporary split-B frame figures for selected manifold-sweep runs.

Unlike the canonical winner catalog, this renderer never consults
``data/winner.json`` and never opens split C.  It reads the exact A-selected
block identities from an ``eval-manifolds`` report, computes sequence-disjoint
split-B class means from the corresponding activation capture, and projects
those means through each checkpoint's decoder frame.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from plotly.offline import get_plotlyjs

from block_crosscoder_experiment.analysis.catalog import FAMILIES, MANIFOLD_SPECS
from block_crosscoder_experiment.analysis.figures import (
    _stack_figure,
    _stack_planes,
    _write_figure,
)
from block_crosscoder_experiment.analysis.manifold_metrics import (
    sequence_folds,
    stratified_folds,
)
from block_crosscoder_experiment.store import Whitener

CAP_FAMILIES = {"month", "weekday", "zodiac"}


def _config_summary(config: dict) -> str:
    """Compact, explicit configuration label for the sweep index."""

    tokens = int(config.get("optimizer_tokens", 0))
    token_label = f"{tokens / 1_000_000:.0f}M" if tokens else "? tokens"
    gauge = "renorm" if config.get("site_renorm") else "primary"
    return (
        f"G{config.get('G', '?')} b{config.get('b', '?')} "
        f"k{config.get('k', '?'):g}; lr {config.get('lr', '?')}; "
        f"λ {config.get('lambda_rank', '?')}; {gauge}; "
        f"{token_label}; seed {config.get('seed', '?')}"
    )


def _upsert_figure(manifest: dict, figure: dict) -> None:
    """Replace one run/family view while preserving other catalog cohorts."""

    key = (figure["run"], figure["family"])
    manifest["figures"] = [
        current for current in manifest["figures"]
        if (current["run"], current["family"]) != key
    ]
    manifest["figures"].append(figure)


def _write_index(out: Path, manifest: dict) -> None:
    rows_html = []
    for figure in sorted(
        manifest["figures"],
        key=lambda item: (item["cohort"], item["label"], item["family"]),
    ):
        rows_html.append(
            f"<tr><td>{html.escape(figure['label'])}</td>"
            f"<td>{html.escape(figure['cohort'])}</td>"
            f"<td>{html.escape(figure['config'])}</td>"
            f"<td>{figure['family']}</td><td>b{figure['block']}</td>"
            f"<td>{figure['fvu']:.4f}</td>"
            f"<td>{figure['shape_r2']:.3f}</td>"
            f"<td>{figure['roundness']:.3f}</td>"
            f"<td>{figure['token_cosine']:.3f}</td>"
            f"<td><a href=\"{html.escape(figure['file'])}\">frames</a>"
            "</td></tr>"
        )
    (out / "index.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>BSC sweep and control frame figures</title><style>"
        "body{font:15px/1.5 system-ui,sans-serif;max-width:1440px;margin:3rem auto;"
        "padding:0 1.5rem;color:#0b0b0b;background:#fcfcfb}"
        ".table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;"
        "min-width:1240px}th,td{padding:.55rem .7rem;border-bottom:1px solid "
        "#e1e0d9;text-align:left;vertical-align:top}th{white-space:nowrap}"
        "td:nth-child(n+5):not(:last-child){font-variant-numeric:tabular-nums}"
        "a{color:#185fa5}</style></head><body>"
        "<h1>BSC sweep and control frame figures</h1>"
        "<p>Temporary split-B diagnostics; split C remains sealed. Compare "
        "metrics within cohorts: legacy sample-stratified controls are not "
        "numerically interchangeable with the sequence-disjoint sweep.</p>"
        "<div class=\"table-wrap\"><table><thead><tr><th>run</th>"
        "<th>cohort</th><th>configuration</th><th>family</th><th>block</th>"
        "<th>FVU</th><th>shape R²</th><th>roundness</th><th>token cosine</th>"
        f"<th>view</th></tr></thead><tbody>{''.join(rows_html)}"
        "</tbody></table></div></body></html>"
    )


def _heldout_class_means(
    acts: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return split-B means as ``[site, class, d]`` plus class counts."""

    heldout = folds == 1
    heldout_rows = rows[heldout]
    heldout_labels = labels[heldout]
    counts = np.bincount(heldout_labels, minlength=n_classes)[:n_classes]
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"split B is missing classes {missing}")
    means = np.stack([
        acts[heldout_rows[heldout_labels == class_id]].mean(0)
        for class_id in range(n_classes)
    ])
    return means.transpose(1, 0, 2), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--cohort",
        help="comparison cohort shown in the combined index",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="add or replace these run/family views in an existing catalog",
    )
    parser.add_argument(
        "--families", nargs="+", choices=tuple(MANIFOLD_SPECS),
        default=["month", "weekday"],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text())
    if evaluation["protocol"].get("holdout_fold") != "sealed":
        raise SystemExit("sweep-frame rendering requires split C to remain sealed")
    unknown_runs = set(args.runs) - set(evaluation["runs"])
    if unknown_runs:
        raise SystemExit(f"runs absent from evaluation: {sorted(unknown_runs)}")

    artifact = np.load(args.acts)
    meta = json.loads(str(artifact["meta"]))
    artifact_families = list(meta["families"])
    unknown_families = set(args.families) - set(artifact_families)
    if unknown_families:
        raise SystemExit(
            f"families absent from activation artifact: {sorted(unknown_families)}"
        )
    if meta.get("model") != args.tokenizer:
        raise SystemExit(
            f"artifact/tokenizer mismatch: {meta.get('model')} != {args.tokenizer}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    capital_tokens: set[int] = set()
    for family in CAP_FAMILIES & set(args.families):
        for token in meta["label_maps"][family]:
            decoded = tokenizer.decode([int(token)]).strip()
            if decoded[:1].isupper():
                capital_tokens.add(int(token))

    acts = artifact["acts"].astype(np.float32, copy=False)
    fam = artifact["fam"]
    cls = artifact["cls"]
    token_ids = artifact["token_ids"]
    sequence_ids = artifact["sequence_ids"] if "sequence_ids" in artifact else None
    assigned_folds = artifact["fold_ids"] if "fold_ids" in artifact else None
    is_cap = np.isin(token_ids, list(capital_tokens))

    means: dict[str, np.ndarray] = {}
    class_counts: dict[str, np.ndarray] = {}
    for family in args.families:
        family_index = artifact_families.index(family)
        spec = MANIFOLD_SPECS[family]
        n_classes = spec.fit_count or len(FAMILIES[family])
        mask = (fam == family_index) & (cls < n_classes)
        if family in CAP_FAMILIES:
            mask &= is_cap
        rows = np.flatnonzero(mask)
        labels = cls[rows].astype(np.int64)
        if assigned_folds is not None:
            folds = assigned_folds[rows].astype(np.int8)
        elif sequence_ids is not None:
            folds = sequence_folds(
                sequence_ids[rows], labels, n_folds=3, seed=0
            )
        else:
            folds = stratified_folds(labels, seed=0)
        means[family], class_counts[family] = _heldout_class_means(
            acts, rows, labels, folds, n_classes
        )

    whitener = Whitener.load(args.store / "whitener.pt")
    if meta.get("whitener_hash") != whitener.hash:
        raise SystemExit("activation artifact and store whitener hashes differ")
    sites = [int(site) for site in meta["sites"]]
    renorm = whitener.site_rms_scalars().numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    assets = args.out.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    plotly_runtime = assets / "plotly.min.js"
    if not plotly_runtime.exists():
        plotly_runtime.write_text(get_plotlyjs())

    manifest_path = args.out / "manifest.json"
    if args.append and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("split") != "B" or manifest.get("holdout_C") != "sealed":
            raise SystemExit("existing sweep-frame catalog has incompatible folds")
    else:
        manifest = {
            "format": 2,
            "split": "B",
            "holdout_C": "sealed",
            "sources": [],
            "figures": [],
        }
    cohort = args.cohort or evaluation["protocol"].get(
        "split_kind", "unspecified"
    )
    source = {
        "cohort": cohort,
        "evaluation": str(args.evaluation),
        "acts": str(args.acts),
        "split_kind": evaluation["protocol"].get("split_kind"),
    }
    manifest["sources"] = [
        current for current in manifest.get("sources", [])
        if not (
            current["cohort"] == cohort
            and current["evaluation"] == str(args.evaluation)
        )
    ]
    manifest["sources"].append(source)
    for run_name in args.runs:
        run = evaluation["runs"][run_name]
        display_name = run.get("display_name", run_name)
        root = Path(run["path"])
        report = json.loads((root / "report.json").read_text())
        checkpoint = torch.load(
            root / "latest.pt", map_location="cpu", weights_only=False
        )
        decoder = checkpoint["model"]["D"].float().numpy()
        site_scales = (
            renorm
            if report.get("site_renorm_at_load", report.get("site_renorm", False))
            else np.ones(len(sites), dtype=np.float32)
        )
        if decoder.shape[0] != len(sites):
            raise ValueError(f"{run_name}: decoder/site mismatch")

        for family in args.families:
            family_result = run["families"][family]
            block = int(family_result["selected_block"])
            frame = decoder[:, block]
            framed = np.stack([
                (
                    means[family][site] - means[family][site].mean(0)
                ) * site_scales[site] @ frame[site].T
                for site in range(len(sites))
            ])
            operational = family_result["B"]["operational"]
            recall = family_result["B"]["activation_recall"]
            eligible = bool(family_result["selected"]["eligible"])
            subtitle = (
                f"{display_name}; {cohort}; b{block}; pooled FVU "
                f"{run['constraints']['fvu_pooled_topk']:.4f}; "
                f"A-selected eligible={'yes' if eligible else 'no'}.",
                f"Held-out B: shape R² {operational['shape_r2']:.3f}, "
                f"roundness {operational['roundness_eval']:.3f}, chord "
                f"{operational['chord_corr']:.3f}, token cosine "
                f"{operational['token_cosine']:.3f}; recall "
                f"{recall['mean']:.3f} (min {recall['min']:.3f}). "
                f"n={int(class_counts[family].sum()):,}; split C sealed.",
            )
            filename = f"{run_name}-{family}-frames.html"
            _write_figure(
                _stack_figure(
                    _stack_planes(framed, MANIFOLD_SPECS[family]),
                    MANIFOLD_SPECS[family],
                    sites,
                    f"{family}: {display_name} frame geometry on split B",
                    subtitle,
                    "1st harmonic",
                ),
                args.out / filename,
            )
            _upsert_figure(manifest, {
                "run": run_name,
                "label": display_name,
                "cohort": cohort,
                "config": _config_summary(run["config"]),
                "family": family,
                "block": block,
                "file": filename,
                "eligible": eligible,
                "fvu": run["constraints"]["fvu_pooled_topk"],
                "shape_r2": operational["shape_r2"],
                "roundness": operational["roundness_eval"],
                "token_cosine": operational["token_cosine"],
            })
            print(f"{run_name} {family}: b{block} -> {args.out / filename}")
        del checkpoint, decoder

    manifest["figures"].sort(
        key=lambda item: (item["cohort"], item["label"], item["family"])
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _write_index(args.out, manifest)


if __name__ == "__main__":
    main()
