"""Phase-0 target: ring hunt on gemma-3-4b layer 22 (gemma-scope-2 16k SAE).

The actual Phase-0 science — the control (docs/findings-phase0-control.md)
passed, so a null here is interpretable at demonstrated power. Pipeline
mirrors run_phase0_control.py with three deltas:

- Families: weekday + month only. Year dies at the tokenizer — gemma's
  SentencePiece has no single-token years, so the class can't sit in one
  residual position (matches Engels' own non-GPT-2 scope).
- Unknown-cluster scan stage (design §Phase 0): co-fire-gated harmonic
  contrast over ALL clusters, frequency-matched random-member nulls,
  BH over the search width. Runs LAST — the labeled verdicts land early
  in the log; the scan is a surfacing instrument with hours of nulls.
- 16,384 features → the similarity/eigh stages are 2.5× smaller than the
  control's; same code paths.

  python scripts/run_phase0_gemma.py --stage all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

RELEASE = "gemma-scope-2-4b-pt-res"
SAE_ID = "layer_22_width_16k_l0_medium"
TOKENIZER = "google/gemma-3-4b-pt"
N_CLUSTERS = 1000
FAMILIES = ("weekday", "month")
N_CLASSES = {"weekday": 7, "month": 12}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_store(root: Path):
    from block_crosscoder_experiment.phase0.harvest import CodeStore

    return CodeStore(root)


def load_decoder(device: str) -> torch.Tensor:
    from sae_lens import SAE

    sae = SAE.from_pretrained(RELEASE, SAE_ID, device="cpu")
    return sae.W_dec.detach().to(device)


def stage_cluster(store, out: Path, device: str) -> dict[str, torch.Tensor]:
    """Two geometric candidate sources.

    Gemma-16k decoders are far more orthogonal than Bloom's GPT-2 SAE
    (median max-neighbor cosine 0.218 vs 0.547 — 6.4× vs 32×
    overcomplete), so Engels-style spectral clustering degenerates toward
    singletons + a background blob. The kNN-graph method (Engels' own
    answer for their 65k Mistral SAEs) handles orthogonal-bulk
    dictionaries gracefully; both run as candidate sources.
    """
    from block_crosscoder_experiment.phase0.clustering import (
        angular_similarity,
        knn_graph_clusters,
        spectral_clusters,
    )

    decoder = load_decoder(device)
    labelings: dict[str, torch.Tensor] = {}
    spath = out / "cluster_labels.pt"
    if spath.exists():
        labelings["spectral"] = torch.load(spath, weights_only=True)
    else:
        labelings["spectral"] = spectral_clusters(
            angular_similarity(decoder), N_CLUSTERS, seed=0
        ).cpu()
        torch.save(labelings["spectral"], spath)
    gpath = out / "graph_labels.pt"
    if gpath.exists():
        labelings["graph"] = torch.load(gpath, weights_only=True)
    else:
        labelings["graph"] = knn_graph_clusters(decoder).cpu()
        torch.save(labelings["graph"], gpath)
    return labelings


def stage_coactivation(store, out: Path, device: str) -> torch.Tensor:
    path = out / "coact_labels.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    from block_crosscoder_experiment.phase0.clustering import spectral_clusters

    f = store.n_features
    counts = torch.zeros(f, f, device=device)
    for chunk in store.iter_dense_chunks(chunk=16384, device=device):
        b = chunk.gt(0).to(torch.bfloat16)
        counts += (b.T @ b).float()
        del chunk, b
    diag = counts.diagonal().clamp_min(1.0).sqrt()
    counts.div_(diag.unsqueeze(0)).div_(diag.unsqueeze(1))
    if device == "cuda":
        torch.cuda.empty_cache()
    labels = spectral_clusters(counts, N_CLUSTERS, seed=0).cpu()
    torch.save(labels, path)
    return labels


def _family_label_vector(store, family: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    from block_crosscoder_experiment.phase0.labels import (
        build_label_map,
        label_tokens,
    )

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return label_tokens(
        store.token_ids().to(torch.long), build_label_map(tok, family)
    )


def _family_affinity(
    store, labels: torch.Tensor, class_ids: torch.Tensor, n_classes: int
) -> list[tuple[int, float, int, int]]:
    class_ids = class_ids.cpu()
    scores = []
    for cid in range(int(labels.max()) + 1):
        members = (labels == cid).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        fired_rows = store.member_row_union(members).cpu().to(torch.long)
        n_fired = int(fired_rows.shape[0])
        if n_fired < 50:
            continue
        fired_classes = class_ids[fired_rows]
        fired_classes = fired_classes[fired_classes >= 0]
        coverage = int(fired_classes.unique().numel())
        affinity = float(fired_classes.shape[0]) / n_fired
        scores.append((cid, affinity, n_fired, coverage))
    scores.sort(key=lambda t: (-(t[3] >= n_classes - 1), -t[1]))
    return scores


def stage_battery(
    store, labelings: dict[str, torch.Tensor], out: Path, device: str
) -> dict:
    from block_crosscoder_experiment.phase0.battery import run_cluster_battery

    decoder = load_decoder(device)
    report: dict = {}
    for family in FAMILIES:
        class_ids = _family_label_vector(store, family)
        fam: dict = {}
        for branch, labels in labelings.items():
            affinity = _family_affinity(
                store, labels, class_ids, N_CLASSES[family]
            )
            top = affinity[:3]
            entry: dict = {"top_affinity": [
                {"cluster": c, "affinity": round(a, 4), "n_fired": n,
                 "coverage": cov}
                for c, a, n, cov in top
            ], "batteries": []}
            for cid, aff, _, _ in top:
                members = (labels == cid).nonzero(as_tuple=True)[0]
                battery = run_cluster_battery(
                    store,
                    decoder,
                    members,
                    class_ids=class_ids,
                    n_classes=N_CLASSES[family],
                    n_perm=200,
                    seed=0,
                )
                battery["cluster"] = cid
                battery["affinity"] = aff
                battery["members"] = members.tolist()
                entry["batteries"].append(_jsonable(battery))
            if entry["batteries"]:
                entry["battery"] = entry["batteries"][0]
            fam[branch] = entry
        report[family] = fam
    (out / "family_battery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def stage_ranking(store, labels: torch.Tensor, out: Path, device: str) -> dict:
    from block_crosscoder_experiment.phase0.battery import (
        cluster_restricted_reconstruction,
    )
    from block_crosscoder_experiment.phase0.rings import (
        pca_projections,
        plane_scan,
    )

    path = out / "engels_ranking.json"
    if path.exists():
        return json.loads(path.read_text())
    decoder = load_decoder(device)
    scores: dict[int, float] = {}
    for cid in range(N_CLUSTERS):
        members = (labels == cid).nonzero(as_tuple=True)[0]
        if members.numel() < 2:
            continue
        recon, kept = cluster_restricted_reconstruction(
            store, decoder, members, max_tokens=20_000, seed=0
        )
        if kept.shape[0] < 200:
            continue
        proj, explained = pca_projections(recon, k=5)
        scan = plane_scan(proj, explained=explained, mixture_steps=1000, seed=0)
        scores[cid] = scan["mean"]["score"]
    ranking = sorted(scores, key=lambda c: -scores[c])
    result = {"scores": {str(c): scores[c] for c in ranking}, "ranking": ranking}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def stage_scan(
    store, labelings: dict[str, torch.Tensor], out: Path, device: str
) -> dict:
    """Unknown-cluster surfacing scan + BH over the search width.

    Runs over BOTH candidate branches (spectral + kNN graph) — in gemma's
    near-orthogonal geometry the multi-member graph components are the
    natural co-fire candidates. Exact duplicate member sets across branches
    are scanned once; BH runs over the deduplicated combined width.
    """
    from block_crosscoder_experiment.phase0.battery import unknown_cluster_scan
    from block_crosscoder_experiment.phase0.nulls import benjamini_hochberg

    path = out / "unknown_scan.json"
    if path.exists():
        return json.loads(path.read_text())
    decoder = load_decoder(device)
    clusters: dict[str, torch.Tensor] = {}
    seen: dict[tuple, str] = {}
    duplicates = 0
    for branch, labels in labelings.items():
        uniq, cnt = labels.unique(return_counts=True)
        for cid, c in zip(uniq.tolist(), cnt.tolist()):
            if cid < 0 or c < 2:
                continue
            members = (labels == cid).nonzero(as_tuple=True)[0]
            key = tuple(members.tolist())
            if key in seen:
                duplicates += 1
                continue
            seen[key] = f"{branch}:{cid}"
            clusters[f"{branch}:{cid}"] = members
    per_branch = {
        b: sum(k.startswith(f"{b}:") for k in clusters) for b in labelings
    }
    print(
        f"scan width: {len(clusters)} clusters {per_branch} "
        f"({duplicates} cross-branch duplicates scanned once)",
        flush=True,
    )
    results = unknown_cluster_scan(
        store,
        decoder,
        clusters,
        n_null_draws=100,
        mixture_steps=400,
        firing_counts=store.firing_counts(),
        seed=0,
        progress=lambda msg: print(msg, flush=True),
    )
    tested = {c: r for c, r in results.items() if "contrast" in r}
    pvals = [tested[c]["p"] for c in sorted(tested)]
    mask = benjamini_hochberg(pvals, alpha=0.05)
    flagged = [c for c, m in zip(sorted(tested), mask.tolist()) if m]
    summary = {
        "n_tested": len(tested),
        "n_gated_out": len(results) - len(tested),
        "bh_flagged": flagged,
        "results": _jsonable({str(c): r for c, r in results.items()}),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def stage_figures(store, labels: torch.Tensor, report: dict, out: Path, device: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from block_crosscoder_experiment.phase0.battery import (
        cluster_restricted_reconstruction,
    )
    from block_crosscoder_experiment.phase0.rings import pca_projections

    def plot_battery(
        family: str, branch: str, battery: dict, class_ids: torch.Tensor
    ) -> None:
        cid = battery["cluster"]
        members = torch.tensor(battery["members"])
        recon, kept = cluster_restricted_reconstruction(store, decoder, members)
        proj, _ = pca_projections(recon, k=5)
        best = tuple(battery.get("circular_plane") or battery["plane_scan"]["best_plane"])
        ids = class_ids[kept.cpu()]
        labeled = ids >= 0
        pts = proj[:, list(best)].cpu()[labeled]
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], c=ids[labeled], cmap="hsv", s=8, alpha=0.7,
            vmin=0, vmax=N_CLASSES[family],
        )
        ax.set_title(
            f"gemma {family} {branch} cluster {cid} — "
            f"PCs {best[0]+1}/{best[1]+1}, "
            f"circ {battery['circular']:.2f} (p={battery['circular_p']:.3f})"
        )
        fig.colorbar(sc, ax=ax, label="class")
        fig.savefig(fig_dir / f"{family}_{branch}_cluster{cid}.png", dpi=150)
        plt.close(fig)

    decoder = load_decoder(device)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    for family in FAMILIES:
        class_ids = _family_label_vector(store, family)
        for branch, entry in report.get(family, {}).items():
            for battery in entry.get("batteries", []):
                if "circular" in battery:
                    plot_battery(family, branch, battery, class_ids)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path,
        default=Path("/data/stores/bcc-phase0/gemma3_4b_l22_pile"),
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "cluster", "battery", "ranking", "coact", "figures", "scan"],
    )
    args = parser.parse_args()
    device = _device()
    store = load_store(args.store)
    store.load_csc()
    out = args.store / "target_run"
    out.mkdir(exist_ok=True)

    labelings = stage_cluster(store, out, device)
    for branch, labels in labelings.items():
        sizes = torch.bincount(labels)
        print(
            f"{branch} clusters: {int((sizes > 0).sum())} nonempty, "
            f"median size {float(sizes[sizes > 0].median()):.0f}, "
            f"max {int(sizes.max())}"
        )
    spectral = labelings["spectral"]
    if args.stage in ("all", "battery", "figures"):
        report = stage_battery(store, labelings, out, device)
        for family in FAMILIES:
            for branch, entry in report[family].items():
                printed = False
                for b in entry.get("batteries", []):
                    if "circular" in b:
                        print(
                            f"{family}/{branch}: cluster {b['cluster']} "
                            f"affinity {b['affinity']:.2f} "
                            f"n_members {b['n_members']} "
                            f"circ {b['circular']:.3f} (p={b['circular_p']:.4f}) "
                            f"plane {b['circular_plane']} "
                            f"ngon {b['ngon']['alignment']:.2f}"
                        )
                        printed = True
                if not printed:
                    print(
                        f"{family}/{branch}: no battery-eligible cluster — "
                        f"{entry['top_affinity']}"
                    )
    if args.stage in ("all", "ranking"):
        ranking = stage_ranking(store, spectral, out, device)
        order = ranking["ranking"]
        report = json.loads((out / "family_battery.json").read_text())
        for family in FAMILIES:
            for b in report[family].get("spectral", {}).get("batteries", []):
                cid = b["cluster"]
                rank = order.index(cid) + 1 if cid in order else None
                print(f"{family} cluster {cid}: Engels rank {rank}/{len(order)}")
    if args.stage in ("all", "coact"):
        coact = stage_coactivation(store, out, device)
        report = json.loads((out / "family_battery.json").read_text())
        for family in FAMILIES:
            for branch, entry in report[family].items():
                for b in entry.get("batteries", []):
                    members = torch.tensor(b["members"])
                    geo = set(members.tolist())
                    ids, cnts = coact[members].unique(return_counts=True)
                    best = int(ids[cnts.argmax()])
                    co = set((coact == best).nonzero(as_tuple=True)[0].tolist())
                    jac = len(geo & co) / len(geo | co)
                    print(
                        f"{family}/{branch} cluster {b['cluster']}: "
                        f"co-activation best-match Jaccard {jac:.2f}"
                    )
    if args.stage in ("all", "figures"):
        report = json.loads((out / "family_battery.json").read_text())
        stage_figures(store, spectral, report, out, device)
        print(f"figures -> {out / 'figures'}")
    if args.stage in ("all", "scan"):
        summary = stage_scan(store, labelings, out, device)
        print(
            f"scan: {summary['n_tested']} tested, "
            f"{summary['n_gated_out']} gated out, "
            f"BH flagged: {summary['bh_flagged']}"
        )


if __name__ == "__main__":
    main()
