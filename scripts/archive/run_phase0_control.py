"""Phase-0 positive control: Engels weekday/month/year rings on GPT-2.

Runs on jobe (CUDA) against the CodeStore from harvest_phase0_control.py.
Observational only (P2: GPT-2 scores near-chance on the causal tasks).

Pipeline (design §Phase 0):
 1. Spectral clustering of the 24,576 layer-7 decoder rows at Engels'
    n=1000 scale (angular similarity, App. F.1).
 2. Family identification: for each cyclic family (weekday/month/year),
    clusters ranked by labeled-firing affinity.
 3. Labeled battery on the top family clusters: PC-plane scan, cone
    check, held-out circular decoding vs class-permutation null,
    n-gon alignment, harmonic power.
 4. Engels ranking: all 1000 clusters scored by mean-plane (1−M)·S —
    the family clusters' ranks are the reproduction target (paper:
    weekday 9, month 28, year 15 of 1000).
 5. Co-activation branch (P15): same spectral machinery on binarized
    co-occurrence similarity; family-cluster membership compared across
    branches (Jaccard).
 6. Figures: best-plane scatters colored by class for each family.

Every stage caches under <store>/control_run/ so reruns are incremental.

  python scripts/run_phase0_control.py --stage all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.7.hook_resid_pre"
N_CLUSTERS = 1000
FAMILIES = ("weekday", "month", "year")
N_CLASSES = {"weekday": 7, "month": 12, "year": 100}
PAPER_RANKS = {"weekday": 9, "month": 28, "year": 15}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_store(root: Path):
    from block_crosscoder_experiment.phase0.harvest import CodeStore

    return CodeStore(root)


def load_decoder(device: str) -> torch.Tensor:
    from sae_lens import SAE

    sae = SAE.from_pretrained(RELEASE, SAE_ID, device="cpu")
    return sae.W_dec.detach().to(device)


def stage_cluster(store, out: Path, device: str) -> torch.Tensor:
    """Geometric branch: angular-similarity spectral clustering."""
    from block_crosscoder_experiment.phase0.clustering import (
        angular_similarity,
        spectral_clusters,
    )

    path = out / "cluster_labels.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    decoder = load_decoder(device)
    sim = angular_similarity(decoder)
    labels = spectral_clusters(sim, N_CLUSTERS, seed=0).cpu()
    torch.save(labels, path)
    return labels


def stage_coactivation(store, out: Path, device: str) -> torch.Tensor:
    """P15 branch: spectral clustering on binarized co-occurrence."""
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
    # In-place normalization: a separate sim matrix plus the 24k² eigh
    # workspace OOM'd the first run on the 4090.
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

    tok = AutoTokenizer.from_pretrained("gpt2")
    mapping = build_label_map(tok, family)
    return label_tokens(store.token_ids().to(torch.long), mapping)


def _family_affinity(
    store, labels: torch.Tensor, class_ids: torch.Tensor, n_classes: int
) -> list[tuple[int, float, int, int]]:
    """Rank clusters for a family: class coverage first, then affinity.

    Coverage (how many of the family's classes appear among the cluster's
    fired-and-labeled tokens) outranks affinity — a single-day feature
    cluster is 100% weekday-affine but is not the ring. Requires
    store.load_csc().
    """
    class_ids = class_ids.cpu()
    scores = []
    for cid in range(N_CLUSTERS):
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


def stage_battery(store, labels: torch.Tensor, out: Path, device: str) -> dict:
    from block_crosscoder_experiment.phase0.battery import run_cluster_battery

    decoder = load_decoder(device)
    report: dict = {}
    for family in FAMILIES:
        class_ids = _family_label_vector(store, family)
        affinity = _family_affinity(store, labels, class_ids, N_CLASSES[family])
        top = affinity[:3]
        fam: dict = {"top_affinity": [
            {"cluster": c, "affinity": round(a, 4), "n_fired": n, "coverage": cov}
            for c, a, n, cov in top
        ], "batteries": []}
        # Battery all three candidates: affinity top-1 stays the primary
        # (no post-hoc selection), the others are recorded diagnostics —
        # the paper's ring may sit in a tighter lower-affinity cluster.
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
            fam["batteries"].append(_jsonable(battery))
        if fam["batteries"]:
            fam["battery"] = fam["batteries"][0]
        report[family] = fam
    (out / "family_battery.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def stage_ranking(store, labels: torch.Tensor, out: Path, device: str) -> dict:
    """Engels ranking of all clusters by mean-plane (1−M)·S."""
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
        scan = plane_scan(
            proj, explained=explained, mixture_steps=1000, seed=0
        )
        scores[cid] = scan["mean"]["score"]
    ranking = sorted(scores, key=lambda c: -scores[c])
    result = {
        "scores": {str(c): scores[c] for c in ranking},
        "ranking": ranking,
    }
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def stage_figures(store, labels: torch.Tensor, report: dict, out: Path, device: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from block_crosscoder_experiment.phase0.battery import (
        cluster_restricted_reconstruction,
    )
    from block_crosscoder_experiment.phase0.rings import pca_projections

    def plot_battery(family: str, battery: dict, class_ids: torch.Tensor) -> None:
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
            f"{family} cluster {cid} — PCs {best[0]+1}/{best[1]+1}, "
            f"circ {battery['circular']:.2f} (p={battery['circular_p']:.3f})"
        )
        fig.colorbar(sc, ax=ax, label="class")
        fig.savefig(fig_dir / f"{family}_cluster{cid}.png", dpi=150)
        plt.close(fig)

    decoder = load_decoder(device)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    for family in FAMILIES:
        class_ids = _family_label_vector(store, family)
        for battery in report.get(family, {}).get("batteries", []):
            if "circular" in battery:
                plot_battery(family, battery, class_ids)


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
        "--store", type=Path, default=Path("/data/stores/bcc-phase0/gpt2_l7_owt")
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "cluster", "battery", "ranking", "coact", "figures"],
    )
    args = parser.parse_args()
    device = _device()
    store = load_store(args.store)
    store.load_csc()  # consolidate once; selection becomes O(members)
    out = args.store / "control_run"
    out.mkdir(exist_ok=True)

    labels = stage_cluster(store, out, device)
    sizes = torch.bincount(labels, minlength=N_CLUSTERS)
    print(
        f"clusters: {int((sizes > 0).sum())}/{N_CLUSTERS} nonempty, "
        f"median size {float(sizes[sizes > 0].median()):.0f}, max {int(sizes.max())}"
    )
    if args.stage in ("all", "battery", "figures"):
        report = stage_battery(store, labels, out, device)
        for family in FAMILIES:
            fam = report[family]
            printed = False
            for b in fam.get("batteries", []):
                if "circular" in b:
                    print(
                        f"{family}: cluster {b['cluster']} affinity {b['affinity']:.2f} "
                        f"n_members {b['n_members']} "
                        f"circ {b['circular']:.3f} (p={b['circular_p']:.4f}) "
                        f"ngon {b['ngon']['alignment']:.2f} "
                        f"peak-m {max(b['harmonics'], key=lambda m: b['harmonics'][m])}"
                    )
                    printed = True
            if not printed:
                print(f"{family}: no battery-eligible cluster — {fam['top_affinity']}")
    if args.stage in ("all", "ranking"):
        ranking = stage_ranking(store, labels, out, device)
        order = ranking["ranking"]
        report = json.loads((out / "family_battery.json").read_text())
        for family in FAMILIES:
            for b in report[family].get("batteries", []):
                cid = b["cluster"]
                rank = order.index(cid) + 1 if cid in order else None
                print(
                    f"{family} cluster {cid}: Engels rank {rank}/{len(order)} "
                    f"(paper: {PAPER_RANKS[family]}/1000)"
                )
    if args.stage in ("all", "coact"):
        coact = stage_coactivation(store, out, device)
        report = json.loads((out / "family_battery.json").read_text())
        for family in FAMILIES:
            for b in report[family].get("batteries", []):
                members = torch.tensor(b["members"])
                geo = set(members.tolist())
                ids, cnts = coact[members].unique(return_counts=True)
                best = int(ids[cnts.argmax()])
                co = set((coact == best).nonzero(as_tuple=True)[0].tolist())
                jac = len(geo & co) / len(geo | co)
                print(
                    f"{family} cluster {b['cluster']}: "
                    f"co-activation best-match Jaccard {jac:.2f}"
                )
    if args.stage in ("all", "figures"):
        report = json.loads((out / "family_battery.json").read_text())
        stage_figures(store, labels, report, out, device)
        print(f"figures -> {out / 'figures'}")


if __name__ == "__main__":
    main()
