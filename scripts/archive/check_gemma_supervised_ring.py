"""Supervised ring probe: does ring geometry exist even where discovery can't see it?

The gemma-16k discovery result is singleton family features (one
"day-of-week" / "month-of-year" feature firing across all classes) and no
multi-member family-affine cluster on either branch. That leaves one
loophole: per-CLASS features (a " Monday" feature, a " June" feature, ...)
could exist but sit too orthogonal to cosine-cluster, so decoder-geometry
discovery would miss a ring that is nonetheless there.

Close it with labels: per class, rank features by class-selective firing
(fires-on-class / fires-total, minimum support), take the top per class,
battery the UNION through the exact production pipeline. If even the
supervised union has no ring plane, the null covers representation, not
just discovery. Verification-only use of labels — no discovery role
(2026-07-15 corpus decision).

Defaults reproduce the 16k probe; the width_65k reroute passes --store
and --sae-id. The per-class selectivity table this prints is the direct
readout of the splitting prediction (findings §Interpretation).

  python scripts/check_gemma_supervised_ring.py
  python scripts/check_gemma_supervised_ring.py \
      --store /data/stores/bcc-phase0/gemma3_4b_l22_65k_pile \
      --sae-id layer_22_width_65k_l0_medium
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

STORE = Path("/data/stores/bcc-phase0/gemma3_4b_l22_pile")
RELEASE = "gemma-scope-2-4b-pt-res"
SAE_ID = "layer_22_width_16k_l0_medium"
TOKENIZER = "google/gemma-3-4b-pt"
FAMILIES = ("weekday", "month")
N_CLASSES = {"weekday": 7, "month": 12}
MIN_CLASS_FIRES = 20
TOP_PER_CLASS = 2


def main() -> None:
    import argparse

    from sae_lens import SAE
    from transformers import AutoTokenizer

    from block_crosscoder_experiment.phase0.battery import run_cluster_battery
    from block_crosscoder_experiment.phase0.harvest import CodeStore
    from block_crosscoder_experiment.phase0.labels import (
        build_label_map,
        label_tokens,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument("--sae-id", default=SAE_ID)
    parser.add_argument(
        "--figures", action="store_true",
        help="render the union scatter per family into <store>/target_run/figures",
    )
    args = parser.parse_args()
    print(f"config: release={RELEASE} sae_id={args.sae_id} store={args.store}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = CodeStore(args.store)
    store.load_csc()
    ccol, row, _ = store._csc
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    token_ids = store.token_ids().to(torch.long)
    decoder = (
        SAE.from_pretrained(RELEASE, args.sae_id, device="cpu").W_dec.detach().to(device)
    )

    # feature id per nnz entry (feature-major CSC order)
    feat_ids = torch.repeat_interleave(
        torch.arange(store.n_features, dtype=torch.int64), ccol.diff()
    )
    total_fires = ccol.diff().clamp_min(1)

    results: dict = {}
    for family in FAMILIES:
        class_ids = label_tokens(token_ids, build_label_map(tok, family)).cpu()
        nnz_class = class_ids[row.to(torch.int64)]
        members: list[int] = []
        per_class: dict[int, list] = {}
        for c in range(N_CLASSES[family]):
            fires_on_c = torch.bincount(
                feat_ids[nnz_class == c], minlength=store.n_features
            )
            selectivity = fires_on_c.float() / total_fires.float()
            selectivity[fires_on_c < MIN_CLASS_FIRES] = 0.0
            top = torch.topk(selectivity, TOP_PER_CLASS)
            picks = [
                (int(f), float(s), int(fires_on_c[f]))
                for f, s in zip(top.indices.tolist(), top.values.tolist())
                if s > 0
            ]
            per_class[c] = picks
            members.extend(f for f, _, _ in picks)
        members_t = torch.tensor(sorted(set(members)), dtype=torch.long)
        print(f"=== {family}: supervised union of {members_t.numel()} features ===")
        for c, picks in per_class.items():
            desc = ", ".join(
                f"{f} (sel {s:.2f}, n {n})" for f, s, n in picks
            ) or "none above support floor"
            print(f"  class {c}: {desc}", flush=True)
        if members_t.numel() < 2:
            print("  fewer than 2 class-selective features — no union to test")
            results[family] = {"per_class": per_class, "verdict": "no_features"}
            continue
        battery = run_cluster_battery(
            store,
            decoder,
            members_t,
            class_ids=class_ids,
            n_classes=N_CLASSES[family],
            n_perm=200,
            seed=0,
        )
        battery["members"] = members_t.tolist()
        if args.figures and battery.get("circular") is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from block_crosscoder_experiment.phase0.battery import (
                cluster_restricted_reconstruction,
            )
            from block_crosscoder_experiment.phase0.rings import pca_projections

            recon, kept = cluster_restricted_reconstruction(
                store, decoder, members_t
            )
            proj, _ = pca_projections(recon, k=5)
            best = tuple(battery["circular_plane"])
            ids = class_ids[kept.cpu()]
            labeled = ids >= 0
            pts = proj[:, list(best)].cpu()[labeled]
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], c=ids[labeled], cmap="hsv", s=8, alpha=0.7,
                vmin=0, vmax=N_CLASSES[family],
            )
            ax.set_title(
                f"{args.sae_id} {family} supervised union "
                f"({members_t.numel()} feats) — PCs {best[0]+1}/{best[1]+1}, "
                f"circ {battery['circular']:.2f} (p={battery['circular_p']:.3f})"
            )
            fig.colorbar(sc, ax=ax, label="class")
            fig_dir = args.store / "target_run" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_dir / f"{family}_supervised_union.png", dpi=150)
            plt.close(fig)
        circ = battery.get("circular")
        print(
            f"  battery: "
            + (
                f"circ {circ:.3f} p {battery['circular_p']:.4f} "
                f"plane {battery['circular_plane']} "
                f"by_plane { {str(k): round(v, 3) for k, v in battery['circular_by_plane'].items()} }"
                if circ is not None
                else f"verdict {battery.get('verdict', 'insufficient labeled tokens')}"
            ),
            flush=True,
        )
        results[family] = {"per_class": per_class, "battery": battery}

    def jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonable(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    out = args.store / "target_run" / "supervised_ring.json"
    out.write_text(json.dumps(jsonable(results), indent=2) + "\n")
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
