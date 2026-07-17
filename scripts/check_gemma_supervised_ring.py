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

  python scripts/check_gemma_supervised_ring.py
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
    from sae_lens import SAE
    from transformers import AutoTokenizer

    from block_crosscoder_experiment.phase0.battery import run_cluster_battery
    from block_crosscoder_experiment.phase0.harvest import CodeStore
    from block_crosscoder_experiment.phase0.labels import (
        build_label_map,
        label_tokens,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = CodeStore(STORE)
    store.load_csc()
    ccol, row, _ = store._csc
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    token_ids = store.token_ids().to(torch.long)
    decoder = (
        SAE.from_pretrained(RELEASE, SAE_ID, device="cpu").W_dec.detach().to(device)
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

    out = STORE / "target_run" / "supervised_ring.json"
    out.write_text(json.dumps(jsonable(results), indent=2) + "\n")
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
