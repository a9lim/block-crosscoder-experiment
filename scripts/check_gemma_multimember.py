"""Robustness check for the gemma singleton-candidate finding (one-off).

Every top family-affinity cluster on both branches came out a singleton
(one feature firing across all classes — a "day-of-week"/"month-of-year"
feature, not a ring of features). Two follow-ups the finding needs:

1. Was a MULTI-member family-affine cluster passed over by the top-3
   selection? Rerun the coverage-first affinity rule restricted to
   ≥2-member clusters and battery the top hit per family/branch.
2. What ARE the singleton candidates? Top fired tokens per feature, so
   the findings doc can name them.

  python scripts/check_gemma_multimember.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

STORE = Path("/data/stores/bcc-phase0/gemma3_4b_l22_pile")
RELEASE = "gemma-scope-2-4b-pt-res"
SAE_ID = "layer_22_width_16k_l0_medium"
TOKENIZER = "google/gemma-3-4b-pt"
FAMILIES = ("weekday", "month")
N_CLASSES = {"weekday": 7, "month": 12}


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
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    token_ids = store.token_ids().to(torch.long)
    out = STORE / "target_run"
    labelings = {
        "spectral": torch.load(out / "cluster_labels.pt", weights_only=True),
        "graph": torch.load(out / "graph_labels.pt", weights_only=True),
    }
    decoder = SAE.from_pretrained(RELEASE, SAE_ID, device="cpu").W_dec.detach().to(device)

    # -- 2. name the singleton candidates ---------------------------------
    report = json.loads((out / "family_battery.json").read_text())
    print("=== singleton candidate identities (top fired tokens) ===", flush=True)
    for family in FAMILIES:
        for branch, entry in report[family].items():
            for b in entry["batteries"][:1]:
                members = torch.tensor(b["members"])
                rows = store.member_row_union(members).to(torch.long)
                fired = Counter(token_ids[rows].tolist())
                names = [
                    f"{tok.decode([t])!r}×{c}" for t, c in fired.most_common(15)
                ]
                print(
                    f"{family}/{branch} cluster {b['cluster']} "
                    f"(features {members.tolist()}): {', '.join(names)}",
                    flush=True,
                )

    # -- 1. multi-member affinity rerun ------------------------------------
    results: dict = {}
    print("\n=== coverage-first affinity, >=2-member clusters only ===", flush=True)
    for family in FAMILIES:
        class_ids = label_tokens(token_ids, build_label_map(tok, family)).cpu()
        results[family] = {}
        for branch, labels in labelings.items():
            scores = []
            uniq, cnt = labels.unique(return_counts=True)
            for cid, c in zip(uniq.tolist(), cnt.tolist()):
                if cid < 0 or c < 2:
                    continue
                members = (labels == cid).nonzero(as_tuple=True)[0]
                rows = store.member_row_union(members).cpu().to(torch.long)
                if rows.shape[0] < 50:
                    continue
                fired = class_ids[rows]
                fired = fired[fired >= 0]
                coverage = int(fired.unique().numel())
                affinity = float(fired.shape[0]) / rows.shape[0]
                scores.append((cid, affinity, int(rows.shape[0]), coverage, c))
            scores.sort(key=lambda t: (-(t[3] >= N_CLASSES[family] - 1), -t[1]))
            top = scores[:5]
            print(f"{family}/{branch} top-5 multi-member:", flush=True)
            for cid, aff, n, cov, size in top:
                print(
                    f"  cluster {cid}: size {size} affinity {aff:.4f} "
                    f"fired {n} coverage {cov}",
                    flush=True,
                )
            batteries = []
            for cid, aff, _, _, size in top[:1]:
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
                circ = battery.get("circular")
                print(
                    f"  battery cluster {cid} (size {size}): "
                    + (
                        f"circ {circ:.3f} p {battery['circular_p']:.4f} "
                        f"plane {battery['circular_plane']}"
                        if circ is not None
                        else f"verdict {battery.get('verdict', 'no labeled tokens')}"
                    ),
                    flush=True,
                )
                batteries.append(battery)
            results[family][branch] = {
                "top_multimember": [
                    {"cluster": c, "size": s, "affinity": round(a, 4),
                     "n_fired": n, "coverage": cov}
                    for c, a, n, cov, s in top
                ],
                "batteries": batteries,
            }

    def jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonable(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj

    (out / "multimember_check.json").write_text(
        json.dumps(jsonable(results), indent=2) + "\n"
    )
    print(f"\n-> {out / 'multimember_check.json'}", flush=True)


if __name__ == "__main__":
    main()
