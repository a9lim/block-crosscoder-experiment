"""Phase 0.5 cross-layer coherence pre-test (design §Phase 0.5, zero training).

Do the per-depth month-family subspaces cohere across layers well enough
to justify one shared code across sites? Three metrics per depth pair,
none of them per-latent cosine matching (review finding 20):

(i)   principal angles between per-depth family decoder subspaces,
      against a random-feature-subset null (both sides drawn random,
      rank-matched, from that depth's fired features);
(ii)  paired-token correspondence: CCA canonical correlations between
      member-code vectors, and orthogonal-Procrustes-aligned held-out
      R^2 between subspace coordinates, against shuffled-pairing nulls;
(iii) out-of-sample coordinate prediction: affine layer-A -> layer-B
      code map fit on half the paired tokens, held-out R^2 on the rest
      (both directions), against the same shuffled-pairing null.

Gate 2x2 per pair: spans match AND positions correspond -> the
shared-code premise has legs. Span-match without correspondence ->
frames persist, codes transform (steers toward stage-blocks).
Correspondence without raw-basis span-match is the BSC-native outcome
(per-site frames differ, one code) and is reported as such.

Members per depth are the supervised top-1-per-class features
(verification-only labels, 2026-07-15 ruling) from each store's
target_run/supervised_ring.json; a depth with < 4 distinct members
drops out of subspace tests with a printed reason. Paired tokens are
the family-labeled tokens of the shared deterministic Pile stream
(token identity across stores is asserted, not assumed).

Runs on jobe (stores under /data/stores/bcc-phase0):

  python scripts/check_phase05_cross_layer.py --figures
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

RELEASE = "gemma-scope-2-4b-pt-res"
TOKENIZER = "google/gemma-3-4b-pt"
DEPTHS = (
    (9, "layer_9_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l9_65k_pile")),
    (17, "layer_17_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l17_65k_pile")),
    (22, "layer_22_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l22_65k_pile")),
    (29, "layer_29_width_65k_l0_medium", Path("/data/stores/bcc-phase0/gemma3_4b_l29_65k_pile")),
)
FAMILIES = ("month", "weekday")
N_CLASSES = {"month": 12, "weekday": 7}
MIN_DISTINCT = 4  # subspace tests need >= 4 distinct members (split-geometry gate)
MIN_FIRES = 20  # null-draw feature pool: fired at least this often
N_NULL_SUBSPACE = 1000
N_NULL_PAIRING = 200
SEED = 0


def principal_cosines(qa, qb):
    """Cosines of principal angles between two orthonormal bases (d, r)."""
    import torch

    return torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)


def orthonormal(rows):
    """Orthonormal basis (d, r) for the span of the given rows (r, d)."""
    import torch

    q, _ = torch.linalg.qr(rows.T)
    return q[:, : rows.shape[0]]


def cca_corrs(x, y, eps: float = 1e-6):
    """Canonical correlations between paired (n, rx) and (n, ry) data."""
    import torch

    x = x - x.mean(0)
    y = y - y.mean(0)
    ux, sx, _ = torch.linalg.svd(x, full_matrices=False)
    uy, sy, _ = torch.linalg.svd(y, full_matrices=False)
    kx = int((sx > eps * sx.max()).sum()) if sx.numel() else 0
    ky = int((sy > eps * sy.max()).sum()) if sy.numel() else 0
    if kx == 0 or ky == 0:
        return torch.zeros(0)
    return torch.linalg.svdvals(ux[:, :kx].T @ uy[:, :ky]).clamp(0.0, 1.0)


def heldout_r2_affine(xa_tr, xb_tr, xa_te, xb_te):
    """Held-out R^2 of the affine least-squares map xa -> xb."""
    import torch

    ones_tr = torch.ones(xa_tr.shape[0], 1)
    ones_te = torch.ones(xa_te.shape[0], 1)
    a_tr = torch.cat([xa_tr, ones_tr], dim=1)
    a_te = torch.cat([xa_te, ones_te], dim=1)
    m = torch.linalg.lstsq(a_tr, xb_tr).solution
    resid = xb_te - a_te @ m
    denom = (xb_te - xb_tr.mean(0)).pow(2).sum()
    if float(denom) <= 0:
        return 0.0
    return float(1.0 - resid.pow(2).sum() / denom)


def heldout_r2_procrustes(ca_tr, cb_tr, ca_te, cb_te):
    """Held-out R^2 after orthogonal Procrustes (with scale) on train.

    Rank mismatch is fine: U @ Vt below is a (ra, rb) partial isometry.
    """
    import torch

    mu_a, mu_b = ca_tr.mean(0), cb_tr.mean(0)
    a_tr, b_tr = ca_tr - mu_a, cb_tr - mu_b
    u, s, vt = torch.linalg.svd(a_tr.T @ b_tr, full_matrices=False)
    rot = u @ vt
    denom_tr = a_tr.pow(2).sum()
    scale = float(s.sum() / denom_tr) if float(denom_tr) > 0 else 1.0
    pred = (ca_te - mu_a) @ rot * scale + mu_b
    denom = (cb_te - mu_b).pow(2).sum()
    if float(denom) <= 0:
        return 0.0
    return float(1.0 - (cb_te - pred).pow(2).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/phase05/cross_layer.json")
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="render per-pair angle spectra + R^2 panels into figures/phase05",
    )
    args = parser.parse_args()
    print(f"config: release={RELEASE} depths={[d for d, _, _ in DEPTHS]}", flush=True)

    import torch
    from sae_lens import SAE
    from transformers import AutoTokenizer

    from block_crosscoder_experiment.phase0.harvest import CodeStore
    from block_crosscoder_experiment.phase0.labels import (
        build_label_map,
        label_tokens,
    )

    gen = torch.Generator().manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    # -- per-depth state: decoders, member features, firing pools ----------
    stores: dict[int, CodeStore] = {}
    decoders: dict[int, torch.Tensor] = {}
    pools: dict[int, torch.Tensor] = {}
    members: dict[str, dict[int, list[int]]] = {f: {} for f in FAMILIES}
    for layer, sae_id, root in DEPTHS:
        stores[layer] = CodeStore(root)
        probe = json.loads((root / "target_run" / "supervised_ring.json").read_text())
        for family in FAMILIES:
            per_class = probe[family]["per_class"]
            top1 = {int(c): int(p[0][0]) for c, p in per_class.items() if p}
            members[family][layer] = sorted(set(top1.values()))
        decoders[layer] = (
            SAE.from_pretrained(RELEASE, sae_id, device="cpu", dtype="float32")
            .W_dec.detach()
        )
        pools[layer] = (stores[layer].firing_counts() >= MIN_FIRES).nonzero().squeeze(1)
        print(
            f"layer {layer}: "
            + ", ".join(
                f"{f} {len(members[f][layer])} distinct members" for f in FAMILIES
            )
            + f"; null pool {pools[layer].numel():,} feats",
            flush=True,
        )

    # -- shared token stream (assert, don't assume) -------------------------
    ref_ids = stores[DEPTHS[0][0]].token_ids()
    for layer, _, _ in DEPTHS[1:]:
        if not torch.equal(ref_ids, stores[layer].token_ids()):
            raise AssertionError(f"token stream mismatch at layer {layer}")
    print(f"token stream identical across depths ({ref_ids.numel():,} tokens)")

    report: dict = {"_config": {"release": RELEASE, "depths": {}}}
    for layer, sae_id, root in DEPTHS:
        report["_config"]["depths"][str(layer)] = {"sae_id": sae_id, "store": str(root)}

    for family in FAMILIES:
        class_ids = label_tokens(ref_ids.to(torch.long), build_label_map(tok, family))
        labeled = (class_ids >= 0).nonzero().squeeze(1)
        fam_report: dict = {
            "n_labeled_tokens": int(labeled.numel()),
            "members": {str(l): members[family][l] for l in stores},
            "pairs": {},
        }
        report[family] = fam_report
        print(f"\n=== {family}: {labeled.numel():,} labeled tokens ===", flush=True)

        eligible = [l for l, _, _ in DEPTHS if len(members[family][l]) >= MIN_DISTINCT]
        for l, _, _ in DEPTHS:
            if l not in eligible:
                print(
                    f"  layer {l}: {len(members[family][l])} distinct members "
                    f"< {MIN_DISTINCT} — dropped from subspace tests"
                )
        if len(eligible) < 2:
            print(f"  fewer than 2 eligible depths — no pairs to test")
            fam_report["verdict"] = "insufficient_depths"
            continue

        # member codes on labeled tokens, per eligible depth
        codes: dict[int, torch.Tensor] = {}
        bases: dict[int, torch.Tensor] = {}
        coord_maps: dict[int, torch.Tensor] = {}
        for l in eligible:
            m = torch.tensor(members[family][l], dtype=torch.long)
            z = stores[l].select_members(m)[labeled]
            codes[l] = z
            rows = decoders[l][m]
            bases[l] = orthonormal(rows)
            coord_maps[l] = rows @ bases[l]  # (r, r): code -> subspace coords
            active = float((z.abs().sum(1) > 0).float().mean())
            print(
                f"  layer {l}: member codes on labeled tokens, "
                f"active fraction {active:.2f}"
            )

        # split halves once per family (same split for all pairs and nulls)
        perm = torch.randperm(labeled.numel(), generator=gen)
        half = labeled.numel() // 2
        tr_idx, te_idx = perm[:half], perm[half:]

        for la, lb in itertools.combinations(eligible, 2):
            key = f"{la}->{lb}"
            za, zb = codes[la], codes[lb]
            ra, rb = za.shape[1], zb.shape[1]

            # (i) principal angles vs rank-matched random-feature null
            cos = principal_cosines(bases[la], bases[lb])
            stat = float(cos.pow(2).mean())
            null_stats = torch.empty(N_NULL_SUBSPACE)
            for i in range(N_NULL_SUBSPACE):
                fa = pools[la][torch.randperm(pools[la].numel(), generator=gen)[:ra]]
                fb = pools[lb][torch.randperm(pools[lb].numel(), generator=gen)[:rb]]
                null_stats[i] = principal_cosines(
                    orthonormal(decoders[la][fa]), orthonormal(decoders[lb][fb])
                ).pow(2).mean()
            span_p = float(((null_stats >= stat).sum() + 1) / (N_NULL_SUBSPACE + 1))

            # (ii) CCA + Procrustes with shuffled-pairing nulls
            cca = cca_corrs(za, zb)
            cca_mean = float(cca.mean()) if cca.numel() else 0.0
            ca, cb = za @ coord_maps[la], zb @ coord_maps[lb]
            proc_r2 = heldout_r2_procrustes(
                ca[tr_idx], cb[tr_idx], ca[te_idx], cb[te_idx]
            )

            # (iii) out-of-sample code map, both directions
            map_ab = heldout_r2_affine(za[tr_idx], zb[tr_idx], za[te_idx], zb[te_idx])
            map_ba = heldout_r2_affine(zb[tr_idx], za[tr_idx], zb[te_idx], za[te_idx])

            null_cca = torch.empty(N_NULL_PAIRING)
            null_map = torch.empty(N_NULL_PAIRING)
            for i in range(N_NULL_PAIRING):
                shuf = torch.randperm(zb.shape[0], generator=gen)
                zbs = zb[shuf]
                c = cca_corrs(za, zbs)
                null_cca[i] = float(c.mean()) if c.numel() else 0.0
                null_map[i] = heldout_r2_affine(
                    za[tr_idx], zbs[tr_idx], za[te_idx], zbs[te_idx]
                )
            cca_p = float(((null_cca >= cca_mean).sum() + 1) / (N_NULL_PAIRING + 1))
            map_p = float(((null_map >= map_ab).sum() + 1) / (N_NULL_PAIRING + 1))

            span_match = span_p < 0.05
            correspond = map_p < 0.05 and map_ab > float(null_map.max())
            verdict = {
                (True, True): "spans match AND positions correspond",
                (True, False): "frames persist, codes transform",
                (False, True): "correspondence without raw-basis span match (BSC-native)",
                (False, False): "no cross-layer coherence",
            }[(span_match, correspond)]

            fam_report["pairs"][key] = {
                "principal_cos": [round(float(c), 4) for c in cos],
                "mean_cos2": round(stat, 4),
                "span_null_mean": round(float(null_stats.mean()), 4),
                "span_p": span_p,
                "cca_corrs": [round(float(c), 4) for c in cca],
                "cca_mean": round(cca_mean, 4),
                "cca_p": cca_p,
                "procrustes_heldout_r2": round(proc_r2, 4),
                "codemap_heldout_r2_ab": round(map_ab, 4),
                "codemap_heldout_r2_ba": round(map_ba, 4),
                "codemap_null_max": round(float(null_map.max()), 4),
                "codemap_p": map_p,
                "verdict": verdict,
            }
            print(
                f"  {key}: mean cos^2 {stat:.3f} (null {float(null_stats.mean()):.3f}, "
                f"p {span_p:.4f}) | CCA mean {cca_mean:.3f} (p {cca_p:.4f}) | "
                f"Procrustes R2 {proc_r2:.3f} | code map R2 {map_ab:.3f}/{map_ba:.3f} "
                f"(null max {float(null_map.max()):.3f}, p {map_p:.4f})\n"
                f"    -> {verdict}",
                flush=True,
            )

        if args.figures and fam_report["pairs"]:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pairs = fam_report["pairs"]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            for key, res in pairs.items():
                ax1.plot(
                    range(1, len(res["principal_cos"]) + 1),
                    res["principal_cos"],
                    "o-",
                    label=f"{key} (p {res['span_p']:.3f})",
                )
            ax1.set_xlabel("principal angle index")
            ax1.set_ylabel("cos θ")
            ax1.set_title(f"{family}: subspace principal angles across depths")
            ax1.set_ylim(0, 1.05)
            ax1.legend(fontsize=8)
            keys = list(pairs)
            x = range(len(keys))
            ax2.bar(x, [pairs[k]["codemap_heldout_r2_ab"] for k in keys], 0.35,
                    label="code map A→B (held-out R²)")
            ax2.scatter(x, [pairs[k]["codemap_null_max"] for k in keys],
                        color="k", marker="_", s=200, label="shuffled-pairing null max")
            ax2.set_xticks(list(x))
            ax2.set_xticklabels(keys, rotation=30, fontsize=8)
            ax2.set_title(f"{family}: out-of-sample coordinate prediction")
            ax2.legend(fontsize=8)
            fig.tight_layout()
            fig_dir = Path("figures/phase05")
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_dir / f"{family}_cross_layer.png", dpi=150)
            plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
