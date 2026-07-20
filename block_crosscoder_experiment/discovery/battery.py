"""Per-cluster ring battery: Engels' protocol + our additions, assembled.

Pure functions of (codes, decoder, cluster members) — no SAE object, no
model forward. The harvest step produces codes once; everything here reruns
cheaply and deterministically, so nulls and BH sweeps are tractable.

Evidence policy (findings §2.3): all scores are computed gate-conditionally
— only on tokens where the cluster actually fires (Engels' discard rule) —
and no single score is a ring verdict. The labeled tests (circular
decoding + n-gon) carry the positive control; the label-free harmonic scan
carries the unknown-cluster search, calibrated by random-cluster nulls
under BH.
"""

from __future__ import annotations

import torch

from block_crosscoder_experiment.discovery.nulls import (
    class_permutation_pvalue,
    empirical_pvalue,
    random_member_sets,
)
from block_crosscoder_experiment.discovery.rings import (
    angle_harmonic_power,
    circular_decoding,
    cone_normalize,
    ngon_alignment,
    pca_projections,
    plane_scan,
)

__all__ = [
    "cluster_restricted_reconstruction",
    "run_cluster_battery",
    "unknown_cluster_scan",
]


_DENSE_BYTES_CAP = 4 * 1024**3


def _subsample(idx: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor:
    if idx.shape[0] <= max_tokens:
        return idx
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(idx.shape[0], generator=gen)[:max_tokens]
    return idx[pick.to(idx.device)]


def _gated_member_codes(
    codes,
    members: torch.Tensor,
    *,
    min_active: int,
    max_tokens: int | None,
    seed: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(member_codes (T', |C|), kept_idx (T',)) gated on ≥min_active firing
    members, subsampled to max_tokens BEFORE densifying.

    On a CodeStore the gate runs on sparse counts, so peak memory is
    max_tokens × |members|, never n_tokens × |members| — densify-first cost
    35 GB per call on the production 2192-member blob, and the scan's null
    draws repeat the allocation per draw (the 2026-07-16 scan OOM).
    Oversized results still fall back to CPU RAM; downstream matmuls follow
    the codes' device.
    """
    if hasattr(codes, "member_firing_counts"):
        counts = codes.member_firing_counts(members)
        kept_idx = (counts >= min_active).nonzero(as_tuple=True)[0]
        if max_tokens is not None:
            kept_idx = _subsample(kept_idx, max_tokens, seed)
        if kept_idx.shape[0] * int(members.shape[0]) * 4 > _DENSE_BYTES_CAP:
            device = "cpu"
        return codes.select_member_rows(members, kept_idx, device=device), kept_idx
    dense = codes[:, members]
    kept_idx = (dense.gt(0).sum(dim=1) >= min_active).nonzero(as_tuple=True)[0]
    if max_tokens is not None:
        kept_idx = _subsample(kept_idx, max_tokens, seed)
    return dense[kept_idx], kept_idx


def cluster_restricted_reconstruction(
    codes,
    decoder: torch.Tensor,
    members: torch.Tensor,
    *,
    max_tokens: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Engels step 2: reconstruct with only the cluster's latents.

    Returns (recon (T', d), kept_idx (T',)): tokens where no member latent
    fires are discarded, survivors are decoded through member rows only.
    `codes` is a dense (T, F) tensor or a harvest.CodeStore; `max_tokens`
    caps T' by seeded subsampling (production stores are millions of
    tokens; the indices don't need them all).
    """
    member_codes, kept_idx = _gated_member_codes(
        codes, members, min_active=1, max_tokens=max_tokens, seed=seed,
        device=decoder.device,
    )
    dec = decoder[members.to(decoder.device)].to(
        device=member_codes.device, dtype=member_codes.dtype
    )
    recon = member_codes @ dec
    return recon, kept_idx


def run_cluster_battery(
    codes,
    decoder: torch.Tensor,
    members: torch.Tensor,
    *,
    class_ids: torch.Tensor | None = None,
    n_classes: int | None = None,
    mixture_steps: int = 10_000,
    n_perm: int = 200,
    min_tokens: int = 200,
    max_tokens: int = 100_000,
    seed: int = 0,
) -> dict:
    """Full battery for one candidate cluster.

    class_ids (per token, −1 = unlabeled) unlocks the labeled tests; without
    them the battery reports geometry + label-free harmonics only.
    """
    recon, kept = cluster_restricted_reconstruction(
        codes, decoder, members, max_tokens=max_tokens, seed=seed
    )
    out: dict = {"n_tokens": int(kept.shape[0]), "n_members": int(members.shape[0])}
    if out["n_tokens"] < min_tokens:
        out["verdict"] = "insufficient_tokens"
        return out

    proj, explained = pca_projections(recon, k=5)
    out["plane_scan"] = plane_scan(
        proj, explained=explained, mixture_steps=mixture_steps, seed=seed
    )
    best = out["plane_scan"]["best_plane"]
    best_points = proj[:, list(best)]
    out["harmonics"] = angle_harmonic_power(best_points)

    # PC1-as-intensity cone check: rerun the top plane scores on the
    # cone-normalized remainder (ring may live in PCs 2–3 of a cone).
    if proj.shape[1] >= 3:
        cone = cone_normalize(proj)
        out["cone_scan"] = plane_scan(
            cone, planes=((0, 1),), mixture_steps=mixture_steps, seed=seed
        )

    if class_ids is not None and n_classes is not None:
        ids = class_ids[kept.to(class_ids.device)].to(recon.device)
        labeled = ids >= 0
        out["n_labeled"] = int(labeled.sum())
        if out["n_labeled"] >= min_tokens:
            ids_l = ids[labeled]
            # The ring can live on any scanned plane (Engels chose planes by
            # eye; the irreducibility-best plane is often the cone, not the
            # ring — weekday cluster 937 taught us this in vivo). Statistic:
            # max over planes of held-out circular decoding, with the SAME
            # max-over-planes statistic under the class-identity permutation
            # null, so plane selection cannot inflate significance.
            plane_pts = {
                p: proj[:, list(p)][labeled]
                for p in out["plane_scan"]["planes"]
            }

            def circ_stat(lab: torch.Tensor) -> float:
                return max(
                    circular_decoding(pts, lab, n_classes, seed=seed)
                    for pts in plane_pts.values()
                )

            out["circular"], out["circular_p"] = class_permutation_pvalue(
                circ_stat, ids_l, n_classes, n_perm=n_perm, seed=seed
            )
            by_plane = {
                p: circular_decoding(pts, ids_l, n_classes, seed=seed)
                for p, pts in plane_pts.items()
            }
            circ_plane = max(by_plane, key=lambda p: by_plane[p])
            out["circular_plane"] = circ_plane
            out["circular_by_plane"] = by_plane
            ring_pts = plane_pts[circ_plane]
            out["ngon"] = ngon_alignment(ring_pts, ids_l, n_classes)
            out["harmonics_circ_plane"] = angle_harmonic_power(ring_pts)
    return out


def unknown_cluster_scan(
    codes,
    decoder: torch.Tensor,
    clusters: dict[int, torch.Tensor],
    *,
    harmonics: range = range(3, 13),
    n_null_draws: int = 100,
    mixture_steps: int = 1000,
    min_tokens: int = 200,
    max_tokens: int = 100_000,
    firing_counts: torch.Tensor | None = None,
    seed: int = 0,
    progress=None,
) -> dict[int, dict]:
    """Label-free ring SURFACING over many clusters, null-calibrated.

    A calibrated candidate finder, not a ring verdict — verdicts come from
    the labeled battery on whatever this flags. Two design points, both
    taught by planted controls:

    - Gate on CO-FIRING (≥2 active members per token). Multi-d features
      require co-activation to reconstruct (the Engels/SASA premise);
      one-hot ray clusters never co-fire and mustn't reach the statistic —
      their reconstructions score high on irreducibility (rays are genuine
      multi-d structure) while being everything but rings.
    - Statistic: harmonic CONTRAST (peak − median power over m) on the best
      variance-filtered PC plane. An equispaced n-ring puts power at m = n
      and nothing elsewhere; arbitrary ray/clump layouts smear power across
      every m, so raw peak height cannot separate them but contrast does.

    The Engels irreducibility score is reported as a diagnostic alongside.
    p-values come from matched-size random-member clusters (candidate's own
    members excluded) through the SAME gated scorer; clusters or draws that
    never co-fire get p = 1 with a verdict note. BH over the returned
    p-values is the caller's final step (the caller knows the search width).
    """

    def gated_stats(
        members: torch.Tensor, stat_seed: int
    ) -> tuple[float, float, dict[int, float]] | None:
        member_codes, cofire_idx = _gated_member_codes(
            codes, members, min_active=2, max_tokens=max_tokens,
            seed=stat_seed, device=decoder.device,
        )
        if cofire_idx.shape[0] < min_tokens:
            return None
        dec = decoder[members.to(decoder.device)].to(
            device=member_codes.device, dtype=member_codes.dtype
        )
        recon = member_codes @ dec
        proj, explained = pca_projections(recon, k=5)
        scan = plane_scan(
            proj, explained=explained, mixture_steps=mixture_steps, seed=stat_seed
        )
        pts = proj[:, list(scan["best_plane"])]
        power = angle_harmonic_power(pts, harmonics=harmonics)
        vals = sorted(power.values())
        contrast = vals[-1] - vals[len(vals) // 2]
        return contrast, scan["mean"]["score"], power

    results: dict[int, dict] = {}
    null_cache: dict[int, list[float]] = {}
    for cid, members in clusters.items():
        observed = gated_stats(members, seed)
        if observed is None:
            results[cid] = {"verdict": "insufficient_cofire_tokens", "p": 1.0}
            if progress:
                progress(f"[scan] {cid} (size {int(members.shape[0])}): gated out")
            continue
        contrast, irr, power = observed
        size = int(members.shape[0])
        if size not in null_cache:
            draws = random_member_sets(
                decoder.shape[0],
                size,
                n_draws=n_null_draws,
                seed=seed,
                exclude=members.cpu(),
                frequencies=firing_counts,
                match_to=members.cpu() if firing_counts is not None else None,
            )
            null_cache[size] = [
                s[0]
                for i, d in enumerate(draws)
                if (s := gated_stats(d.to(members.device), seed + 1 + i))
                is not None
            ]
            if progress:
                progress(
                    f"[scan] null cache size {size}: "
                    f"{len(null_cache[size])}/{n_null_draws} draws co-fired"
                )
        null = null_cache[size]
        results[cid] = {
            "contrast": contrast,
            "irreducibility": irr,
            "p": empirical_pvalue(contrast, null) if null else 1.0,
            "n_null": len(null),
            "harmonics": power,
        }
        if progress:
            progress(
                f"[scan] {cid} (size {size}): contrast {contrast:.3f} "
                f"p {results[cid]['p']:.4f} (n_null {len(null)})"
            )
    return results
