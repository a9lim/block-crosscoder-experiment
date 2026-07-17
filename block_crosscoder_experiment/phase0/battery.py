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

from block_crosscoder_experiment.phase0.nulls import (
    class_permutation_pvalue,
    empirical_pvalue,
    random_member_sets,
)
from block_crosscoder_experiment.phase0.rings import (
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


def _member_codes(codes, members: torch.Tensor, device) -> torch.Tensor:
    """Dense (T, |members|) submatrix from a tensor or a CodeStore.

    Oversized selections (huge clusters × production stores) fall back to
    CPU RAM; downstream matmuls follow the codes' device.
    """
    if hasattr(codes, "select_members"):
        est = codes.n_tokens * int(members.shape[0]) * 4
        if est > _DENSE_BYTES_CAP:
            device = "cpu"
        return codes.select_members(members, device=device)
    return codes[:, members]


def _subsample(idx: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor:
    if idx.shape[0] <= max_tokens:
        return idx
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(idx.shape[0], generator=gen)[:max_tokens]
    return idx[pick.to(idx.device)]


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
    member_codes = _member_codes(codes, members, decoder.device)  # (T, |C|)
    kept_idx = member_codes.gt(0).any(dim=1).nonzero(as_tuple=True)[0]
    if max_tokens is not None:
        kept_idx = _subsample(kept_idx, max_tokens, seed)
    dec = decoder[members.to(decoder.device)].to(
        device=member_codes.device, dtype=member_codes.dtype
    )
    recon = member_codes[kept_idx] @ dec
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
            pts = best_points[labeled]
            ids_l = ids[labeled]
            # Class-identity permutation: the null must keep class clumps
            # intact and randomize only their cyclic placement.
            out["circular"], out["circular_p"] = class_permutation_pvalue(
                lambda lab: circular_decoding(pts, lab, n_classes, seed=seed),
                ids_l,
                n_classes,
                n_perm=n_perm,
                seed=seed,
            )
            out["ngon"] = ngon_alignment(pts, ids_l, n_classes)
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
        member_codes = _member_codes(codes, members, decoder.device)
        cofire_idx = (member_codes.gt(0).sum(dim=1) >= 2).nonzero(as_tuple=True)[0]
        if cofire_idx.shape[0] < min_tokens:
            return None
        cofire_idx = _subsample(cofire_idx, max_tokens, stat_seed)
        dec = decoder[members.to(decoder.device)].to(
            device=member_codes.device, dtype=member_codes.dtype
        )
        recon = member_codes[cofire_idx] @ dec
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
        null = null_cache[size]
        results[cid] = {
            "contrast": contrast,
            "irreducibility": irr,
            "p": empirical_pvalue(contrast, null) if null else 1.0,
            "n_null": len(null),
            "harmonics": power,
        }
    return results
