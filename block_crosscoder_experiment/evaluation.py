"""Method-aware reconstruction and shared-code endpoints.

Every caller supplies a checkpoint, data stream, and output path through a
resolved cell.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from .model import BSCConfig, BlockCrosscoder

__all__ = [
    "centered_fvu",
    "checkpoint_sha256",
    "evaluate_shared_code",
    "load_trained_model",
]

_DECODER_GRAM_BLOCK_CHUNK = 256


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trained_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[BlockCrosscoder, dict]:
    """Load the complete saved model configuration without optimizer setup."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg_payload = dict(payload["model_cfg"])
    if cfg_payload.get("site_dims") is not None:
        cfg_payload["site_dims"] = tuple(cfg_payload["site_dims"])
    model = BlockCrosscoder(BSCConfig(**cfg_payload), device=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, {
        "model_cfg": asdict(model.cfg),
        "train_cfg": payload.get("train_cfg"),
        "run_binding": payload.get("run_binding"),
        "step_idx": payload.get("step_idx"),
        "accepted_tokens": payload.get("accepted_tokens"),
        "data_cursor": payload.get("data_cursor"),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
    }


def centered_fvu(
    target: torch.Tensor,
    prediction: torch.Tensor,
    coordinate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-site centered FVU with padded coordinates excluded."""
    if coordinate_mask is None:
        coordinate_mask = torch.ones(
            target.shape[1:], dtype=torch.bool, device=target.device
        )
    mask = coordinate_mask.to(target.device).unsqueeze(0)
    # Center each activation coordinate over tokens.  A single scalar mean per
    # site would count stable coordinate offsets as variance and can make FVU
    # look arbitrarily good on residual-stream activations.
    mean = (target * mask).sum(dim=0) / target.shape[0]
    centered = (target - mean.unsqueeze(0)) * mask
    error = (target - prediction) * mask
    denominator = centered.double().pow(2).sum(dim=(0, 2))
    numerator = error.double().pow(2).sum(dim=(0, 2))
    return numerator / denominator.clamp_min(1e-30)


@torch.no_grad()
def evaluate_shared_code(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
    selection_mode: str = "threshold",
) -> dict:
    """Evaluate endpoints valid for every implemented encoder topology.

    Site-only and leave-one-site-out views are operational re-encodings through
    the model, so biases, nonlinear scores, source-only fusion, and
    dense scaffolds retain their actual semantics. No direct ``model.E`` access
    is used.
    """
    if selection_mode not in {"topk", "threshold"}:
        raise ValueError("selection_mode must be 'topk' or 'threshold'")
    model = model.to(device).eval()
    cfg = model.cfg
    S, G, b = cfg.n_sites, cfg.n_blocks, cfg.block_dim
    coord = model.coordinate_mask[:, 0, 0].to(device)
    has_padded_coordinates = model._has_padded_coordinates

    target_sum = torch.zeros(S, cfg.d_model, dtype=torch.float64, device=device)
    target_sumsq = torch.zeros_like(target_sum)
    full_sse = torch.zeros(S, dtype=torch.float64, device=device)
    site_sse = torch.zeros(S, S, dtype=torch.float64, device=device)
    loo_sse = torch.zeros(S, S, dtype=torch.float64, device=device)
    support_intersection = torch.zeros(S, dtype=torch.float64, device=device)
    support_union = torch.zeros(S, dtype=torch.float64, device=device)
    loo_intersection = torch.zeros(S, dtype=torch.float64, device=device)
    loo_union = torch.zeros(S, dtype=torch.float64, device=device)
    site_intersection_count = torch.zeros(S, G, dtype=torch.float64, device=device)
    site_full_count = torch.zeros_like(site_intersection_count)
    site_concordance_numerator = torch.zeros(S, G, dtype=torch.float64, device=device)
    site_concordance_denominator = torch.zeros_like(site_concordance_numerator)
    site_full_code_sum = torch.zeros(S, G, b, dtype=torch.float64, device=device)
    site_partial_code_sum = torch.zeros_like(site_full_code_sum)
    site_intersection_full_energy = torch.zeros_like(site_concordance_numerator)
    site_full_energy = torch.zeros_like(site_concordance_numerator)
    loo_intersection_count = torch.zeros(S, G, dtype=torch.float64, device=device)
    loo_full_count = torch.zeros_like(loo_intersection_count)
    loo_concordance_numerator = torch.zeros(S, G, dtype=torch.float64, device=device)
    loo_concordance_denominator = torch.zeros_like(loo_concordance_numerator)
    loo_full_code_sum = torch.zeros(S, G, b, dtype=torch.float64, device=device)
    loo_partial_code_sum = torch.zeros_like(loo_full_code_sum)
    loo_intersection_full_energy = torch.zeros_like(loo_concordance_numerator)
    loo_full_energy = torch.zeros_like(loo_concordance_numerator)
    pre_selection_loo_delta_sq = torch.zeros(S, G, dtype=torch.float64, device=device)
    post_selection_loo_delta_sq = torch.zeros_like(pre_selection_loo_delta_sq)
    fire = torch.zeros(G, dtype=torch.float64, device=device)
    zsum = torch.zeros(G, b, dtype=torch.float64, device=device)
    zz = torch.zeros(G, b, b, dtype=torch.float64, device=device)
    materialized_decoder = model.decoder_tensor()
    materialized_encoder = (
        materialized_decoder * model.log_gamma.exp()
        if cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    # A publication-scale decoder can be several GiB in fp32.  Keeping a
    # complete fp64 copy alive for evaluation can therefore exceed the 24 GiB
    # device even though the Gram products themselves are tiny.  Materialize
    # bounded fp64 block slices, retain only their O(S*G*b^2) Grams, and release
    # each wide slice before the next one.
    decoder_gram_chunks: list[torch.Tensor] = []
    all_site_decoder_gram_chunks: list[torch.Tensor] = []
    for start in range(0, G, _DECODER_GRAM_BLOCK_CHUNK):
        decoder_chunk = materialized_decoder[
            :, start : start + _DECODER_GRAM_BLOCK_CHUNK
        ].double()
        site_gram_chunk = torch.einsum(
            "sgbd,sgcd->sgbc", decoder_chunk, decoder_chunk
        )
        decoder_gram_chunks.append(site_gram_chunk)
        all_site_decoder_gram_chunks.append(site_gram_chunk.sum(dim=0))
        del decoder_chunk
    decoder_gram = torch.cat(decoder_gram_chunks, dim=1)
    all_site_decoder_gram = torch.cat(all_site_decoder_gram_chunks, dim=0)
    del decoder_gram_chunks, all_site_decoder_gram_chunks, site_gram_chunk
    n_tokens = 0

    def accumulate_coordinate_concordance(
        partial,
        full,
        index: int,
        *,
        intersection_count: torch.Tensor,
        full_count: torch.Tensor,
        concordance_numerator: torch.Tensor,
        concordance_denominator: torch.Tensor,
        full_code_sum: torch.Tensor,
        partial_code_sum: torch.Tensor,
        intersection_full_energy: torch.Tensor,
        full_energy: torch.Tensor,
    ) -> None:
        """Accumulate gauge-invariant agreement on matched support events."""

        intersection = partial.mask & full.mask
        full_code = full_code64
        partial_code = partial.z.double()
        intersection_f = intersection.unsqueeze(-1).double()
        full_intersection = full_code * intersection_f
        partial_intersection = partial_code * intersection_f
        full_mapped = torch.einsum(
            "ngb,gbc->ngc",
            full_intersection,
            all_site_decoder_gram,
        )
        full_q = (full_mapped * full_intersection).sum(dim=-1)
        partial_q = torch.einsum(
            "ngb,gbc,ngc->ng",
            partial_intersection,
            all_site_decoder_gram,
            partial_intersection,
        )
        cross = (full_mapped * partial_intersection).sum(dim=-1)
        all_full_q = all_full_q64
        intersection_count[index] += intersection.sum(dim=0).double()
        full_count[index] += full_mask_count64
        concordance_numerator[index] += 2.0 * cross.sum(dim=0)
        concordance_denominator[index] += full_q.sum(dim=0) + partial_q.sum(dim=0)
        full_code_sum[index] += full_intersection.sum(dim=0)
        partial_code_sum[index] += partial_intersection.sum(dim=0)
        intersection_full_energy[index] += full_q.sum(dim=0)
        full_energy[index] += all_full_q.sum(dim=0)

    for raw in batches:
        if max_tokens is not None and n_tokens >= max_tokens:
            break
        x = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        if max_tokens is not None:
            x = x[: max_tokens - n_tokens]
        if not x.numel():
            break
        full, _, _ = model.forward_with_materialized(
            x,
            mode=selection_mode,
            _decoder=materialized_decoder,
            _encoder=materialized_encoder,
        )
        x64 = x.double()
        valid = x64 * coord if has_padded_coordinates else x64
        target_sum += valid.sum(dim=0)
        target_sumsq += valid.square().sum(dim=0)
        full_residual = x64 - full.xhat.double()
        if has_padded_coordinates:
            full_residual = full_residual * coord
        full_sse += full_residual.square().sum(dim=(0, 2))
        full_code64 = full.z.double()
        full_selected64 = full.z_selected.double()
        full_mask_count64 = full.mask.sum(dim=0).double()
        all_full_q64 = torch.einsum(
            "ngb,gbc,ngc->ng",
            full_selected64,
            all_site_decoder_gram,
            full_selected64,
        )
        fire += full_mask_count64
        zsum += full_selected64.sum(dim=0)
        zz += torch.einsum("ngb,ngc->gbc", full_selected64, full_selected64)
        observed_all = torch.ones(x.shape[0], S, dtype=torch.bool, device=x.device)
        zero_x: torch.Tensor | None = None

        def run_view(
            view_observed: torch.Tensor,
            *,
            source_missing: bool,
            empty: bool,
        ):
            """Run a partial view, using a zero-information null encoding.

            Source-only fusion cannot encode a view which omits its declared
            source, and the one-site LOO view has no observed site at all.
            In those cases a zero input with one synthetic observation bit is
            the operational null: it preserves learned encoder/decoder biases
            without leaking any held-out activation.
            """
            if source_missing or empty:
                nonlocal zero_x
                if zero_x is None:
                    zero_x = torch.zeros_like(x)
                null_observed = torch.zeros_like(view_observed)
                fallback = cfg.source_site if cfg.encoder_fusion == "source" else 0
                null_observed[:, fallback] = True
                return model.forward_with_materialized(
                    zero_x,
                    mode=selection_mode,
                    observed=null_observed,
                    validate_observed=False,
                    _decoder=materialized_decoder,
                    _encoder=materialized_encoder,
                )[0]
            return model.forward_with_materialized(
                x,
                mode=selection_mode,
                observed=view_observed,
                validate_observed=False,
                _decoder=materialized_decoder,
                _encoder=materialized_encoder,
            )[0]

        for source in range(S):
            only_observed = torch.zeros_like(observed_all)
            only_observed[:, source] = True
            only = run_view(
                only_observed,
                source_missing=(
                    cfg.encoder_fusion == "source" and source != cfg.source_site
                ),
                empty=False,
            )
            site_residual = x64 - only.xhat.double()
            if has_padded_coordinates:
                site_residual = site_residual * coord
            site_sse[source] += site_residual.square().sum(dim=(0, 2))
            support_intersection[source] += (only.mask & full.mask).sum()
            support_union[source] += (only.mask | full.mask).sum()
            accumulate_coordinate_concordance(
                only,
                full,
                source,
                intersection_count=site_intersection_count,
                full_count=site_full_count,
                concordance_numerator=site_concordance_numerator,
                concordance_denominator=site_concordance_denominator,
                full_code_sum=site_full_code_sum,
                partial_code_sum=site_partial_code_sum,
                intersection_full_energy=site_intersection_full_energy,
                full_energy=site_full_energy,
            )

            missing_observed = observed_all.clone()
            missing_observed[:, source] = False
            missing = run_view(
                missing_observed,
                source_missing=(
                    cfg.encoder_fusion == "source" and source == cfg.source_site
                ),
                empty=S == 1,
            )
            loo_residual = x64 - missing.xhat.double()
            if has_padded_coordinates:
                loo_residual = loo_residual * coord
            loo_sse[source] += loo_residual.square().sum(dim=(0, 2))
            loo_intersection[source] += (missing.mask & full.mask).sum()
            loo_union[source] += (missing.mask | full.mask).sum()
            accumulate_coordinate_concordance(
                missing,
                full,
                source,
                intersection_count=loo_intersection_count,
                full_count=loo_full_count,
                concordance_numerator=loo_concordance_numerator,
                concordance_denominator=loo_concordance_denominator,
                full_code_sum=loo_full_code_sum,
                partial_code_sum=loo_partial_code_sum,
                intersection_full_energy=loo_intersection_full_energy,
                full_energy=loo_full_energy,
            )
            pre_selection_loo_delta_sq[source] += (
                (missing.z.double() - full_code64).square().sum(dim=(0, 2))
            )
            post_selection_loo_delta_sq[source] += (
                (missing.z_selected.double() - full_selected64)
                .square()
                .sum(dim=(0, 2))
            )
        n_tokens += x.shape[0]

    if n_tokens == 0:
        raise ValueError("evaluation stream produced no tokens")
    centered_ss = target_sumsq - target_sum.square() / n_tokens
    denominator = centered_ss.sum(dim=1).clamp_min(1e-30)
    full_fvu = full_sse / denominator
    site_matrix = site_sse / denominator.unsqueeze(0)
    loo_matrix = loo_sse / denominator.unsqueeze(0)

    # Functional-dependence profiles are descriptive block endpoints. For
    # each omitted site and block, delta is the RMS Euclidean code change over
    # tokens. The max-normalized site profile has maximum one, and its sum is
    # C in [1,S] whenever a block has nonzero dependence. C=0 marks an exactly
    # invariant/zero block. Larger C means dependence is distributed across
    # more sites; it is not a universal quality ordering (local features and
    # broad cross-layer features can both be scientifically meaningful).
    pre_delta = (pre_selection_loo_delta_sq / n_tokens).clamp_min(0).sqrt()
    post_delta = (post_selection_loo_delta_sq / n_tokens).clamp_min(0).sqrt()

    def normalized_profile(
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum = delta.max(dim=0).values
        defined = maximum > 0
        profile = torch.where(
            defined.unsqueeze(0),
            delta / maximum.clamp_min(torch.finfo(delta.dtype).tiny).unsqueeze(0),
            torch.zeros_like(delta),
        )
        return profile, profile.sum(dim=0), defined

    pre_profile, pre_coherence, pre_defined = normalized_profile(pre_delta)
    post_profile, post_coherence, post_defined = normalized_profile(post_delta)

    def concordance_payload(
        numerator: torch.Tensor,
        concordance_denominator: torch.Tensor,
        full_code_sum: torch.Tensor,
        partial_code_sum: torch.Tensor,
        intersection_count: torch.Tensor,
        full_count: torch.Tensor,
        intersection_full_energy: torch.Tensor,
        full_energy: torch.Tensor,
    ) -> dict[str, object]:
        safe_count = intersection_count.clamp_min(1.0)
        mean_full = full_code_sum / safe_count.unsqueeze(-1)
        mean_partial = partial_code_sum / safe_count.unsqueeze(-1)
        means = torch.stack((mean_full, mean_partial))
        mapped_means = torch.einsum(
            "xsgb,gbc->xsgc",
            means,
            all_site_decoder_gram,
        )
        full_mean_q = (mapped_means[0] * mean_full).sum(dim=-1)
        partial_mean_q = (mapped_means[1] * mean_partial).sum(dim=-1)
        mean_cross = (mapped_means[0] * mean_partial).sum(dim=-1)
        centered_numerator = numerator - 2.0 * intersection_count * mean_cross
        centered_denominator = concordance_denominator - intersection_count * (
            full_mean_q + partial_mean_q
        )
        mean_delta = mean_full - mean_partial
        mean_offset_energy = intersection_count * (
            (mapped_means[0] - mapped_means[1]) * mean_delta
        ).sum(
            dim=-1
        )
        # Lin-style concordance in decoded-coordinate geometry.  Centering the
        # covariance prevents a common bias from dominating, while the mean-
        # offset term prevents an additive partial-view coordinate drift from
        # being mistaken for exact agreement.
        concordance_denominator = centered_denominator + mean_offset_energy
        scale = concordance_denominator.abs().clamp_min(1.0)
        if bool((concordance_denominator < -1e-9 * scale).any()):
            raise ValueError("coordinate concordance energy became materially negative")
        concordance_denominator = concordance_denominator.clamp_min(0.0)
        micro_numerator = centered_numerator.sum(dim=1)
        micro_denominator = concordance_denominator.sum(dim=1)
        micro_intersection_count = intersection_count.sum(dim=1)
        micro_full_count = full_count.sum(dim=1)
        micro_intersection_energy = intersection_full_energy.sum(dim=1)
        micro_full_energy = full_energy.sum(dim=1)
        concordance = torch.where(
            micro_denominator > 0,
            micro_numerator / micro_denominator,
            torch.zeros_like(micro_numerator),
        )
        recall = torch.where(
            micro_full_count > 0,
            micro_intersection_count / micro_full_count,
            torch.zeros_like(micro_intersection_count),
        )
        energy_coverage = torch.where(
            micro_full_energy > 0,
            micro_intersection_energy / micro_full_energy,
            torch.zeros_like(micro_intersection_energy),
        )
        tolerance = 1e-9
        if (
            bool((concordance < -1.0 - tolerance).any())
            or bool((concordance > 1.0 + tolerance).any())
            or bool((recall < -tolerance).any())
            or bool((recall > 1.0 + tolerance).any())
            or bool((energy_coverage < -tolerance).any())
            or bool((energy_coverage > 1.0 + tolerance).any())
        ):
            raise ValueError("partial-view coordinate metric left its algebraic range")
        concordance = concordance.clamp(-1.0, 1.0)
        recall = recall.clamp(0.0, 1.0)
        energy_coverage = energy_coverage.clamp(0.0, 1.0)

        block_concordance = torch.where(
            concordance_denominator > 0,
            centered_numerator / concordance_denominator,
            torch.zeros_like(centered_numerator),
        )
        block_energy_coverage = torch.where(
            full_energy > 0,
            intersection_full_energy / full_energy,
            torch.zeros_like(intersection_full_energy),
        )
        block_recall = torch.where(
            full_count > 0,
            intersection_count / full_count,
            torch.zeros_like(intersection_count),
        )
        minimum_events = 32
        concordance_eligible = (intersection_count >= minimum_events) & (
            concordance_denominator > 0
        )
        recall_eligible = full_count > 0
        energy_eligible = full_energy > 0
        if (
            bool((block_concordance[concordance_eligible] < -1.0 - tolerance).any())
            or bool((block_concordance[concordance_eligible] > 1.0 + tolerance).any())
            or bool((block_recall[recall_eligible] < -tolerance).any())
            or bool((block_recall[recall_eligible] > 1.0 + tolerance).any())
            or bool((block_energy_coverage[energy_eligible] < -tolerance).any())
            or bool((block_energy_coverage[energy_eligible] > 1.0 + tolerance).any())
        ):
            raise ValueError("per-block coordinate metric left its algebraic range")
        block_concordance = block_concordance.clamp(-1.0, 1.0)
        block_recall = block_recall.clamp(0.0, 1.0)
        block_energy_coverage = block_energy_coverage.clamp(0.0, 1.0)

        def distribution(
            values: torch.Tensor,
            eligible: torch.Tensor,
        ) -> dict[str, float | int | list[int] | None]:
            selected = values[eligible]
            eligible_count = int(selected.numel())
            total_count = int(eligible.numel())
            if selected.numel() == 0:
                return {
                    "eligible_block_pattern_count": 0,
                    "ineligible_block_pattern_count": total_count,
                    "eligible_block_patterns_per_site": (
                        eligible.sum(dim=1).cpu().tolist()
                    ),
                    "median": None,
                    "q10": None,
                    "fraction_at_least_0p5": None,
                }
            return {
                "eligible_block_pattern_count": eligible_count,
                "ineligible_block_pattern_count": total_count - eligible_count,
                "eligible_block_patterns_per_site": (
                    eligible.sum(dim=1).cpu().tolist()
                ),
                "median": float(selected.median()),
                "q10": float(torch.quantile(selected, 0.10)),
                "fraction_at_least_0p5": float((selected >= 0.5).double().mean()),
            }

        return {
            "concordance": concordance.cpu().tolist(),
            "support_intersection_recall": recall.cpu().tolist(),
            "decoded_energy_coverage": energy_coverage.cpu().tolist(),
            "per_block_distribution": {
                "minimum_intersection_events_for_concordance": minimum_events,
                "total_block_pattern_count": S * G,
                "eligibility_contract": {
                    "concordance": (
                        "intersection_count>=32_and_positive_concordance_denominator"
                    ),
                    "support_intersection_recall": "positive_full_support_count",
                    "decoded_energy_coverage": "positive_full_decoded_energy",
                },
                "concordance": distribution(
                    block_concordance,
                    concordance_eligible,
                ),
                "support_intersection_recall": distribution(
                    block_recall,
                    recall_eligible,
                ),
                "decoded_energy_coverage": distribution(
                    block_energy_coverage,
                    energy_eligible,
                ),
            },
        }

    site_coordinate = concordance_payload(
        site_concordance_numerator,
        site_concordance_denominator,
        site_full_code_sum,
        site_partial_code_sum,
        site_intersection_count,
        site_full_count,
        site_intersection_full_energy,
        site_full_energy,
    )
    loo_coordinate = concordance_payload(
        loo_concordance_numerator,
        loo_concordance_denominator,
        loo_full_code_sum,
        loo_partial_code_sum,
        loo_intersection_count,
        loo_full_count,
        loo_intersection_full_energy,
        loo_full_energy,
    )

    # Used dimension is estimated from the *centered conditional covariance*
    # of active codes and each effective site decoder Gram.  Centering keeps a
    # constant nonzero code from masquerading as a varying used direction.
    # The algebra is batched in bounded chunks; a Python loop over S*G is
    # prohibitive for Phase-3 scalar dictionaries.
    used_eigenvalues = torch.zeros(S, G, b, dtype=torch.float64, device=device)
    chunk_size = 4096
    for start in range(0, G, chunk_size):
        stop = min(start + chunk_size, G)
        denominator_g = fire[start:stop].clamp_min(1.0)
        mean_z = zsum[start:stop] / denominator_g.unsqueeze(-1)
        covariance = zz[start:stop] / denominator_g[:, None, None] - torch.einsum(
            "gi,gj->gij", mean_z, mean_z
        )
        covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
        if b == 1:
            used_eigenvalues[:, start:stop, 0] = (
                covariance[:, 0, 0].clamp_min(0).unsqueeze(0)
                * decoder_gram[:, start:stop, 0, 0]
            )
            continue
        root_eval, root_vec = torch.linalg.eigh(covariance)
        root = torch.matmul(
            root_vec * root_eval.clamp_min(0).sqrt().unsqueeze(-2),
            root_vec.transpose(-1, -2),
        )
        contribution = torch.matmul(
            torch.matmul(root.unsqueeze(0), decoder_gram[:, start:stop]),
            root.unsqueeze(0),
        )
        used_eigenvalues[:, start:stop] = torch.linalg.eigvalsh(contribution).flip(-1)

    payload = {
        "schema_version": 5,
        "selection_mode": selection_mode,
        "n_tokens": n_tokens,
        "model_cfg": asdict(cfg),
        "full_fvu_per_site": full_fvu.tolist(),
        "full_fvu_pooled": float(full_sse.sum() / denominator.sum()),
        "site_only_fvu": site_matrix.tolist(),
        "leave_one_site_out_fvu": loo_matrix.tolist(),
        "site_only_support_iou": (
            support_intersection / support_union.clamp_min(1)
        ).tolist(),
        "leave_one_site_out_support_iou": (
            loo_intersection / loo_union.clamp_min(1)
        ).tolist(),
        "partial_view_coordinate_concordance": {
            "definition": (
                "lin_decoder_gram_concordance_covariance_over_variance_plus_mean_offset"
            ),
            "decoder_gram": "sum_all_sites_D_g_s_D_g_s_transpose",
            "support_contract": "all_view_partial_view_support_intersection",
            "site_only": site_coordinate,
            "leave_one_site_out": loo_coordinate,
        },
        "functional_dependence": {
            "delta_definition": ("rms_l2_code_change_when_site_is_omitted_over_tokens"),
            "profile_normalization": "divide_each_block_by_its_max_site_delta",
            "coherence_definition": "sum_site_delta_divided_by_max_site_delta",
            "interpretation": (
                "descriptive_only; larger_coherence_is_not_universally_better"
            ),
            "pre_selection": {
                "delta_by_site_block": pre_delta.cpu().tolist(),
                "normalized_profile_by_site_block": pre_profile.cpu().tolist(),
                "coherence_per_block": pre_coherence.cpu().tolist(),
                "defined_per_block": pre_defined.cpu().tolist(),
            },
            "post_selection": {
                "delta_by_site_block": post_delta.cpu().tolist(),
                "normalized_profile_by_site_block": post_profile.cpu().tolist(),
                "coherence_per_block": post_coherence.cpu().tolist(),
                "defined_per_block": post_defined.cpu().tolist(),
            },
        },
        "fire_count": fire.cpu().tolist(),
        "used_contribution_eigenvalues": used_eigenvalues.cpu().tolist(),
    }
    # JSON round-trip is a cheap schema guard against accidental tensors/NaNs.
    json.dumps(payload, allow_nan=False)
    return payload
