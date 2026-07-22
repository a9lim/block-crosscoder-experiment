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

from .model import BSCConfig, BSCOutput, BlockCrosscoder
from .runtime_limits import (
    EVALUATION_CONCORDANCE_BLOCK_CHUNK,
    EVALUATION_REDUCTION_TOKEN_CHUNK,
)

__all__ = [
    "centered_fvu",
    "checkpoint_sha256",
    "evaluate_shared_code",
    "evaluate_shared_code_modes",
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
def evaluate_shared_code_modes(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
    selection_modes: tuple[str, ...] = ("topk", "threshold"),
) -> dict[str, dict]:
    """Evaluate one or more selector endpoints over one shared view pass.

    Site-only and leave-one-site-out views are operational re-encodings through
    the model, so biases, nonlinear scores, source-only fusion, and
    dense scaffolds retain their actual semantics. No direct ``model.E`` access
    is used. Encoding, scoring, target statistics, decoder geometry, and the
    pre-selection dependence profile are selector-independent and therefore
    computed once. Reconstruction and every selector-dependent reduction remain
    isolated by mode.
    """
    if (
        not selection_modes
        or len(set(selection_modes)) != len(selection_modes)
        or any(mode not in {"topk", "threshold"} for mode in selection_modes)
    ):
        raise ValueError(
            "selection_modes must be a nonempty unique tuple drawn from "
            "{'topk', 'threshold'}"
        )
    model = model.to(device).eval()
    cfg = model.cfg
    S, G, b = cfg.n_sites, cfg.n_blocks, cfg.block_dim
    coord = model.coordinate_mask[:, 0, 0].to(device)
    has_padded_coordinates = model._has_padded_coordinates

    target_sum = torch.zeros(S, cfg.d_model, dtype=torch.float64, device=device)
    target_sumsq = torch.zeros_like(target_sum)
    pre_selection_loo_delta_sq = torch.zeros(S, G, dtype=torch.float64, device=device)

    def new_mode_state() -> dict[str, torch.Tensor]:
        site_block = torch.zeros(S, G, dtype=torch.float64, device=device)
        site_block_code = torch.zeros(S, G, b, dtype=torch.float64, device=device)
        return {
            "full_sse": torch.zeros(S, dtype=torch.float64, device=device),
            "site_sse": torch.zeros(S, S, dtype=torch.float64, device=device),
            "loo_sse": torch.zeros(S, S, dtype=torch.float64, device=device),
            "support_intersection": torch.zeros(
                S, dtype=torch.float64, device=device
            ),
            "support_union": torch.zeros(S, dtype=torch.float64, device=device),
            "loo_intersection": torch.zeros(S, dtype=torch.float64, device=device),
            "loo_union": torch.zeros(S, dtype=torch.float64, device=device),
            "site_intersection_count": site_block.clone(),
            "site_full_count": site_block.clone(),
            "site_concordance_numerator": site_block.clone(),
            "site_concordance_denominator": site_block.clone(),
            "site_full_code_sum": site_block_code.clone(),
            "site_partial_code_sum": site_block_code.clone(),
            "site_intersection_full_energy": site_block.clone(),
            "site_full_energy": site_block.clone(),
            "loo_intersection_count": site_block.clone(),
            "loo_full_count": site_block.clone(),
            "loo_concordance_numerator": site_block.clone(),
            "loo_concordance_denominator": site_block.clone(),
            "loo_full_code_sum": site_block_code.clone(),
            "loo_partial_code_sum": site_block_code.clone(),
            "loo_intersection_full_energy": site_block.clone(),
            "loo_full_energy": site_block.clone(),
            "post_selection_loo_delta_sq": site_block.clone(),
            "fire": torch.zeros(G, dtype=torch.float64, device=device),
            "zsum": torch.zeros(G, b, dtype=torch.float64, device=device),
            "zz": torch.zeros(G, b, b, dtype=torch.float64, device=device),
        }

    states = {mode: new_mode_state() for mode in selection_modes}
    materialized_decoder = model.decoder_tensor()
    materialized_encoder = (
        materialized_decoder * model.log_gamma.exp()
        if cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    score_geometry = model._frozen_score_geometry(materialized_decoder)
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

    def token_slices(n: int):
        chunk = (
            EVALUATION_REDUCTION_TOKEN_CHUNK
            if torch.device(device).type == "cuda"
            else n
        )
        for start in range(0, n, chunk):
            yield slice(start, min(start + chunk, n))

    def accumulate_target_statistics(target: torch.Tensor) -> None:
        for token_slice in token_slices(len(target)):
            values = target[token_slice].double()
            if has_padded_coordinates:
                values = values * coord
            target_sum.add_(values.sum(dim=0))
            target_sumsq.add_(values.square().sum(dim=0))

    def squared_error_by_site(
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros(S, dtype=torch.float64, device=device)
        for token_slice in token_slices(len(target)):
            residual = target[token_slice].double() - prediction[token_slice].double()
            if has_padded_coordinates:
                residual = residual * coord
            result.add_(residual.square().sum(dim=(0, 2)))
        return result

    def accumulate_coordinate_concordance(
        partial,
        full,
        index: int,
        *,
        full_mask_count64: torch.Tensor,
        all_full_energy64: torch.Tensor,
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

        full_count[index] += full_mask_count64
        full_energy[index] += all_full_energy64
        for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
            stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
            block_slice = slice(start, stop)
            intersection = (
                partial.mask[:, block_slice] & full.mask[:, block_slice]
            )
            full_code = full.z[:, block_slice].double()
            partial_code = partial.z[:, block_slice].double()
            intersection_f = intersection.unsqueeze(-1).double()
            full_intersection = full_code * intersection_f
            partial_intersection = partial_code * intersection_f
            gram = all_site_decoder_gram[block_slice]
            if b == 1:
                full_scalar = full_intersection[..., 0]
                partial_scalar = partial_intersection[..., 0]
                weight = gram[:, 0, 0].unsqueeze(0)
                full_q = full_scalar.square() * weight
                partial_q = partial_scalar.square() * weight
                cross = full_scalar * partial_scalar * weight
            else:
                full_mapped = torch.einsum(
                    "ngb,gbc->ngc",
                    full_intersection,
                    gram,
                )
                full_q = (full_mapped * full_intersection).sum(dim=-1)
                partial_q = torch.einsum(
                    "ngb,gbc,ngc->ng",
                    partial_intersection,
                    gram,
                    partial_intersection,
                )
                cross = (full_mapped * partial_intersection).sum(dim=-1)
            intersection_count[index, block_slice] += intersection.sum(
                dim=0
            ).double()
            concordance_numerator[index, block_slice] += 2.0 * cross.sum(dim=0)
            concordance_denominator[index, block_slice] += full_q.sum(
                dim=0
            ) + partial_q.sum(dim=0)
            full_code_sum[index, block_slice] += full_intersection.sum(dim=0)
            partial_code_sum[index, block_slice] += partial_intersection.sum(dim=0)
            intersection_full_energy[index, block_slice] += full_q.sum(dim=0)

    primary_mode = selection_modes[0]

    def outputs_for_view(
        value: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        encoder_sites=None,
    ) -> dict[str, BSCOutput]:
        selection, _, _ = model.select_with_materialized(
            value,
            mode=primary_mode,
            observed=observed,
            validate_observed=validate_observed,
            _decoder=materialized_decoder,
            _encoder=materialized_encoder,
            _score_geometry=score_geometry,
            _encoder_sites=encoder_sites,
        )
        result: dict[str, BSCOutput] = {}
        for mode in selection_modes:
            if mode == primary_mode:
                mask = selection.mask
                selected = selection.z_selected
            else:
                mask = model._select_scores(
                    selection.scores,
                    mode=mode,
                    z=selection.z,
                )
                selected = selection.z * mask.unsqueeze(-1)
            prediction = model.decode(selected, _decoder=materialized_decoder)
            result[mode] = BSCOutput(
                prediction,
                selection.z,
                selected,
                selection.scores,
                mask,
            )
        return result

    for raw in batches:
        if max_tokens is not None and n_tokens >= max_tokens:
            break
        x = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        if max_tokens is not None:
            x = x[: max_tokens - n_tokens]
        if not x.numel():
            break
        encoder_sites = (
            None
            if cfg.encoder_fusion == "source" or S == 1
            else model._frozen_encoder_sites(x, materialized_encoder)
        )
        full_outputs = outputs_for_view(x, encoder_sites=encoder_sites)
        accumulate_target_statistics(x)
        full_mask_counts: dict[str, torch.Tensor] = {}
        full_energies: dict[str, torch.Tensor] = {}
        for mode, full in full_outputs.items():
            state = states[mode]
            state["full_sse"] += squared_error_by_site(x, full.xhat)
            full_mask_count64 = full.mask.sum(dim=0).double()
            all_full_energy64 = torch.zeros(
                G, dtype=torch.float64, device=device
            )
            state["fire"] += full_mask_count64
            for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
                stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
                block_slice = slice(start, stop)
                selected = full.z_selected[:, block_slice].double()
                gram = all_site_decoder_gram[block_slice]
                if b == 1:
                    selected_energy = (
                        selected[..., 0].square() * gram[:, 0, 0].unsqueeze(0)
                    )
                else:
                    selected_energy = torch.einsum(
                        "ngb,gbc,ngc->ng",
                        selected,
                        gram,
                        selected,
                    )
                all_full_energy64[block_slice] = selected_energy.sum(dim=0)
                state["zsum"][block_slice] += selected.sum(dim=0)
                state["zz"][block_slice] += torch.einsum(
                    "ngb,ngc->gbc", selected, selected
                )
            full_mask_counts[mode] = full_mask_count64
            full_energies[mode] = all_full_energy64
        del selected, selected_energy, gram

        observed_all = torch.ones(x.shape[0], S, dtype=torch.bool, device=x.device)
        zero_x: torch.Tensor | None = None
        null_outputs: dict[str, BSCOutput] | None = None

        def run_view(
            view_observed: torch.Tensor,
            *,
            source_missing: bool,
            empty: bool,
            frozen_encoder_sites,
        ) -> dict[str, BSCOutput]:
            """Run one mode-independent partial encoding and both selectors."""
            if source_missing or empty:
                nonlocal zero_x, null_outputs
                if null_outputs is not None:
                    return null_outputs
                if zero_x is None:
                    zero_x = torch.zeros_like(x)
                null_observed = torch.zeros_like(view_observed)
                fallback = cfg.source_site if cfg.encoder_fusion == "source" else 0
                null_observed[:, fallback] = True
                null_outputs = outputs_for_view(
                    zero_x,
                    observed=null_observed,
                    validate_observed=False,
                )
                return null_outputs
            if cfg.encoder_fusion == "source" or S == 1:
                return full_outputs
            return outputs_for_view(
                x,
                observed=view_observed,
                validate_observed=False,
                encoder_sites=frozen_encoder_sites,
            )

        for source in range(S):
            only_observed = torch.zeros_like(observed_all)
            only_observed[:, source] = True
            only_outputs = run_view(
                only_observed,
                source_missing=(
                    cfg.encoder_fusion == "source" and source != cfg.source_site
                ),
                empty=False,
                frozen_encoder_sites=encoder_sites,
            )
            for mode, only in only_outputs.items():
                full = full_outputs[mode]
                state = states[mode]
                state["site_sse"][source] += squared_error_by_site(x, only.xhat)
                state["support_intersection"][source] += (
                    only.mask & full.mask
                ).sum()
                state["support_union"][source] += (only.mask | full.mask).sum()
                accumulate_coordinate_concordance(
                    only,
                    full,
                    source,
                    full_mask_count64=full_mask_counts[mode],
                    all_full_energy64=full_energies[mode],
                    intersection_count=state["site_intersection_count"],
                    full_count=state["site_full_count"],
                    concordance_numerator=state["site_concordance_numerator"],
                    concordance_denominator=state["site_concordance_denominator"],
                    full_code_sum=state["site_full_code_sum"],
                    partial_code_sum=state["site_partial_code_sum"],
                    intersection_full_energy=state[
                        "site_intersection_full_energy"
                    ],
                    full_energy=state["site_full_energy"],
                )
            del only, full, state, only_outputs

            missing_observed = observed_all.clone()
            missing_observed[:, source] = False
            missing_outputs = run_view(
                missing_observed,
                source_missing=(
                    cfg.encoder_fusion == "source" and source == cfg.source_site
                ),
                empty=S == 1,
                frozen_encoder_sites=encoder_sites,
            )
            primary_missing = missing_outputs[primary_mode]
            primary_full = full_outputs[primary_mode]
            for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
                stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
                block_slice = slice(start, stop)
                pre_selection_loo_delta_sq[source, block_slice] += (
                    (
                        primary_missing.z[:, block_slice].double()
                        - primary_full.z[:, block_slice].double()
                    )
                    .square()
                    .sum(dim=(0, 2))
                )
            for mode, missing in missing_outputs.items():
                full = full_outputs[mode]
                state = states[mode]
                state["loo_sse"][source] += squared_error_by_site(x, missing.xhat)
                state["loo_intersection"][source] += (
                    missing.mask & full.mask
                ).sum()
                state["loo_union"][source] += (missing.mask | full.mask).sum()
                accumulate_coordinate_concordance(
                    missing,
                    full,
                    source,
                    full_mask_count64=full_mask_counts[mode],
                    all_full_energy64=full_energies[mode],
                    intersection_count=state["loo_intersection_count"],
                    full_count=state["loo_full_count"],
                    concordance_numerator=state["loo_concordance_numerator"],
                    concordance_denominator=state["loo_concordance_denominator"],
                    full_code_sum=state["loo_full_code_sum"],
                    partial_code_sum=state["loo_partial_code_sum"],
                    intersection_full_energy=state[
                        "loo_intersection_full_energy"
                    ],
                    full_energy=state["loo_full_energy"],
                )
                for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
                    stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
                    block_slice = slice(start, stop)
                    state["post_selection_loo_delta_sq"][source, block_slice] += (
                        (
                            missing.z_selected[:, block_slice].double()
                            - full.z_selected[:, block_slice].double()
                        )
                        .square()
                        .sum(dim=(0, 2))
                    )
            del missing, full, state, primary_missing, primary_full, missing_outputs
        del (
            encoder_sites,
            null_outputs,
            full_outputs,
            full_mask_counts,
            full_energies,
            full_mask_count64,
            all_full_energy64,
        )
        n_tokens += x.shape[0]

    if n_tokens == 0:
        raise ValueError("evaluation stream produced no tokens")
    centered_ss = target_sumsq - target_sum.square() / n_tokens
    denominator = centered_ss.sum(dim=1).clamp_min(1e-30)

    # Functional-dependence profiles are descriptive block endpoints. For
    # each omitted site and block, delta is the RMS Euclidean code change over
    # tokens. The max-normalized site profile has maximum one, and its sum is
    # C in [1,S] whenever a block has nonzero dependence. C=0 marks an exactly
    # invariant/zero block. Larger C means dependence is distributed across
    # more sites; it is not a universal quality ordering (local features and
    # broad cross-layer features can both be scientifically meaningful).
    pre_delta = (pre_selection_loo_delta_sq / n_tokens).clamp_min(0).sqrt()

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

    def finalize_mode(
        selection_mode: str,
        state: dict[str, torch.Tensor],
    ) -> dict:
        full_fvu = state["full_sse"] / denominator
        site_matrix = state["site_sse"] / denominator.unsqueeze(0)
        loo_matrix = state["loo_sse"] / denominator.unsqueeze(0)
        post_delta = (
            state["post_selection_loo_delta_sq"] / n_tokens
        ).clamp_min(0).sqrt()
        post_profile, post_coherence, post_defined = normalized_profile(post_delta)

        site_coordinate = concordance_payload(
            state["site_concordance_numerator"],
            state["site_concordance_denominator"],
            state["site_full_code_sum"],
            state["site_partial_code_sum"],
            state["site_intersection_count"],
            state["site_full_count"],
            state["site_intersection_full_energy"],
            state["site_full_energy"],
        )
        loo_coordinate = concordance_payload(
            state["loo_concordance_numerator"],
            state["loo_concordance_denominator"],
            state["loo_full_code_sum"],
            state["loo_partial_code_sum"],
            state["loo_intersection_count"],
            state["loo_full_count"],
            state["loo_intersection_full_energy"],
            state["loo_full_energy"],
        )

        # Used dimension is estimated from the *centered conditional
        # covariance* of active codes and each effective site decoder Gram.
        # Centering keeps a constant nonzero code from masquerading as a
        # varying used direction. The algebra is batched in bounded chunks; a
        # Python loop over S*G is prohibitive for Phase-3 scalar dictionaries.
        used_eigenvalues = torch.zeros(
            S, G, b, dtype=torch.float64, device=device
        )
        chunk_size = 4096
        for start in range(0, G, chunk_size):
            stop = min(start + chunk_size, G)
            denominator_g = state["fire"][start:stop].clamp_min(1.0)
            mean_z = state["zsum"][start:stop] / denominator_g.unsqueeze(-1)
            covariance = state["zz"][start:stop] / denominator_g[
                :, None, None
            ] - torch.einsum("gi,gj->gij", mean_z, mean_z)
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
            used_eigenvalues[:, start:stop] = torch.linalg.eigvalsh(
                contribution
            ).flip(-1)

        payload = {
            "schema_version": 5,
            "selection_mode": selection_mode,
            "n_tokens": n_tokens,
            "model_cfg": asdict(cfg),
            "full_fvu_per_site": full_fvu.tolist(),
            "full_fvu_pooled": float(
                state["full_sse"].sum() / denominator.sum()
            ),
            "site_only_fvu": site_matrix.tolist(),
            "leave_one_site_out_fvu": loo_matrix.tolist(),
            "site_only_support_iou": (
                state["support_intersection"]
                / state["support_union"].clamp_min(1)
            ).tolist(),
            "leave_one_site_out_support_iou": (
                state["loo_intersection"] / state["loo_union"].clamp_min(1)
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
                "delta_definition": (
                    "rms_l2_code_change_when_site_is_omitted_over_tokens"
                ),
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
            "fire_count": state["fire"].cpu().tolist(),
            "used_contribution_eigenvalues": used_eigenvalues.cpu().tolist(),
        }
        # JSON round-trip is a cheap schema guard against accidental
        # tensors/NaNs.
        json.dumps(payload, allow_nan=False)
        return payload

    return {
        mode: finalize_mode(mode, states[mode])
        for mode in selection_modes
    }


@torch.no_grad()
def evaluate_shared_code(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
    selection_mode: str = "threshold",
) -> dict:
    """Evaluate one selector endpoint through the shared evaluator."""
    return evaluate_shared_code_modes(
        model,
        batches,
        device=device,
        max_tokens=max_tokens,
        selection_modes=(selection_mode,),
    )[selection_mode]
