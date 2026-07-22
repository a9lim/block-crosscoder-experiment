"""Method-aware reconstruction and shared-code endpoints.

Every caller supplies a checkpoint, data stream, and output path through a
resolved cell.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import torch

from .model import BSCConfig, BlockCrosscoder
from .runtime_limits import (
    EVALUATION_CONCORDANCE_BLOCK_CHUNK,
    EVALUATION_REDUCTION_TOKEN_CHUNK,
    EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR,
    MODEL_IMPLEMENTATION_IDENTITY_FIELDS,
)
from .trainer import validate_run_binding

__all__ = [
    "centered_fvu",
    "checkpoint_sha256",
    "EvaluationModeEndpoints",
    "evaluate_selector_and_shared_code_modes",
    "evaluate_shared_code",
    "evaluate_shared_code_modes",
    "load_trained_model",
]

_DECODER_GRAM_BLOCK_CHUNK = 256


class EvaluationModeEndpoints(NamedTuple):
    """Selector and shared-code payloads produced by one evaluation stream."""

    selector: dict[str, dict]
    shared_code: dict[str, dict]


class _ModeSelectedMoments(NamedTuple):
    """Mode-first reductions of one shared raw code geometry."""

    decoded_energy: torch.Tensor
    code_sum: torch.Tensor
    code_outer: torch.Tensor


class _ModeConcordanceReductions(NamedTuple):
    """Mode-first matched-support reductions for one concordance chunk."""

    intersection_count: torch.Tensor
    concordance_numerator: torch.Tensor
    concordance_denominator: torch.Tensor
    full_code_sum: torch.Tensor
    partial_code_sum: torch.Tensor
    intersection_full_energy: torch.Tensor


@dataclass(slots=True)
class _EvaluationViewOutput:
    """Lean evaluator state retained after one selector decode.

    The dense selected code is a decode input, not an endpoint dependency.
    Keeping a public ``BSCOutput`` here retained one full ``[N,G,b]`` tensor
    per selector and view until the concordance reductions completed.
    """

    xhat: torch.Tensor | None
    z: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor
    sse: torch.Tensor | None = None


def _raw_quadratic_terms(
    full_code: torch.Tensor,
    gram: torch.Tensor,
    partial_code: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute fp64 per-event decoder-Gram terms before selector masking."""

    block_width = full_code.shape[-1]
    if block_width == 1:
        full_scalar = full_code[..., 0]
        weight = gram[:, 0, 0].unsqueeze(0)
        full_q = full_scalar.square() * weight
        if partial_code is None:
            return full_q, None, None
        partial_scalar = partial_code[..., 0]
        return (
            full_q,
            partial_scalar.square() * weight,
            full_scalar * partial_scalar * weight,
        )

    full_mapped = torch.einsum("ngb,gbc->ngc", full_code, gram)
    full_q = (full_mapped * full_code).sum(dim=-1)
    if partial_code is None:
        return full_q, None, None
    partial_q = torch.einsum(
        "ngb,gbc,ngc->ng",
        partial_code,
        gram,
        partial_code,
    )
    cross = (full_mapped * partial_code).sum(dim=-1)
    return full_q, partial_q, cross


def _batched_mode_selected_moments(
    code: torch.Tensor,
    gram: torch.Tensor,
    mode_masks: torch.Tensor,
) -> _ModeSelectedMoments:
    """Reduce selected-code moments for every selector from one raw geometry.

    ``mode_masks`` is ``[M,N,G]``.  Applying it with ``torch.where`` is
    intentional: an unselected non-finite raw coordinate must not leak through
    a multiplication by zero into an otherwise finite selector endpoint.
    """

    code64 = code.double()
    if code.shape[-1] == 1:
        raw_energy = code64[..., 0].square() * gram[:, 0, 0].unsqueeze(0)
    else:
        # Retain the direct three-operand contraction.  Concordance uses a
        # mapped full code because it also needs the cross term, while this
        # standalone decoded-energy reduction must match per-mode execution.
        raw_energy = torch.einsum(
            "ngb,gbc,ngc->ng",
            code64,
            gram,
            code64,
        )
    zero = torch.zeros((), dtype=torch.float64, device=code.device)
    decoded_energy = torch.where(
        mode_masks,
        raw_energy.unsqueeze(0),
        zero,
    ).sum(dim=1)
    del raw_energy
    selected = torch.where(
        mode_masks.unsqueeze(-1),
        code64.unsqueeze(0),
        zero,
    )
    code_sum = selected.sum(dim=1)
    # Keep the per-mode contraction schedule.  It benchmarks faster at the
    # campaign payload size, and folding the mode axis into this einsum also
    # changes CUDA's reduction order for short final batches.
    code_outer = torch.stack(
        tuple(
            torch.einsum("ngb,ngc->gbc", selected[mode], selected[mode])
            for mode in range(mode_masks.shape[0])
        )
    )
    return _ModeSelectedMoments(decoded_energy, code_sum, code_outer)


def _batched_mode_concordance(
    full_code: torch.Tensor,
    partial_code: torch.Tensor,
    gram: torch.Tensor,
    full_masks: torch.Tensor,
    partial_masks: torch.Tensor,
) -> _ModeConcordanceReductions:
    """Reduce both selector modes from one raw fp64 concordance geometry."""

    full64 = full_code.double()
    partial64 = partial_code.double()
    full_q, partial_q, cross = _raw_quadratic_terms(full64, gram, partial64)
    assert partial_q is not None and cross is not None
    intersection = full_masks & partial_masks
    zero = torch.zeros((), dtype=torch.float64, device=full_code.device)

    # Materialize only one mode-expanded value family at a time.  This keeps
    # the peak below the existing 8*b+4 fp64 concordance workspace bound.
    full_code_sum = torch.where(
        intersection.unsqueeze(-1),
        full64.unsqueeze(0),
        zero,
    ).sum(dim=1)
    partial_code_sum = torch.where(
        intersection.unsqueeze(-1),
        partial64.unsqueeze(0),
        zero,
    ).sum(dim=1)
    masked_full_q = torch.where(intersection, full_q.unsqueeze(0), zero)
    intersection_full_energy = masked_full_q.sum(dim=1)
    masked_partial_q = torch.where(intersection, partial_q.unsqueeze(0), zero)
    masked_cross = torch.where(intersection, cross.unsqueeze(0), zero)
    return _ModeConcordanceReductions(
        intersection.sum(dim=1).double(),
        2.0 * masked_cross.sum(dim=1),
        intersection_full_energy + masked_partial_q.sum(dim=1),
        full_code_sum,
        partial_code_sum,
        intersection_full_energy,
    )


def _batched_mode_selected_delta_sq(
    full_code: torch.Tensor,
    partial_code: torch.Tensor,
    full_masks: torch.Tensor,
    partial_masks: torch.Tensor,
) -> torch.Tensor:
    """Mode-first selected-code squared deltas with exact mask suppression."""

    zero = torch.zeros((), dtype=torch.float64, device=full_code.device)
    full_selected = torch.where(
        full_masks.unsqueeze(-1),
        full_code.double().unsqueeze(0),
        zero,
    )
    partial_selected = torch.where(
        partial_masks.unsqueeze(-1),
        partial_code.double().unsqueeze(0),
        zero,
    )
    # Preserve the per-mode CUDA reduction schedule.  The mode-first multi-axis
    # sum does not improve campaign-scale throughput and is not bit-identical
    # on short final batches.
    return torch.stack(
        tuple(
            (partial_selected[mode] - full_selected[mode]).square().sum(dim=(0, 2))
            for mode in range(full_masks.shape[0])
        )
    )


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
    raw_cfg_payload = payload.get("model_cfg")
    if not isinstance(raw_cfg_payload, dict):
        raise ValueError("checkpoint lacks model configuration")
    cfg_payload = dict(raw_cfg_payload)
    for identity in MODEL_IMPLEMENTATION_IDENTITY_FIELDS:
        if identity not in cfg_payload:
            raise ValueError(f"checkpoint lacks {identity} identity")
    validate_run_binding(
        payload.get("run_binding"),
        {
            "model_cfg": payload["model_cfg"],
            "train_cfg": payload.get("train_cfg"),
        },
        keys=("model_cfg", "train_cfg"),
    )
    if cfg_payload.get("site_dims") is not None:
        cfg_payload["site_dims"] = tuple(cfg_payload["site_dims"])
    model = BlockCrosscoder(BSCConfig(**cfg_payload), device=device)
    model.load_state_dict(payload["model"])
    model.validate_decoded_energy_implementation()
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
def _decode_selected_for_evaluation(
    model: BlockCrosscoder,
    selected: torch.Tensor,
    mask: torch.Tensor,
    decoder: torch.Tensor,
) -> torch.Tensor:
    """Decode sparse CUDA support below the fixed density crossover.

    The event count is resolved before ``nonzero`` allocates a dynamic event
    stream.  Denser CUDA support and every non-CUDA device retain the native
    dense reduction.
    """
    if (
        not selected.is_cuda
        or selected.dtype != torch.float32
        or decoder.dtype != torch.float32
    ):
        return model.decode(selected, _decoder=decoder)

    counts = mask.sum(dim=1, dtype=torch.long)
    event_count = int(counts.sum().item())
    max_block_events = mask.numel() // EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR
    if event_count > max_block_events:
        return model.decode(selected, _decoder=decoder)

    cfg = model.cfg
    if event_count:
        prediction = torch.empty(
            selected.shape[0],
            cfg.n_sites,
            cfg.d_model,
            dtype=selected.dtype,
            device=selected.device,
        )
        events = mask.nonzero(as_tuple=False)
        rows = events[:, 0]
        block_ids = events[:, 1]
        values = selected[rows, block_ids]
        crow = torch.empty(
            selected.shape[0] + 1,
            dtype=torch.long,
            device=selected.device,
        )
        crow[0] = 0
        crow[1:].copy_(counts).mul_(cfg.block_dim).cumsum_(dim=0)
        block_coordinates = torch.arange(
            cfg.block_dim,
            dtype=torch.long,
            device=selected.device,
        ).unsqueeze(0)
        # Values no longer need dictionary IDs. Mutating this view of the
        # private event tensor avoids an additional int64 [events] start array.
        block_ids.mul_(cfg.block_dim)
        columns = (block_ids.unsqueeze(1) + block_coordinates).reshape(-1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            sparse_code = torch.sparse_csr_tensor(
                crow,
                columns,
                values.reshape(-1),
                size=(selected.shape[0], cfg.n_latents),
                device=selected.device,
                check_invariants=False,
            )
        decoder_by_site = decoder.reshape(
            cfg.n_sites,
            cfg.n_latents,
            cfg.d_model,
        )
        for site in range(cfg.n_sites):
            site_prediction = torch.sparse.mm(sparse_code, decoder_by_site[site])
            prediction[:, site].copy_(site_prediction)
            # Release the previous site before the next SpMM allocates its
            # output; the estimator prices exactly one [tokens, site_dim].
            del site_prediction
    else:
        prediction = torch.zeros(
            selected.shape[0],
            cfg.n_sites,
            cfg.d_model,
            dtype=selected.dtype,
            device=selected.device,
        )

    if cfg.decoder_bias:
        prediction.add_(model.c.unsqueeze(0))
    if model._has_padded_coordinates:
        prediction.mul_(model.coordinate_mask[:, 0, 0].to(prediction.dtype))
    return prediction


@torch.no_grad()
def _evaluate_code_modes(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
    selection_modes: tuple[str, ...] = ("topk", "threshold"),
    include_selector_payloads: bool,
) -> tuple[dict[str, dict], dict[str, dict] | None]:
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

    selector_target_sum = (
        torch.zeros_like(target_sum) if include_selector_payloads else None
    )
    selector_target_sumsq = (
        torch.zeros_like(target_sum) if include_selector_payloads else None
    )
    selector_coordinate_mask = (
        coord.double() if include_selector_payloads and has_padded_coordinates else None
    )
    selector_candidate_gain_counts = (
        torch.zeros(3, dtype=torch.int64, device=device)
        if include_selector_payloads
        else None
    )

    def new_mode_state() -> dict[str, torch.Tensor]:
        site_block = torch.zeros(S, G, dtype=torch.float64, device=device)
        site_block_code = torch.zeros(S, G, b, dtype=torch.float64, device=device)
        return {
            "full_sse": torch.zeros(S, dtype=torch.float64, device=device),
            "site_sse": torch.zeros(S, S, dtype=torch.float64, device=device),
            "loo_sse": torch.zeros(S, S, dtype=torch.float64, device=device),
            "support_intersection": torch.zeros(S, dtype=torch.float64, device=device),
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
    selector_states = (
        {
            mode: {
                "error": torch.zeros(S, dtype=torch.float64, device=device),
                "support_counts": torch.zeros(
                    G + 1,
                    dtype=torch.int64,
                    device=device,
                ),
                "selected_gain_counts": torch.zeros(
                    3,
                    dtype=torch.int64,
                    device=device,
                ),
            }
            for mode in selection_modes
        }
        if include_selector_payloads
        else None
    )
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
        site_gram_chunk = torch.einsum("sgbd,sgcd->sgbc", decoder_chunk, decoder_chunk)
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

    primary_mode = selection_modes[0]
    mode_states = tuple(states[mode] for mode in selection_modes)

    def outputs_for_view(
        value: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        encoder_sites=None,
    ) -> dict[str, _EvaluationViewOutput]:
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
        result: dict[str, _EvaluationViewOutput] = {}
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
            prediction = _decode_selected_for_evaluation(
                model,
                selected,
                mask,
                materialized_decoder,
            )
            result[mode] = _EvaluationViewOutput(
                prediction,
                selection.z,
                selection.scores,
                mask,
            )
        del selected, selection
        return result

    for raw in batches:
        if max_tokens is not None and n_tokens >= max_tokens:
            break
        x = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        if max_tokens is not None:
            x = x[: max_tokens - n_tokens]
        if not x.numel():
            break
        observed_all = torch.ones(x.shape[0], S, dtype=torch.bool, device=x.device)
        encoder_sites = (
            None
            if cfg.encoder_fusion == "source" or S == 1
            else model._frozen_encoder_sites(x, materialized_encoder)
        )
        # The legacy selector summaries explicitly supplied an all-observed
        # mask for isolated-loss scoring.  Preserve that operational path in
        # the joint evaluator; bypassing the partial-view cache keeps this full
        # endpoint on the same direct flattened contraction as native forward.
        explicit_full_observed = (
            observed_all
            if include_selector_payloads
            and cfg.selection_score == "isolated_loss_decrease"
            else None
        )
        full_outputs = outputs_for_view(
            x,
            observed=explicit_full_observed,
            encoder_sites=(
                None if explicit_full_observed is not None else encoder_sites
            ),
        )
        full_mode_masks = torch.stack(
            tuple(full_outputs[mode].mask for mode in selection_modes)
        )
        accumulate_target_statistics(x)
        if include_selector_payloads:
            assert selector_target_sum is not None
            assert selector_target_sumsq is not None
            selector_values = x.double()
            if selector_coordinate_mask is not None:
                selector_values = selector_values * selector_coordinate_mask
            selector_target_sum += selector_values.sum(dim=0)
            selector_target_sumsq += selector_values.square().sum(dim=0)
            del selector_values

            if cfg.selection_score == "isolated_loss_decrease":
                assert selector_candidate_gain_counts is not None
                selector_scores = full_outputs[primary_mode].scores
                assert selector_states is not None
                candidate_batch_counts = []
                selected_batch_counts = {mode: [] for mode in selection_modes}
                for sign in ("negative", "zero", "positive"):
                    if sign == "negative":
                        selector_sign = selector_scores < 0
                    elif sign == "zero":
                        selector_sign = selector_scores == 0
                    else:
                        selector_sign = selector_scores > 0
                    candidate_batch_counts.append(selector_sign.sum())
                    for mode in selection_modes:
                        selected_batch_counts[mode].append(
                            (selector_sign & full_outputs[mode].mask).sum()
                        )
                    del selector_sign
                selector_candidate_gain_counts += torch.stack(candidate_batch_counts)
                for mode in selection_modes:
                    selector_states[mode]["selected_gain_counts"] += torch.stack(
                        selected_batch_counts[mode]
                    )
                del candidate_batch_counts, selected_batch_counts
        for mode, full in full_outputs.items():
            state = states[mode]
            # Dense reconstruction SSE remains intentionally per mode.  It is
            # not interchangeable with a sparse quadratic shortcut when the
            # decoder has a bias or padded coordinates.
            assert full.xhat is not None
            full.sse = squared_error_by_site(x, full.xhat)
            state["full_sse"] += full.sse
            if include_selector_payloads:
                assert selector_states is not None
                selector_state = selector_states[mode]
                # Preserve the former selector evaluator exactly: subtract in
                # fp32, cast that residual to fp64, then reduce the complete
                # batch rather than the shared metric's bounded token chunks.
                selector_residual = (x - full.xhat).double()
                if selector_coordinate_mask is not None:
                    selector_residual = selector_residual * selector_coordinate_mask
                selector_state["error"] += selector_residual.square().sum(dim=(0, 2))
                selector_counts = full.mask.sum(dim=1)
                selector_state["support_counts"] += torch.bincount(
                    selector_counts,
                    minlength=G + 1,
                )
                del selector_residual, selector_counts, selector_state
            full.xhat = None
        full_mask_count64 = full_mode_masks.sum(dim=1).double()
        all_full_energy64 = torch.zeros(
            len(selection_modes),
            G,
            dtype=torch.float64,
            device=device,
        )
        for mode_index, state in enumerate(mode_states):
            state["fire"] += full_mask_count64[mode_index]
        raw_full_code = full_outputs[primary_mode].z
        for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
            stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
            block_slice = slice(start, stop)
            moments = _batched_mode_selected_moments(
                raw_full_code[:, block_slice],
                all_site_decoder_gram[block_slice],
                full_mode_masks[:, :, block_slice],
            )
            all_full_energy64[:, block_slice] = moments.decoded_energy
            for mode_index, state in enumerate(mode_states):
                state["zsum"][block_slice] += moments.code_sum[mode_index]
                state["zz"][block_slice] += moments.code_outer[mode_index]
            del moments
        if (
            include_selector_payloads
            and cfg.selection_score == "isolated_loss_decrease"
        ):
            del selector_scores
        zero_x: torch.Tensor | None = None
        null_outputs: dict[str, _EvaluationViewOutput] | None = None

        def run_view(
            view_observed: torch.Tensor,
            *,
            source_missing: bool,
            empty: bool,
            frozen_encoder_sites,
        ) -> dict[str, _EvaluationViewOutput]:
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

        def accumulate_coordinate_concordance_modes(
            partial_outputs: dict[str, _EvaluationViewOutput],
            index: int,
            *,
            prefix: str,
            support_intersection_key: str,
            support_union_key: str,
        ) -> torch.Tensor:
            """Accumulate every selector from one raw partial-view geometry."""

            partial_mode_masks = torch.stack(
                tuple(partial_outputs[mode].mask for mode in selection_modes)
            )
            intersections = partial_mode_masks & full_mode_masks
            intersection_totals = intersections.sum(dim=(1, 2)).double()
            union_totals = (
                (partial_mode_masks | full_mode_masks).sum(dim=(1, 2)).double()
            )
            for mode_index, state in enumerate(mode_states):
                state[support_intersection_key][index] += intersection_totals[
                    mode_index
                ]
                state[support_union_key][index] += union_totals[mode_index]
                state[f"{prefix}_full_count"][index] += full_mask_count64[mode_index]
                state[f"{prefix}_full_energy"][index] += all_full_energy64[mode_index]
            del intersections, intersection_totals, union_totals

            raw_partial_code = partial_outputs[primary_mode].z
            for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
                stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
                block_slice = slice(start, stop)
                reductions = _batched_mode_concordance(
                    raw_full_code[:, block_slice],
                    raw_partial_code[:, block_slice],
                    all_site_decoder_gram[block_slice],
                    full_mode_masks[:, :, block_slice],
                    partial_mode_masks[:, :, block_slice],
                )
                for mode_index, state in enumerate(mode_states):
                    state[f"{prefix}_intersection_count"][index, block_slice] += (
                        reductions.intersection_count[mode_index]
                    )
                    state[f"{prefix}_concordance_numerator"][index, block_slice] += (
                        reductions.concordance_numerator[mode_index]
                    )
                    state[f"{prefix}_concordance_denominator"][index, block_slice] += (
                        reductions.concordance_denominator[mode_index]
                    )
                    state[f"{prefix}_full_code_sum"][index, block_slice] += (
                        reductions.full_code_sum[mode_index]
                    )
                    state[f"{prefix}_partial_code_sum"][index, block_slice] += (
                        reductions.partial_code_sum[mode_index]
                    )
                    state[f"{prefix}_intersection_full_energy"][index, block_slice] += (
                        reductions.intersection_full_energy[mode_index]
                    )
                del reductions
            return partial_mode_masks

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
                state = states[mode]
                if only.sse is None:
                    assert only.xhat is not None
                    only.sse = squared_error_by_site(x, only.xhat)
                    only.xhat = None
                state["site_sse"][source] += only.sse
            only_mode_masks = accumulate_coordinate_concordance_modes(
                only_outputs,
                source,
                prefix="site",
                support_intersection_key="support_intersection",
                support_union_key="support_union",
            )
            del only, state, only_outputs, only_mode_masks

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
                state = states[mode]
                if missing.sse is None:
                    assert missing.xhat is not None
                    missing.sse = squared_error_by_site(x, missing.xhat)
                    missing.xhat = None
                state["loo_sse"][source] += missing.sse
            missing_mode_masks = accumulate_coordinate_concordance_modes(
                missing_outputs,
                source,
                prefix="loo",
                support_intersection_key="loo_intersection",
                support_union_key="loo_union",
            )
            raw_missing_code = missing_outputs[primary_mode].z
            for start in range(0, G, EVALUATION_CONCORDANCE_BLOCK_CHUNK):
                stop = min(start + EVALUATION_CONCORDANCE_BLOCK_CHUNK, G)
                block_slice = slice(start, stop)
                selected_delta_sq = _batched_mode_selected_delta_sq(
                    raw_full_code[:, block_slice],
                    raw_missing_code[:, block_slice],
                    full_mode_masks[:, :, block_slice],
                    missing_mode_masks[:, :, block_slice],
                )
                for mode_index, state in enumerate(mode_states):
                    state["post_selection_loo_delta_sq"][source, block_slice] += (
                        selected_delta_sq[mode_index]
                    )
                del selected_delta_sq
            del (
                missing,
                state,
                primary_missing,
                primary_full,
                missing_outputs,
                missing_mode_masks,
                raw_missing_code,
            )
        del (
            encoder_sites,
            null_outputs,
            full_outputs,
            full_mode_masks,
            full_mask_count64,
            all_full_energy64,
            raw_full_code,
        )
        n_tokens += x.shape[0]

    if n_tokens == 0:
        raise ValueError("evaluation stream produced no tokens")
    centered_ss = target_sumsq - target_sum.square() / n_tokens
    denominator = centered_ss.sum(dim=1).clamp_min(1e-30)
    selector_denominator: torch.Tensor | None = None
    if include_selector_payloads:
        assert selector_target_sum is not None
        assert selector_target_sumsq is not None
        # The coordinate mask is no longer needed after streaming. Reuse the
        # target-square accumulator for centering so the planner's three
        # target-width selector tensors remain a hard upper bound.
        del selector_coordinate_mask
        selector_target_sumsq.sub_(selector_target_sum.square().div_(n_tokens))
        selector_denominator = selector_target_sumsq.sum(dim=1).clamp_min(1e-30)

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
        ).sum(dim=-1)
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
            (state["post_selection_loo_delta_sq"] / n_tokens).clamp_min(0).sqrt()
        )
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
        used_eigenvalues = torch.zeros(S, G, b, dtype=torch.float64, device=device)
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
            used_eigenvalues[:, start:stop] = torch.linalg.eigvalsh(contribution).flip(
                -1
            )

        payload = {
            "schema_version": 5,
            "selection_mode": selection_mode,
            "n_tokens": n_tokens,
            "model_cfg": asdict(cfg),
            "full_fvu_per_site": full_fvu.tolist(),
            "full_fvu_pooled": float(state["full_sse"].sum() / denominator.sum()),
            "site_only_fvu": site_matrix.tolist(),
            "leave_one_site_out_fvu": loo_matrix.tolist(),
            "site_only_support_iou": (
                state["support_intersection"] / state["support_union"].clamp_min(1)
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

    shared_payloads = {
        mode: finalize_mode(mode, states[mode]) for mode in selection_modes
    }

    if not include_selector_payloads:
        return shared_payloads, None

    assert selector_states is not None
    assert selector_denominator is not None
    assert selector_candidate_gain_counts is not None
    gain_count_names = (
        "candidate_negative",
        "candidate_zero",
        "candidate_positive",
        "selected_negative",
        "selected_zero",
        "selected_positive",
    )
    candidate_gain_counts = selector_candidate_gain_counts.cpu().tolist()

    def finalize_selector_mode(
        selection_mode: str,
        state: dict[str, torch.Tensor],
    ) -> dict:
        error = state["error"]
        fvu = error / selector_denominator
        support_histogram = {
            count: frequency
            for count, frequency in enumerate(state["support_counts"].cpu().tolist())
            if frequency
        }
        selected_gain_counts = state["selected_gain_counts"].cpu().tolist()
        gain_counts = dict(
            zip(
                gain_count_names,
                candidate_gain_counts + selected_gain_counts,
                strict=True,
            )
        )
        event_total = sum(
            count * frequency for count, frequency in support_histogram.items()
        )
        if cfg.selection_score == "isolated_loss_decrease":
            candidate_total = sum(
                gain_counts[f"candidate_{sign}"]
                for sign in ("negative", "zero", "positive")
            )
            selected_total = sum(
                gain_counts[f"selected_{sign}"]
                for sign in ("negative", "zero", "positive")
            )
            if candidate_total != n_tokens * G:
                raise ValueError(
                    "isolated-loss candidate diagnostic count does not cover "
                    "every block"
                )
            if selected_total != event_total:
                raise ValueError(
                    "isolated-loss selected diagnostic count differs from "
                    "selector support"
                )
            isolated_loss_diagnostics: dict[str, object] = {
                "schema": "bsc-isolated-loss-gain-diagnostics-v1",
                "applicable": True,
                "observation_contract": "explicit_true_observed_sites_only_v1",
                "candidate_event_count": candidate_total,
                "candidate_negative_gain_count": gain_counts["candidate_negative"],
                "candidate_zero_gain_count": gain_counts["candidate_zero"],
                "candidate_positive_gain_count": gain_counts["candidate_positive"],
                "candidate_negative_gain_fraction": (
                    gain_counts["candidate_negative"] / candidate_total
                ),
                "candidate_zero_gain_fraction": (
                    gain_counts["candidate_zero"] / candidate_total
                ),
                "candidate_positive_gain_fraction": (
                    gain_counts["candidate_positive"] / candidate_total
                ),
                "selected_event_count": selected_total,
                "selected_negative_gain_count": gain_counts["selected_negative"],
                "selected_zero_gain_count": gain_counts["selected_zero"],
                "selected_positive_gain_count": gain_counts["selected_positive"],
                "selected_negative_gain_fraction": (
                    None
                    if selected_total == 0
                    else gain_counts["selected_negative"] / selected_total
                ),
                "selected_zero_gain_fraction": (
                    None
                    if selected_total == 0
                    else gain_counts["selected_zero"] / selected_total
                ),
                "selected_positive_gain_fraction": (
                    None
                    if selected_total == 0
                    else gain_counts["selected_positive"] / selected_total
                ),
            }
        else:
            isolated_loss_diagnostics = {
                "applicable": False,
                "reason": "selection_score_not_isolated_loss_decrease",
            }
        payload = {
            "selector": cfg.selection,
            "selection_score": cfg.selection_score,
            "mode": selection_mode,
            "n_tokens": n_tokens,
            "fvu_per_site": fvu.cpu().tolist(),
            "fvu_pooled": float(error.sum() / selector_denominator.sum()),
            "avg_active_blocks": event_total / n_tokens,
            "active_block_count_histogram": {
                str(key): support_histogram[key] for key in sorted(support_histogram)
            },
            "isolated_loss_gain_diagnostics": isolated_loss_diagnostics,
        }
        json.dumps(payload, allow_nan=False)
        return payload

    selector_payloads = {
        mode: finalize_selector_mode(mode, selector_states[mode])
        for mode in selection_modes
    }
    return shared_payloads, selector_payloads


@torch.no_grad()
def evaluate_shared_code_modes(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
    selection_modes: tuple[str, ...] = ("topk", "threshold"),
) -> dict[str, dict]:
    """Evaluate shared-code endpoints for one or more selector modes."""
    shared_code, selector = _evaluate_code_modes(
        model,
        batches,
        device=device,
        max_tokens=max_tokens,
        selection_modes=selection_modes,
        include_selector_payloads=False,
    )
    assert selector is None
    return shared_code


@torch.no_grad()
def evaluate_selector_and_shared_code_modes(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_tokens: int | None = None,
    selection_modes: tuple[str, ...] = ("topk", "threshold"),
) -> EvaluationModeEndpoints:
    """Evaluate selector summaries and shared-code endpoints in one traversal."""
    shared_code, selector = _evaluate_code_modes(
        model,
        batches,
        device=device,
        max_tokens=max_tokens,
        selection_modes=selection_modes,
        include_selector_payloads=True,
    )
    assert selector is not None
    return EvaluationModeEndpoints(selector=selector, shared_code=shared_code)


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
