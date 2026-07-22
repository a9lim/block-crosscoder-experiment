"""Trainer for the block-sparse crosscoder.

Implements the load-bearing step ordering:

    optimizer step on fp32 master -> retract master decoders ->
    regenerate bf16 forward copy -> log post-cast Gram residual

with explicitly parameterized Adam or AdamW groups, independently declared
encoder/decoder/bias decay, explicit scalar/fused arithmetic, declared warmup
and decay schedules, and AuxK dead-block machinery in explicit paper/release
and adapted variants:

    "sasa"          SASA App. C.1 — dead = windowed activation frequency
                    <= threshold; per-token top-s_aux dead blocks by
                    residual energy re-encode the frozen residual.
    "decoder_weighted_token_horizon" adapts Minder's released scalar rule — dead = no
                    selected activation for a token horizon; choose dead
                    features with decoder-weighted post-ReLU scores, but
                    decode their unscaled post-ReLU activations.
    "long_horizon"  an adapted block rule — dead = zero activations over a
                    long accepted-token horizon; same selection mechanics.
    "fel"           Fel-style runner-up AuxK — no dead set; the next
                    s_aux runner-up blocks (by main-code norm, unselected)
                    explain the residual with the *main* code;
                    alpha = 1/s_aux. This is a hybrid: Fel App. D uses the
                    next-l runner-ups with alpha = 1/l where l is the MAIN
                    block sparsity — faithful only when s_aux = k.

The data interface is any iterable of declared-coordinate [B, S, d] batches —
synthetic tensors or a normalized/raw disk store. The trainer owns
no permutation randomness: permutation seeds live with the data source.  A
declared clean-target site-masking arm owns only its torch-RNG augmentation;
that RNG and the complete augmentation configuration are checkpointed.
"""

from __future__ import annotations

import copy
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from .gram import _retract_count_tensor_, gram_residual, site_frobenius_shares
from .model import (
    BlockCrosscoder,
    BSCConfig,
    BSCOutput,
    bsc_loss,
    bsc_reconstruction_loss,
)
from .runtime_limits import (
    MODEL_IMPLEMENTATION_IDENTITY_FIELDS,
    decoded_energy_code_norm_eligible,
)

__all__ = [
    "TrainConfig",
    "DeadTracker",
    "Trainer",
    "aux_loss",
    "tensor_batches",
    "validate_optimizer_state_config",
    "validate_run_binding",
]

AUX_VARIANTS = (
    "none",
    "sasa",
    "sasa_release",
    "decoder_weighted_token_horizon",
    "long_horizon",
    "fel",
)
CHECKPOINT_FREE_FLOOR_FRAC = 0.15  # same safety floor as store.ShardWriter


def _payload_nbytes(obj) -> int:
    """Tensor bytes in a (nested) checkpoint payload, for the pre-write
    free-space check. Non-tensor leaves are negligible and counted as 0."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_payload_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_payload_nbytes(v) for v in obj)
    return 0


@dataclass
class TrainConfig:
    total_steps: int
    lr: float = 3e-4
    warmup_steps: int = 1000
    # "cosine" = linear warmup + cosine decay to 0 (current default);
    # "linear_fifth" = SASA B.3 — warmup, constant, then linear decay to 0
    # (our final-fifth reading of SASA's "over one-fifth of the training").
    # Both schedule families remain explicit matrix choices.
    schedule: str = "cosine"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    foreach: bool = False
    fused: bool = False
    encoder_weight_decay: float = 0.0
    decoder_weight_decay: float = 0.0
    bias_weight_decay: float = 0.0
    retract_every: int = 1  # source recipes and throughput arms set this explicitly
    optimizer: str = "adamw"  # adam | adamw
    forward_dtype: str = "bf16"  # "bf16" (production) | "fp32" (exact/dev)
    min_lr_ratio: float = 0.0
    final_decay_fraction: float = 0.2
    gradient_clip_norm: float | None = None
    # AuxK follows the validated SASA-style recovery path.
    aux_variant: str = "sasa"
    aux_reconstruction: str = "squared_l2"
    s_aux: int = 256
    alpha_aux: float = 1.0  # SASA lambda_aux; the Fel arm overrides to 1/s_aux
    dead_threshold: float = 1e-4
    # Deadness is a property of token exposure, not optimizer-step count.
    # The defaults correspond to 100 and 500 full 4,096-token batches while
    # remaining invariant to batch size and partial batches.
    dead_window_tokens: int = 409_600
    dead_horizon_tokens: int = 2_048_000
    # SASA's released SAELens trainer uses forward-pass—not token—age for a
    # scalar coordinate, with AuxK starting only after age > this value.
    dead_window_passes: int = 1000
    # Novel denoising arms: alter only the sites visible to the encoder while
    # retaining every true-observed clean target in L_rec.  ``bernoulli`` is
    # the independent-dropout arm; the two structured modes are deliberately
    # separate hypotheses rather than special probability values.
    encoder_site_mask_mode: str = "bernoulli"
    encoder_site_mask_probability: float = 0.0
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError(
                "warmup_steps must be nonnegative and leave at least one non-warmup step"
            )
        if self.aux_variant not in AUX_VARIANTS:
            raise ValueError(f"aux_variant must be one of {AUX_VARIANTS}")
        if self.aux_reconstruction not in {
            "squared_l2",
            "mean_l2",
            "mean_squared",
            "squared_l2_over_residual_variance",
        }:
            raise ValueError(
                "aux_reconstruction must be squared_l2, mean_l2, mean_squared, "
                "or squared_l2_over_residual_variance"
            )
        if self.forward_dtype not in ("bf16", "fp32"):
            raise ValueError("forward_dtype must be 'bf16' or 'fp32'")
        if self.schedule not in ("constant", "cosine", "linear_fifth"):
            raise ValueError("schedule must be constant, cosine, or linear_fifth")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError("unsupported optimizer")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("optimizer eps must be finite and positive")
        if self.foreach is not False:
            raise ValueError("training requires foreach=False")
        if type(self.fused) is not bool:
            raise ValueError("fused must be an exact boolean")
        if (
            min(
                self.encoder_weight_decay,
                self.decoder_weight_decay,
                self.bias_weight_decay,
            )
            < 0
        ):
            raise ValueError("weight decay values must be non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not 0.0 < self.final_decay_fraction <= 1.0:
            raise ValueError("final_decay_fraction must be in (0, 1]")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.s_aux <= 0:
            raise ValueError("s_aux must be positive")
        if self.dead_threshold < 0.0:
            raise ValueError("dead_threshold must be non-negative")
        if self.dead_window_tokens <= 0 or self.dead_horizon_tokens <= 0:
            raise ValueError("dead token windows must be positive")
        if self.dead_window_passes <= 0:
            raise ValueError("dead_window_passes must be positive")
        if self.encoder_site_mask_probability not in (0.0, 0.02, 0.05, 0.10):
            raise ValueError(
                "encoder_site_mask_probability must be one of {0.0, 0.02, 0.05, 0.10}"
            )
        if self.encoder_site_mask_mode not in {
            "bernoulli",
            "exactly_one_hidden",
            "exactly_one_retained",
        }:
            raise ValueError(
                "encoder_site_mask_mode must be bernoulli, "
                "exactly_one_hidden, or exactly_one_retained"
            )
        if (
            self.encoder_site_mask_mode != "bernoulli"
            and self.encoder_site_mask_probability != 0.0
        ):
            raise ValueError(
                "structured encoder site masking requires probability=0; "
                "the mode itself defines the intervention"
            )


def validate_run_binding(
    actual: dict | None,
    expected: dict,
    *,
    keys: Sequence[str] | None = None,
) -> None:
    """Fail closed when a checkpoint is not bound to the expected run.

    Exact resume validation compares the whole canonical binding. Consumers
    such as codec evaluation may compare only the fields relevant to their
    store/gauge view. Legacy checkpoints deliberately fail whenever an
    expected binding is supplied.
    """
    if actual is None:
        raise ValueError("checkpoint has no run binding (legacy/unbound checkpoint)")
    names = list(keys) if keys is not None else sorted(set(actual) | set(expected))
    mismatches = {
        name: {"checkpoint": actual.get(name), "expected": expected.get(name)}
        for name in names
        if actual.get(name) != expected.get(name)
    }
    if mismatches:
        raise ValueError(
            "checkpoint run binding mismatch: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


class DeadTracker:
    """Exact, policy-specific dead-feature state.

    Each scientific auxiliary owns a disjoint criterion, so a tracker retains
    and updates only that criterion's sufficient state. SASA keeps its exact
    token window in selector-specific sparse or bitpacked form; long-horizon
    rules keep one last-fire index per block; release adapters keep only their
    declared age counters.
    """

    def __init__(
        self,
        n_blocks: int,
        capacity: int,
        device,
        *,
        max_tokens: int | None = None,
        block_dim: int = 1,
        policy: str,
        selector: str | None = None,
        active_blocks: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        self.n_blocks = int(n_blocks)
        self.block_dim = int(block_dim)
        if self.block_dim <= 0:
            raise ValueError("block_dim must be positive")
        if policy not in {
            "disabled",
            "sasa",
            "long_horizon",
            "decoder_weighted_token_horizon",
            "sasa_release",
        }:
            raise ValueError(f"unknown dead-tracker policy {policy!r}")
        if policy == "sasa":
            if max_tokens == 0:
                raise ValueError("SASA dead tracking requires a nonzero token window")
            if selector not in {
                "token_topk",
                "batch_topk",
                "threshold",
                "dense",
            }:
                raise ValueError("SASA dead tracking has an unknown selector")
            if (
                not isinstance(active_blocks, (int, float))
                or isinstance(active_blocks, bool)
                or not 0 < float(active_blocks) <= n_blocks
            ):
                raise ValueError("SASA active_blocks must be in (0, n_blocks]")
        elif max_tokens != 0:
            raise ValueError("non-SASA dead tracking requires max_tokens=0")
        elif selector is not None or active_blocks is not None:
            raise ValueError("selector metadata belongs only to SASA dead tracking")
        self.capacity = int(capacity)
        self.max_tokens = max_tokens
        self.device = torch.device(device)
        self.policy = policy
        self.selector = selector
        self.active_blocks = active_blocks
        self.representation = (
            "fixed_token_indices"
            if selector == "token_topk"
            else (
                "fixed_batch_indices"
                if selector == "batch_topk"
                else ("bitpacked_masks" if policy == "sasa" else None)
            )
        )
        self._bit_shifts = (
            torch.arange(8, dtype=torch.uint8, device=self.device)
            if self.representation == "bitpacked_masks"
            else None
        )
        self.chunks: list[dict[str, torch.Tensor | int]] = []
        self._history_tokens = 0
        self._history_count = (
            torch.zeros(n_blocks, dtype=torch.int64, device=self.device)
            if policy == "sasa"
            else None
        )
        self.tokens_seen = 0
        self.last_fire = (
            torch.full((n_blocks,), -1, dtype=torch.int64, device=self.device)
            if policy == "long_horizon"
            else None
        )
        # Minder's release increments a feature-age counter by the whole
        # batch size, then resets every feature selected anywhere in that
        # batch.  This is deliberately separate from ``last_fire``: the
        # latter preserves exact within-batch token positions for the adapted
        # block horizon, while the release counter is batch-boundary exact.
        self.tokens_since_fired = (
            torch.zeros(n_blocks, dtype=torch.int64, device=self.device)
            if policy == "decoder_weighted_token_horizon"
            else None
        )
        self.coordinate_passes_since_fired = (
            torch.zeros(
                n_blocks,
                self.block_dim,
                dtype=torch.int64,
                device=self.device,
            )
            if policy == "sasa_release"
            else None
        )
        self.forward_passes = 0

    def update(
        self,
        mask: torch.Tensor,
        coordinate_activity: torch.Tensor | None = None,
    ) -> None:
        """mask: [B, G] bool from the training forward."""
        if mask.ndim != 2 or mask.shape[1] != self.n_blocks:
            raise ValueError(
                f"mask must have shape [B, {self.n_blocks}], got {tuple(mask.shape)}"
            )
        if mask.shape[0] <= 0:
            raise ValueError("dead-tracker observations must contain tokens")
        if self.policy == "disabled":
            raise RuntimeError("disabled dead tracker cannot accept observations")
        accepted = mask.detach().to(device=self.device, dtype=torch.bool)
        if self.policy == "long_horizon":
            assert self.last_fire is not None
            any_fire = accepted.any(dim=0)
            reverse_offset = accepted.flip(0).to(torch.int8).argmax(dim=0)
            positions = self.tokens_seen + len(accepted) - 1 - reverse_offset
            self.last_fire.copy_(torch.where(any_fire, positions, self.last_fire))
            self.tokens_seen += len(accepted)
            return
        if self.policy == "decoder_weighted_token_horizon":
            assert self.tokens_since_fired is not None
            any_fire = accepted.any(dim=0)
            self.tokens_since_fired += len(accepted)
            self.tokens_since_fired.masked_fill_(any_fire, 0)
            return

        # Exact SAELens rule: after each forward, increment every scalar
        # coordinate and reset those whose post-selection activation was
        # nonzero anywhere in the batch.  Signed negative values count as
        # firing because bool() is nonzero, matching the release trainer.
        if self.policy == "sasa_release":
            assert self.coordinate_passes_since_fired is not None
            self.coordinate_passes_since_fired += 1
            if coordinate_activity is None:
                coordinate_activity = accepted.unsqueeze(-1).expand(
                    -1, -1, self.block_dim
                )
            if coordinate_activity.shape != (
                mask.shape[0],
                self.n_blocks,
                self.block_dim,
            ):
                raise ValueError(
                    "coordinate_activity must have shape "
                    f"[B, {self.n_blocks}, {self.block_dim}]"
                )
            did_fire = (
                coordinate_activity.detach()
                .to(device=self.device, dtype=torch.bool)
                .any(dim=0)
            )
            self.coordinate_passes_since_fired.masked_fill_(did_fire, 0)
            self.forward_passes += 1
            return

        assert self.policy == "sasa"
        assert self._history_count is not None
        chunk = self._selected_event_chunk(accepted)
        dense_counts: torch.Tensor | None = None
        if self.representation == "bitpacked_masks":
            dense_counts = self._dense_mask_counts(accepted)
        if self.max_tokens != 0:
            if self.max_tokens is not None and len(accepted) >= self.max_tokens:
                start = len(accepted) - self.max_tokens
                if self.representation == "fixed_batch_indices":
                    chunk["start_token"] = start
                else:
                    indices = chunk["indices"]
                    assert torch.is_tensor(indices)
                    chunk = {
                        "indices": indices[start:].clone(),
                        "n_tokens": self.max_tokens,
                        "start_token": 0,
                    }
                self.chunks = [chunk]
                self._history_tokens = self.max_tokens
                self._history_count.copy_(
                    self._dense_mask_counts(accepted[-self.max_tokens :])
                    if dense_counts is not None
                    else self._chunk_counts(chunk, dtype=torch.int64)
                )
                return
            self.chunks.append(chunk)
            self._history_tokens += len(accepted)
            self._history_count += (
                dense_counts
                if dense_counts is not None
                else self._chunk_counts(chunk, dtype=torch.int64)
            )
            if self.max_tokens is None:
                while len(self.chunks) > self.capacity:
                    removed = self.chunks.pop(0)
                    retained = int(removed["n_tokens"]) - int(removed["start_token"])
                    self._history_tokens -= retained
                    self._history_count -= self._chunk_counts(
                        removed,
                        dtype=torch.int64,
                    )
            else:
                excess = self.history_tokens - self.max_tokens
                while excess > 0 and self.chunks:
                    oldest = self.chunks[0]
                    start = int(oldest["start_token"])
                    retained = int(oldest["n_tokens"]) - start
                    if excess >= retained:
                        excess -= retained
                        removed = self.chunks.pop(0)
                        self._history_tokens -= retained
                        self._history_count -= self._chunk_counts(
                            removed,
                            dtype=torch.int64,
                        )
                    else:
                        self._history_count -= self._chunk_counts(
                            oldest,
                            dtype=torch.int64,
                            row_start=start,
                            row_stop=start + excess,
                        )
                        new_start = start + excess
                        if self.representation == "fixed_batch_indices":
                            oldest["start_token"] = new_start
                        else:
                            indices = oldest["indices"]
                            assert torch.is_tensor(indices)
                            oldest["indices"] = indices[new_start:].clone()
                            oldest["n_tokens"] = retained - excess
                            oldest["start_token"] = 0
                        self._history_tokens -= excess
                        excess = 0

    def _selected_event_chunk(
        self,
        accepted: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        assert self.representation in {
            "fixed_token_indices",
            "fixed_batch_indices",
            "bitpacked_masks",
        }
        assert self.active_blocks is not None
        byte_mask = accepted.contiguous().view(torch.uint8)
        if self.representation == "fixed_token_indices":
            n_keep = int(round(self.active_blocks))
            indices = byte_mask.topk(
                n_keep,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            indices = indices.sort(dim=1).values.to(torch.int32)
        elif self.representation == "fixed_batch_indices":
            if accepted.numel() > torch.iinfo(torch.int32).max:
                raise ValueError("batch-topk tracker indices exceed int32 range")
            n_keep = min(
                max(int(round(len(accepted) * self.active_blocks)), 0),
                accepted.numel(),
            )
            indices = (
                byte_mask.reshape(-1)
                .topk(
                    n_keep,
                    largest=True,
                    sorted=False,
                )
                .indices
            )
            indices = indices.sort().values.to(torch.int32)
        else:
            indices = byte_mask[:, 0::8].clone()
            for shift in range(1, 8):
                source = byte_mask[:, shift::8]
                if source.shape[1]:
                    indices[:, : source.shape[1]].bitwise_or_(source << shift)
        return {
            "indices": indices,
            "n_tokens": len(accepted),
            "start_token": 0,
        }

    def _chunk_counts(
        self,
        chunk: dict[str, torch.Tensor | int],
        *,
        dtype: torch.dtype,
        row_start: int | None = None,
        row_stop: int | None = None,
    ) -> torch.Tensor:
        indices = chunk["indices"]
        assert torch.is_tensor(indices)
        n_tokens = int(chunk["n_tokens"])
        start = int(chunk["start_token"]) if row_start is None else row_start
        stop = n_tokens if row_stop is None else row_stop
        counts = torch.zeros(self.n_blocks, dtype=dtype, device=self.device)
        if self.representation == "fixed_token_indices":
            blocks = (
                indices[start:stop]
                .reshape(-1)
                .to(
                    device=self.device,
                    dtype=torch.int64,
                )
            )
            weights = torch.ones(blocks.shape, dtype=dtype, device=self.device)
        elif self.representation == "fixed_batch_indices":
            flat = indices.to(device=self.device, dtype=torch.int64)
            rows = torch.div(flat, self.n_blocks, rounding_mode="floor")
            blocks = flat.remainder(self.n_blocks)
            weights = ((rows >= start) & (rows < stop)).to(dtype)
        else:
            assert self._bit_shifts is not None
            packed = indices[start:stop].to(device=self.device)
            counts = torch.zeros(
                self.n_blocks,
                dtype=dtype,
                device=self.device,
            )
            for rows in packed.split(512):
                unpacked = torch.bitwise_right_shift(
                    rows.unsqueeze(-1),
                    self._bit_shifts,
                ).bitwise_and_(1)
                unpacked = unpacked.reshape(len(rows), -1)[:, : self.n_blocks]
                counts.add_(unpacked.sum(dim=0, dtype=torch.int32))
            return counts
        counts.scatter_add_(
            0,
            blocks,
            weights,
        )
        return counts

    def _dense_mask_counts(self, mask: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(
            self.n_blocks,
            dtype=torch.int64,
            device=self.device,
        )
        for rows in mask.split(512):
            counts.add_(rows.sum(dim=0, dtype=torch.int32))
        return counts

    @property
    def history_tokens(self) -> int:
        return self._history_tokens

    def frequency(self, window_tokens: int) -> torch.Tensor:
        """Per-block frequency over the last ``window_tokens`` accepted tokens.

        Sparse fixed-cardinality or bitpacked variable-support chunks are
        sliced at the exact token boundary, so batch size and partial final
        batches do not change the criterion.
        """
        if self.policy != "sasa":
            raise RuntimeError("frequency is available only for the SASA policy")
        if window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        if not self.chunks:
            return torch.zeros(self.n_blocks, device=self.device)
        history_tokens = self.history_tokens
        if window_tokens >= history_tokens:
            assert self._history_count is not None
            return self._history_count.float() / max(1, history_tokens)
        remaining = window_tokens
        total = torch.zeros(
            self.n_blocks,
            dtype=torch.int64,
            device=self.device,
        )
        for chunk in reversed(self.chunks):
            n_tokens = int(chunk["n_tokens"])
            retained = n_tokens - int(chunk["start_token"])
            take = min(remaining, retained)
            total += self._chunk_counts(
                chunk,
                dtype=torch.int64,
                row_start=n_tokens - take,
                row_stop=n_tokens,
            )
            remaining -= take
            if remaining == 0:
                break
        return total.float() / window_tokens

    def dead(
        self,
        variant: str,
        *,
        threshold: float,
        window_tokens: int,
        horizon_tokens: int,
    ) -> torch.Tensor:
        """Bool [G]. All-False until the token-denominated history is full."""
        if variant != self.policy or variant not in {"sasa", "long_horizon"}:
            raise ValueError(
                f"dead criterion {variant!r} is unavailable for {self.policy!r}"
            )
        G = self.n_blocks
        device = self.device
        if variant == "sasa":
            if self.history_tokens < window_tokens:
                return torch.zeros(G, dtype=torch.bool, device=device)
            return self.frequency(window_tokens) <= threshold
        if variant == "long_horizon":
            assert self.last_fire is not None
            if self.tokens_seen < horizon_tokens:
                return torch.zeros(G, dtype=torch.bool, device=device)
            return self.last_fire < self.tokens_seen - horizon_tokens
        raise ValueError(f"no dead criterion for variant {variant!r}")

    def dead_coordinates(self, window_passes: int) -> torch.Tensor:
        """SAELens scalar dead mask, shape [G,b], evaluated before a forward."""
        if self.policy != "sasa_release":
            raise RuntimeError(
                "coordinate deadness is available only for the SASA-release policy"
            )
        if window_passes <= 0:
            raise ValueError("window_passes must be positive")
        assert self.coordinate_passes_since_fired is not None
        return self.coordinate_passes_since_fired > window_passes

    def token_horizon_dead_after_current(
        self,
        current_mask: torch.Tensor,
        horizon_tokens: int,
    ) -> torch.Tensor:
        """Token-horizon dead mask evaluated after the current forward.

        The pinned trainer increments all feature ages by the current batch
        size and resets every feature selected at least once before computing
        AuxK.  The real tracker is mutated only after an accepted optimizer
        step, so guarded/skipped batches cannot contaminate resume state.
        """
        if self.policy != "decoder_weighted_token_horizon":
            raise RuntimeError(
                "token-age deadness is available only for the Minder policy"
            )
        if horizon_tokens <= 0:
            raise ValueError("horizon_tokens must be positive")
        if current_mask.ndim != 2 or current_mask.shape[1] != self.n_blocks:
            raise ValueError(f"current_mask must have shape [B, {self.n_blocks}]")
        assert self.tokens_since_fired is not None
        projected = self.tokens_since_fired + int(current_mask.shape[0])
        projected = projected.clone()
        projected.masked_fill_(
            current_mask.detach().to(self.device).any(dim=0),
            0,
        )
        return projected >= horizon_tokens

    def state_dict(self) -> dict:
        state = {
            "capacity": self.capacity,
            "max_tokens": self.max_tokens,
            "block_dim": self.block_dim,
            "policy": self.policy,
        }
        if self.policy == "sasa":
            state["selector"] = self.selector
            state["active_blocks"] = self.active_blocks
            state["representation"] = self.representation
            state["chunks"] = self.chunks
        elif self.policy == "long_horizon":
            state["tokens_seen"] = self.tokens_seen
            state["last_fire"] = self.last_fire
        elif self.policy == "decoder_weighted_token_horizon":
            state["tokens_since_fired"] = self.tokens_since_fired
        elif self.policy == "sasa_release":
            state["coordinate_passes_since_fired"] = self.coordinate_passes_since_fired
            state["forward_passes"] = self.forward_passes
        return state

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("dead-tracker state must be a mapping")
        if state.get("policy") != self.policy:
            raise ValueError("dead-tracker policy changed across resume")
        expected_keys = set(self.state_dict())
        if set(state) != expected_keys:
            raise ValueError("dead-tracker state keys do not match its policy")
        capacity = state["capacity"]
        block_dim = state["block_dim"]
        max_tokens = state["max_tokens"]
        max_tokens_valid = (
            max_tokens is None
            if self.max_tokens is None
            else type(max_tokens) is int and max_tokens == self.max_tokens
        )
        if (
            type(capacity) is not int
            or capacity != self.capacity
            or type(block_dim) is not int
            or block_dim != self.block_dim
            or not max_tokens_valid
        ):
            raise ValueError("dead-tracker configuration changed across resume")

        def require_tensor(
            name: str,
            shape: tuple[int, ...],
            dtype: torch.dtype,
        ) -> torch.Tensor:
            value = state[name]
            if (
                not torch.is_tensor(value)
                or value.shape != shape
                or value.dtype != dtype
            ):
                raise ValueError(
                    f"dead-tracker {name} must have shape {shape} and dtype {dtype}"
                )
            return value

        if self.policy == "sasa":
            if (
                state["selector"] != self.selector
                or type(state["active_blocks"]) is not type(self.active_blocks)
                or state["active_blocks"] != self.active_blocks
                or state["representation"] != self.representation
            ):
                raise ValueError("dead-tracker selector changed across resume")
            assert self.representation in {
                "fixed_token_indices",
                "fixed_batch_indices",
                "bitpacked_masks",
            }
            assert self.active_blocks is not None
            chunks = state["chunks"]
            if not isinstance(chunks, list):
                raise ValueError("dead-tracker chunks must be a list")
            for chunk in chunks:
                if not isinstance(chunk, dict) or set(chunk) != {
                    "indices",
                    "n_tokens",
                    "start_token",
                }:
                    raise ValueError("dead-tracker chunk fields are malformed")
                n_tokens = chunk["n_tokens"]
                start_token = chunk["start_token"]
                if (
                    not isinstance(n_tokens, int)
                    or isinstance(n_tokens, bool)
                    or n_tokens <= 0
                    or not isinstance(start_token, int)
                    or isinstance(start_token, bool)
                    or not 0 <= start_token < n_tokens
                ):
                    raise ValueError("dead-tracker chunk token bounds are malformed")
                if self.representation != "fixed_batch_indices" and start_token != 0:
                    raise ValueError(
                        "compact dead-tracker chunks cannot retain a stale prefix"
                    )
                indices = chunk["indices"]
                if not torch.is_tensor(indices) or indices.layout != torch.strided:
                    raise ValueError(
                        "dead-tracker chunk indices must be dense strided tensors"
                    )
                if self.representation == "fixed_token_indices":
                    if indices.dtype != torch.int32:
                        raise ValueError("dead-tracker chunk indices must be int32")
                    width = int(round(self.active_blocks))
                    if indices.shape != (n_tokens, width):
                        raise ValueError(
                            "token-topk tracker indices have the wrong shape"
                        )
                    out_of_range = bool(
                        ((indices < 0) | (indices >= self.n_blocks)).any()
                    )
                    unordered = width > 1 and bool(
                        (indices[:, 1:] <= indices[:, :-1]).any()
                    )
                    if out_of_range or unordered:
                        raise ValueError("token-topk tracker indices are not canonical")
                elif self.representation == "fixed_batch_indices":
                    if indices.dtype != torch.int32:
                        raise ValueError("dead-tracker chunk indices must be int32")
                    expected = min(
                        max(int(round(n_tokens * self.active_blocks)), 0),
                        n_tokens * self.n_blocks,
                    )
                    if indices.shape != (expected,):
                        raise ValueError(
                            "batch-topk tracker indices have the wrong shape"
                        )
                    out_of_range = bool(
                        ((indices < 0) | (indices >= n_tokens * self.n_blocks)).any()
                    )
                    unordered = len(indices) > 1 and bool(
                        (indices[1:] <= indices[:-1]).any()
                    )
                    if out_of_range or unordered:
                        raise ValueError("batch-topk tracker indices are not canonical")
                else:
                    packed_width = (self.n_blocks + 7) // 8
                    if indices.dtype != torch.uint8 or indices.shape != (
                        n_tokens,
                        packed_width,
                    ):
                        raise ValueError(
                            "bitpacked tracker masks have the wrong shape or dtype"
                        )
                    padding = -self.n_blocks % 8
                    if padding and bool(
                        (
                            indices[:, -1]
                            >> torch.tensor(
                                8 - padding,
                                dtype=torch.uint8,
                                device=indices.device,
                            )
                        ).any()
                    ):
                        raise ValueError("bitpacked tracker padding is nonzero")
            history_tokens = sum(
                int(chunk["n_tokens"]) - int(chunk["start_token"]) for chunk in chunks
            )
            if (self.max_tokens is not None and history_tokens > self.max_tokens) or (
                self.max_tokens is None and len(chunks) > self.capacity
            ):
                raise ValueError("dead-tracker chunks exceed their retention bound")
            self.chunks = [
                {
                    "indices": chunk["indices"].to(device=self.device).clone(),
                    "n_tokens": int(chunk["n_tokens"]),
                    "start_token": int(chunk["start_token"]),
                }
                for chunk in chunks
            ]
            self._history_tokens = history_tokens
            assert self._history_count is not None
            self._history_count.zero_()
            for chunk in self.chunks:
                self._history_count += self._chunk_counts(
                    chunk,
                    dtype=torch.int64,
                )
        elif self.policy == "long_horizon":
            tokens_seen = state["tokens_seen"]
            if (
                not isinstance(tokens_seen, int)
                or isinstance(tokens_seen, bool)
                or tokens_seen < 0
            ):
                raise ValueError("dead-tracker tokens_seen must be non-negative")
            last_fire = require_tensor(
                "last_fire",
                (self.n_blocks,),
                torch.int64,
            )
            if bool(((last_fire < -1) | (last_fire >= tokens_seen)).any()):
                raise ValueError("dead-tracker last_fire is outside the token history")
            self.tokens_seen = tokens_seen
            assert self.last_fire is not None
            self.last_fire.copy_(last_fire.to(device=self.device))
        elif self.policy == "decoder_weighted_token_horizon":
            assert self.tokens_since_fired is not None
            tokens_since_fired = require_tensor(
                "tokens_since_fired",
                (self.n_blocks,),
                torch.int64,
            )
            if bool((tokens_since_fired < 0).any()):
                raise ValueError("dead-tracker token ages must be non-negative")
            self.tokens_since_fired.copy_(tokens_since_fired.to(device=self.device))
        elif self.policy == "sasa_release":
            assert self.coordinate_passes_since_fired is not None
            coordinate_ages = require_tensor(
                "coordinate_passes_since_fired",
                (self.n_blocks, self.block_dim),
                torch.int64,
            )
            forward_passes = state["forward_passes"]
            if (
                bool((coordinate_ages < 0).any())
                or not isinstance(forward_passes, int)
                or isinstance(forward_passes, bool)
                or forward_passes < 0
                or bool((coordinate_ages > forward_passes).any())
            ):
                raise ValueError("dead-tracker pass ages must be non-negative")
            self.coordinate_passes_since_fired.copy_(
                coordinate_ages.to(device=self.device)
            )
            self.forward_passes = forward_passes


def aux_loss(
    model: BlockCrosscoder,
    x: torch.Tensor,
    out: BSCOutput,
    variant: str,
    dead: torch.Tensor | None,
    s_aux: int,
    observation_mask: torch.Tensor | None = None,
    encoder_observed: torch.Tensor | None = None,
    reconstruction_loss: str = "squared_l2",
) -> torch.Tensor | None:
    """L_aux under the same declared fp32 reduction as L_rec.

    The residual is frozen (no gradient through it) in every variant.
    Returns None when the variant has nothing to train on this step.
    """
    B, G = out.scores.shape
    all_observed = observation_mask is None
    site_mask = (
        None
        if all_observed
        else observation_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    )

    def reconstruction(
        error: torch.Tensor,
        residual_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coord = (
            model.coordinate_mask[:, 0, 0].to(error.device)
            if model._has_padded_coordinates
            else None
        )
        masked = error.float()
        if coord is not None:
            masked = masked * coord
        if site_mask is not None:
            masked = masked * site_mask
        if reconstruction_loss == "mean_l2":
            denominator = (
                B * model.cfg.n_sites
                if all_observed
                else site_mask.squeeze(-1).sum().clamp_min(1.0)
            )
            return masked.norm(dim=-1).sum() / denominator
        if reconstruction_loss == "mean_squared":
            if all_observed:
                denominator = B * sum(model.cfg.site_dims)
            elif coord is None:
                denominator = (site_mask.sum() * model.cfg.d_model).clamp_min(1.0)
            else:
                denominator = (coord * site_mask).sum().clamp_min(1.0)
            return masked.pow(2).sum() / denominator
        if reconstruction_loss != "squared_l2":
            if reconstruction_loss != "squared_l2_over_residual_variance":
                raise ValueError(
                    f"unknown auxiliary reconstruction {reconstruction_loss!r}"
                )
            if residual_target is None:
                raise ValueError(
                    "normalized auxiliary loss requires the residual target"
                )
            target = residual_target.float()
            if coord is not None:
                target = target * coord
            if site_mask is not None:
                target = target * site_mask
                observation_count = site_mask.sum(dim=0).clamp_min(1.0)
            else:
                observation_count = B
            observed_values = target.sum(dim=0) / observation_count
            centered = target - observed_values.unsqueeze(0)
            if site_mask is not None:
                centered = centered * site_mask
            residual_variance = centered.pow(2).sum() / B
            return (masked.pow(2).sum() / B) / residual_variance.clamp_min(1e-30)
        return masked.pow(2).sum() / B

    if variant == "fel":
        # Runner-up blocks by main-code norm among the unselected; the main
        # code (not a re-encoding) explains what the selected blocks missed.
        n_unselected = int(G - out.mask.sum(dim=1).max().item())
        keep = min(s_aux, n_unselected)
        if keep <= 0:
            return None
        p = out.scores.masked_fill(out.mask, float("-inf"))
        z_aux = out.z
    elif variant == "sasa_release":
        # The inspected SASA release applies a scalar-coordinate AuxK to the
        # original signed preactivations.  A selected signed group activates
        # every coordinate almost surely, so the group dead mask expands to
        # coordinates here; unlike paper SASA, the auxiliary may retain only
        # part of a block and includes the decoder bias.
        assert dead is not None
        if dead.shape != (G, model.cfg.block_dim):
            raise ValueError(
                "sasa_release requires a scalar-coordinate dead mask with "
                f"shape {(G, model.cfg.block_dim)}"
            )
        dead_coordinates = dead.reshape(-1)
        n_dead = int(dead_coordinates.sum().item())
        keep = min(s_aux, n_dead, sum(model.cfg.site_dims) // 2)
        if keep <= 0:
            return None
        flat = out.z.reshape(B, -1)
        masked = flat.masked_fill(~dead_coordinates.view(1, -1), float("-inf"))
        top = masked.topk(keep, dim=1, sorted=False).indices
        aux_flat = torch.zeros_like(flat)
        aux_flat.scatter_(1, top, masked.gather(1, top))
        residual = (x - out.xhat).detach()
        rhat = model.decode(aux_flat.view_as(out.z), add_bias=True)
        return reconstruction(rhat - residual, residual)
    elif variant == "decoder_weighted_token_horizon":
        # Cross-layer adaptation of the pinned Minder AuxK mechanism. The
        # release ranks dead scalar
        # features by decoder-weighted post-ReLU activation, detaches only
        # the chosen indices, scatters the corresponding *unscaled* ReLU
        # activations, and decodes without the bias.  BSCOutput.scores and
        # BSCOutput.z are those two tensors for the scalar (block_dim=1),
        # decoder_weighted adapter.
        assert dead is not None
        if model.cfg.block_dim != 1:
            raise ValueError("decoder_weighted_token_horizon AuxK requires block_dim=1")
        if model.cfg.selection_score != "decoder_weighted":
            raise ValueError(
                "decoder_weighted_token_horizon AuxK requires decoder_weighted selection scores"
            )
        if dead.shape != (G,):
            raise ValueError(
                f"decoder_weighted_token_horizon dead mask must have shape {(G,)}"
            )
        n_dead = int(dead.sum().item())
        keep = min(s_aux, n_dead)
        if keep <= 0:
            return None
        ranked = out.scores.masked_fill(~dead.view(1, -1), float("-inf")).detach()
        top = ranked.topk(keep, dim=1, sorted=False).indices
        flat = out.z.squeeze(-1)
        aux_flat = torch.zeros_like(flat)
        aux_flat.scatter_(1, top, flat.gather(1, top))
        residual = (x - out.xhat).detach()
        rhat = model.decode(aux_flat.unsqueeze(-1), add_bias=False)
        value = reconstruction(rhat - residual, residual)
        return value.nan_to_num(0.0)
    else:
        # SASA C.1 / long-horizon: re-encode the frozen residual through
        # dead blocks only; top s_aux dead blocks by residual energy.
        assert dead is not None
        n_dead = int(dead.sum().item())
        keep = min(s_aux, n_dead)
        if keep <= 0:
            return None
    residual = (x - out.xhat).detach()
    if variant != "fel":
        z_aux = model.encode(residual, observed=encoder_observed) * dead.view(1, -1, 1)
        p = model.scores(
            z_aux,
            x=residual,
            observed=encoder_observed,
        ).masked_fill(~dead.view(1, -1), float("-inf"))

    top = p.topk(keep, dim=1, sorted=False).indices
    mask = torch.zeros(B, G, dtype=torch.bool, device=p.device)
    mask.scatter_(1, top, True)
    rhat = model.decode(z_aux * mask.unsqueeze(-1), add_bias=False)
    return reconstruction(rhat - residual, residual)


def build_optimizer(
    model: BlockCrosscoder, cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, str]:
    """Build explicitly resolved paper or engineering optimizer groups."""
    kind = cfg.optimizer
    parameters = tuple(model.parameters())
    if cfg.fused and any(
        parameter.device.type != "cuda" or parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise ValueError("fused optimizer requires fp32 CUDA master parameters")
    decoder_names = {"D", "D_site", "D_core"}
    bias_names = {"a", "c", "log_threshold"}
    grouped: dict[float, list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if name == "c" and not model.cfg.decoder_bias:
            continue
        if name in decoder_names:
            wd = cfg.decoder_weight_decay
        elif name in bias_names:
            wd = cfg.bias_weight_decay
        else:
            wd = cfg.encoder_weight_decay
        grouped.setdefault(wd, []).append(param)
    groups = [
        {"params": params, "weight_decay": wd}
        for wd, params in sorted(grouped.items())
        if params
    ]
    if kind == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            foreach=cfg.foreach,
            fused=cfg.fused,
        ), kind
    if kind == "adam":
        # Adam has no decoupled weight decay. Paper recipes that specify Adam
        # must leave all decay fields zero; reject silent AdamW emulation.
        if any(group["weight_decay"] for group in groups):
            raise ValueError("Adam recipes cannot request AdamW weight decay")
        return torch.optim.Adam(
            groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            foreach=cfg.foreach,
            fused=cfg.fused,
        ), kind
    raise ValueError(f"unknown optimizer {kind!r}")


_OPTIMIZER_IMMUTABLE_GROUP_FIELDS = (
    "foreach",
    "fused",
    "betas",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "capturable",
    "differentiable",
)


def _optimizer_group_contract(
    optimizer_state: object,
    optimizer_kind: object,
) -> tuple[object, ...]:
    """Canonical immutable optimizer-kernel and hyperparameter contract."""

    if optimizer_kind not in {"adam", "adamw"}:
        raise ValueError("checkpoint has an invalid optimizer kind")
    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint lacks optimizer state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("checkpoint lacks optimizer parameter groups")
    result: list[tuple[object, ...]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"optimizer parameter group {index} is malformed")
        missing = set(_OPTIMIZER_IMMUTABLE_GROUP_FIELDS).difference(group)
        params = group.get("params")
        if missing or not isinstance(params, list) or not params:
            raise ValueError(f"optimizer parameter group {index} is incomplete")
        result.append(
            (
                len(params),
                *(
                    tuple(group[field]) if field == "betas" else group[field]
                    for field in _OPTIMIZER_IMMUTABLE_GROUP_FIELDS
                ),
            )
        )
    return (optimizer_kind, tuple(result))


def validate_optimizer_state_config(
    optimizer_state: object,
    cfg: TrainConfig,
    optimizer_kind: object,
) -> tuple[object, ...]:
    """Fail closed when serialized groups disagree with the train config."""

    contract = _optimizer_group_contract(optimizer_state, optimizer_kind)
    if optimizer_kind != cfg.optimizer:
        raise ValueError("optimizer kind disagrees with the train config")
    groups = optimizer_state["param_groups"]
    allowed_weight_decay = {
        cfg.encoder_weight_decay,
        cfg.decoder_weight_decay,
        cfg.bias_weight_decay,
    }
    weight_decays: list[float] = []
    for index, group in enumerate(groups):
        if group["foreach"] is not cfg.foreach:
            raise ValueError(f"optimizer group {index} changes foreach identity")
        if group["fused"] is not cfg.fused:
            raise ValueError(f"optimizer group {index} changes fused identity")
        if tuple(group["betas"]) != cfg.betas:
            raise ValueError(f"optimizer group {index} changes betas")
        if group["eps"] != cfg.eps:
            raise ValueError(f"optimizer group {index} changes epsilon")
        weight_decay = float(group["weight_decay"])
        if weight_decay not in allowed_weight_decay:
            raise ValueError(f"optimizer group {index} changes weight decay")
        weight_decays.append(weight_decay)
    if weight_decays != sorted(set(weight_decays)):
        raise ValueError("optimizer weight-decay groups are not canonical")
    if cfg.optimizer == "adam" and any(weight_decays):
        raise ValueError("Adam optimizer state contains decoupled weight decay")
    return contract


def _validate_optimizer_state_shapes(
    optimizer_state: object,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Refuse positional optimizer state whose tensors do not fit parameters.

    PyTorch accepts mismatched Adam moment shapes during ``load_state_dict``
    and fails only at the next step. Check the serialized positional mapping
    against the freshly constructed optimizer before mutation so a stale
    factor-core layout cannot become an apparently resumed campaign.
    """

    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint lacks optimizer state")
    stored_groups = optimizer_state.get("param_groups")
    stored_states = optimizer_state.get("state")
    if not isinstance(stored_groups, list) or not isinstance(stored_states, dict):
        raise ValueError("checkpoint optimizer state is malformed")
    if len(stored_groups) != len(optimizer.param_groups):
        raise ValueError("checkpoint optimizer group count mismatch")
    for group_index, (stored_group, live_group) in enumerate(
        zip(stored_groups, optimizer.param_groups, strict=True)
    ):
        stored_parameters = stored_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(stored_parameters, list) or not isinstance(
            live_parameters, list
        ):
            raise ValueError(f"checkpoint optimizer group {group_index} is malformed")
        if len(stored_parameters) != len(live_parameters):
            raise ValueError(
                f"checkpoint optimizer group {group_index} parameter count mismatch"
            )
        for parameter_index, (state_id, parameter) in enumerate(
            zip(stored_parameters, live_parameters, strict=True)
        ):
            state = stored_states.get(state_id, {})
            if not isinstance(state, dict):
                raise ValueError(
                    "checkpoint optimizer parameter state is malformed at "
                    f"group {group_index}, position {parameter_index}"
                )
            for state_name, tensor in state.items():
                if not torch.is_tensor(tensor):
                    continue
                if state_name == "step" and tensor.ndim == 0:
                    continue
                if tensor.shape != parameter.shape:
                    raise ValueError(
                        f"checkpoint optimizer {state_name} shape {tuple(tensor.shape)} "
                        f"does not match parameter shape {tuple(parameter.shape)} at "
                        f"group {group_index}, position {parameter_index}"
                    )
                if tensor.dtype != parameter.dtype:
                    raise ValueError(
                        f"checkpoint optimizer {state_name} dtype {tensor.dtype} "
                        f"does not match parameter dtype {parameter.dtype} at "
                        f"group {group_index}, position {parameter_index}"
                    )


def _lr_factor(cfg: TrainConfig):
    def factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        if cfg.schedule == "constant":
            return 1.0
        if cfg.schedule == "linear_fifth":
            # SASA B.3: constant after warmup, linear decay over the
            # configured final fraction.
            decay_start = int(cfg.total_steps * (1.0 - cfg.final_decay_fraction))
            if step < decay_start:
                return 1.0
            span = max(1, cfg.total_steps - decay_start - 1)
            raw = max(0.0, 1.0 - (step - decay_start) / span)
            return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * raw
        span = max(1, cfg.total_steps - cfg.warmup_steps - 1)
        progress = min(1.0, (step - cfg.warmup_steps) / span)
        raw = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * raw

    return factor


def tensor_batches(
    x: torch.Tensor, batch_size: int, *, seed: int = 0, epochs: int | None = None
) -> Iterator[torch.Tensor]:
    """Shuffled minibatches from an in-memory [N, S, d] tensor, reshuffled
    each epoch. The seed is the caller's to record (design: the permutation
    seed is shared by BSC and baseline runs)."""
    n = x.shape[0]
    gen = torch.Generator().manual_seed(seed)
    epoch = 0
    while epochs is None or epoch < epochs:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - batch_size + 1, batch_size):
            yield x[perm[i : i + batch_size]]
        epoch += 1


def _floating_tensors(obj) -> Iterator[torch.Tensor]:
    if torch.is_tensor(obj):
        if obj.is_floating_point() or obj.is_complex():
            yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _floating_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _floating_tensors(value)


@torch.no_grad()
def _all_finite(obj) -> bool:
    tensors = [tensor for tensor in _floating_tensors(obj) if tensor.numel()]
    if not tensors:
        return True
    # Infinity norms are finite exactly when every element is finite, but the
    # foreach implementation scans each parameter/state list through a small
    # number of multi-tensor kernels instead of launching isfinite+reduction
    # separately for every tensor.  Grouping by device preserves generic CPU
    # use while production takes one host read for its single CUDA device.
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in tensors:
        by_device.setdefault(tensor.device, []).append(tensor)
    for device_tensors in by_device.values():
        norms = torch._foreach_norm(device_tensors, ord=float("inf"))
        if not bool(torch.isfinite(torch.stack(norms)).all()):
            return False
    return True


@torch.no_grad()
def _finite_gradients_with_l2_guard(
    rec: torch.Tensor,
    total: torch.Tensor,
    gradients: list[torch.Tensor],
) -> bool:
    """Match the historical finite-L2 refusal without a routine L2 scan.

    Infinity norms scan every gradient for non-finite elements.  When their
    maximum is below the dimension-aware bound, the global fp32 L2 norm cannot
    overflow.  Only the pathological high-magnitude branch executes the exact
    historical per-tensor/global L2 reductions to preserve its refusal set.
    """
    infinity_norms = torch._foreach_norm(gradients, ord=float("inf"))
    health = torch.stack(
        (rec.detach().float(), total.detach().float(), *infinity_norms)
    )
    basic_finite = torch.isfinite(health).all()
    total_elements = sum(gradient.numel() for gradient in gradients)
    safe_l2_limit = math.sqrt(torch.finfo(torch.float32).max / total_elements)
    safely_bounded = torch.stack(infinity_norms).max() < safe_l2_limit
    if bool(basic_finite & safely_bounded):
        return True
    if not bool(basic_finite):
        return False
    per_tensor_norms = [
        torch.linalg.vector_norm(gradient.float()) for gradient in gradients
    ]
    historical_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms))
    return bool(torch.isfinite(historical_norm))


def _project_decoder_(
    model: BlockCrosscoder,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    project_with_state = getattr(model, "_project_decoder_with_state_", None)
    if project_with_state is not None:
        return project_with_state()
    project = getattr(model, "project_decoder_", None)
    if project is not None:
        result = project()
        count = torch.tensor(
            0 if result is None else int(result),
            dtype=torch.int64,
            device=next(model.parameters()).device,
        )
        return count, tuple(model.parameters())
    count = _retract_count_tensor_(model.D.data, eig_floor=model.cfg.eig_floor)
    return count, (model.D,)


def _constraint_residual(model: BlockCrosscoder) -> float | None:
    measure = getattr(model, "decoder_constraint_residual", None)
    if measure is not None:
        value = measure()
        return float(value.max() if torch.is_tensor(value) else value)
    constraint = getattr(model.cfg, "decoder_constraint", "gram")
    if constraint in {"gram", "qr"}:
        return float(gram_residual(model.decoder_tensor().float()).max())
    return None


class Trainer:
    """Owns the master/forward-copy pair, the optimizer, the retraction
    schedule, dead-block tracking, and diagnostics logging."""

    def __init__(
        self,
        model: BlockCrosscoder,
        cfg: TrainConfig,
        *,
        log_path: str | Path | None = None,
        run_binding: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_binding = copy.deepcopy(run_binding)
        self.master = model  # fp32 masters
        if model.uses_stiefel_code_norm_decoded_energy and not (
            decoded_energy_code_norm_eligible(
                selection_score=model.cfg.selection_score,
                decoder_constraint=model.cfg.decoder_constraint,
                training_selector=model.cfg.selection,
                site_rank=model.cfg.site_rank,
                retract_every=cfg.retract_every,
            )
        ):
            raise ValueError(
                "stiefel code-norm decoded energy requires decoder retraction "
                "after every optimizer step"
            )
        masking_enabled = (
            cfg.encoder_site_mask_mode != "bernoulli"
            or cfg.encoder_site_mask_probability > 0
        )
        if masking_enabled and model.cfg.encoder_fusion == "source":
            raise ValueError(
                "clean-target site masking is incompatible with source-only "
                "encoder fusion"
            )
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise TypeError("master model must be fp32")
        if cfg.forward_dtype == "bf16":
            self.fwd = copy.deepcopy(model).to(torch.bfloat16)
            for p in self.master.parameters():
                p.requires_grad_(False)
        else:
            self.fwd = model
        # Fail before optimizer state exists when either the fp32 carrier or
        # its regenerated bf16 forward copy lies outside the bound under which
        # decoded energy is specialized to code norm.
        self.master.validate_decoded_energy_implementation()
        if self.fwd is not self.master:
            self.fwd.validate_decoded_energy_implementation()
        self.opt, self.optimizer_kind = build_optimizer(self.master, cfg)
        if self.fwd is not self.master:
            master_manifest = tuple(
                (name, tuple(parameter.shape))
                for name, parameter in self.master.named_parameters()
            )
            forward_manifest = tuple(
                (name, tuple(parameter.shape))
                for name, parameter in self.fwd.named_parameters()
            )
            if forward_manifest != master_manifest:
                raise RuntimeError("master and forward parameter manifests diverged")
        self._optimizer_contract = validate_optimizer_state_config(
            self.opt.state_dict(),
            cfg,
            self.optimizer_kind,
        )
        self.sched = LambdaLR(self.opt, _lr_factor(cfg))
        tracker_selector: str | None = None
        tracker_active_blocks: float | None = None
        if cfg.aux_variant == "sasa":
            tracker_selector = model.cfg.selection
            tracker_active_blocks = float(model.cfg.k)
        self.tracker = DeadTracker(
            model.cfg.n_blocks,
            capacity=128,
            device=next(model.parameters()).device,
            max_tokens=(cfg.dead_window_tokens if cfg.aux_variant == "sasa" else 0),
            block_dim=model.cfg.block_dim,
            policy=(
                cfg.aux_variant
                if cfg.aux_variant not in {"none", "fel"}
                else "disabled"
            ),
            selector=tracker_selector,
            active_blocks=tracker_active_blocks,
        )
        self.step_idx = 0
        self.accepted_tokens = 0
        self.data_cursor: dict[str, int | str] = {}
        self._prev_shares = (
            site_frobenius_shares(self.master.decoder_tensor()).detach().clone()
        )
        self.history: list[dict] = []
        self._log_file = Path(log_path).open("a") if log_path is not None else None

    # -- one training step -------------------------------------------------

    def _auxiliary_can_have_dead_features(self, batch_tokens: int) -> bool:
        """Return whether the bound deadness criterion can be nonempty."""
        variant = self.cfg.aux_variant
        if variant == "sasa":
            return self.tracker.history_tokens >= self.cfg.dead_window_tokens
        if variant == "long_horizon":
            return self.tracker.tokens_seen >= self.cfg.dead_horizon_tokens
        if variant == "sasa_release":
            return self.tracker.forward_passes > self.cfg.dead_window_passes
        if variant == "decoder_weighted_token_horizon":
            return self.accepted_tokens + batch_tokens >= self.cfg.dead_horizon_tokens
        return True

    def _encoder_observation_mask(self, observed: torch.Tensor) -> torch.Tensor:
        """Sample the augmentation mask without changing missing-data truth.

        ``observed`` is the true data-availability mask and is never expanded.
        Bernoulli masking independently retains sites, repairing an all-hidden
        row by selecting one available site.  Structured modes uniformly hide
        exactly one or retain exactly one available site.  All draws use the
        checkpointed torch RNG on the tensor's device.  The Bernoulli
        zero-probability control consumes no RNG and returns the original mask
        exactly.
        """
        mode = self.cfg.encoder_site_mask_mode
        probability = self.cfg.encoder_site_mask_probability
        if mode == "bernoulli" and probability == 0.0:
            return observed
        if observed.ndim != 2 or observed.shape[1] != self.master.cfg.n_sites:
            raise ValueError(f"observed must have shape [B, {self.master.cfg.n_sites}]")
        if not bool(observed.any(dim=1).all()):
            raise ValueError("every token must have at least one true-observed site")
        if mode != "bernoulli":
            available_counts = observed.sum(dim=1)
            if mode == "exactly_one_hidden" and bool((available_counts < 2).any()):
                raise ValueError(
                    "exactly_one_hidden requires at least two true-observed "
                    "sites in every row"
                )
            chosen = torch.multinomial(observed.to(torch.float32), 1).squeeze(1)
            rows = torch.arange(observed.shape[0], device=observed.device)
            if mode == "exactly_one_hidden":
                augmented = observed.clone()
                augmented[rows, chosen] = False
                return augmented
            augmented = torch.zeros_like(observed)
            augmented[rows, chosen] = True
            return augmented
        retained = torch.rand(observed.shape, device=observed.device) >= probability
        augmented = observed & retained
        repair = ~augmented.any(dim=1)
        if bool(repair.any()):
            available = observed[repair].to(torch.float32)
            chosen = torch.multinomial(available, 1).squeeze(1)
            rows = repair.nonzero(as_tuple=False).squeeze(1)
            augmented[rows, chosen] = True
        return augmented

    def step(
        self,
        x: torch.Tensor,
        observed: torch.Tensor | None = None,
        *,
        materialize_record: bool = True,
    ) -> dict | None:
        cfg = self.cfg
        fwd_param = next(self.fwd.parameters())
        x = x.to(
            device=fwd_param.device,
            dtype=fwd_param.dtype,
            non_blocking=True,
        )
        if observed is None:
            if (
                cfg.encoder_site_mask_mode == "bernoulli"
                and cfg.encoder_site_mask_probability == 0.0
            ):
                encoder_observed = None
            else:
                observed = torch.ones(
                    x.shape[0],
                    self.master.cfg.n_sites,
                    dtype=torch.bool,
                    device=x.device,
                )
                encoder_observed = self._encoder_observation_mask(observed)
        else:
            observed = observed.to(
                device=x.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            if observed.shape != (x.shape[0], self.master.cfg.n_sites):
                raise ValueError(
                    "observed must have shape "
                    f"[{x.shape[0]}, {self.master.cfg.n_sites}]"
                )
            if not bool(observed.any(dim=1).all()):
                raise ValueError(
                    "every token must have at least one true-observed site"
                )
            encoder_observed = self._encoder_observation_mask(observed)
        log_step = self.step_idx % cfg.log_every == 0
        want_record = materialize_record or log_step

        sparse_training_decode = (
            cfg.aux_variant == "none"
            and self.fwd.cfg.lambda_regularizer == 0
            and self.fwd._cuda_sparse_topk_decode_shape_eligible(
                batch=x.shape[0],
                device=x.device,
                dtype=x.dtype,
                mode="topk",
            )
        )
        if sparse_training_decode:
            xhat, _ = self.fwd._forward_factorized_cuda_sparse_topk_training(
                x,
                observed=encoder_observed,
                validate_observed=False,
            )
            # The target mask is the true data-availability mask, not the
            # stochastic augmentation mask: hidden clean sites remain targets.
            l_rec = bsc_reconstruction_loss(
                xhat,
                x,
                self.fwd,
                observation_mask=observed,
                validate_observation_mask=False,
            )
            parts = {"rec": l_rec, "total": l_rec}
            out = decoder = encoder = None
            del xhat
        else:
            out, decoder, encoder = self.fwd.forward_with_materialized(
                x,
                observed=encoder_observed,
                validate_observed=False,
                _score_grad=(
                    self.fwd.cfg.lambda_regularizer > 0
                    and self.fwd.cfg.regularizer == "crosscoder_l1"
                ),
            )
            # The target mask is the true data-availability mask, not the
            # stochastic augmentation mask: hidden clean sites remain targets.
            parts = bsc_loss(
                out,
                x,
                self.fwd,
                observation_mask=observed,
                decoder=decoder,
                encoder=encoder,
                validate_observation_mask=False,
            )

        l_aux = None
        if cfg.aux_variant != "none" and self._auxiliary_can_have_dead_features(len(x)):
            assert out is not None
            dead = None
            if cfg.aux_variant in (
                "sasa",
                "sasa_release",
                "decoder_weighted_token_horizon",
                "long_horizon",
            ):
                if cfg.aux_variant == "sasa_release":
                    dead = self.tracker.dead_coordinates(cfg.dead_window_passes)
                elif cfg.aux_variant == "decoder_weighted_token_horizon":
                    dead = self.tracker.token_horizon_dead_after_current(
                        out.mask, cfg.dead_horizon_tokens
                    )
                else:
                    dead = self.tracker.dead(
                        cfg.aux_variant,
                        threshold=cfg.dead_threshold,
                        window_tokens=cfg.dead_window_tokens,
                        horizon_tokens=cfg.dead_horizon_tokens,
                    )
            l_aux = aux_loss(
                self.fwd,
                x,
                out,
                cfg.aux_variant,
                dead,
                cfg.s_aux,
                observation_mask=observed,
                encoder_observed=encoder_observed,
                reconstruction_loss=cfg.aux_reconstruction,
            )
            if l_aux is not None:
                alpha = 1.0 / cfg.s_aux if cfg.aux_variant == "fel" else cfg.alpha_aux
                parts["aux"] = l_aux
                parts["total"] = parts["total"] + alpha * l_aux

        tracker_mask = (
            out.mask
            if out is not None and cfg.aux_variant not in {"none", "fel"}
            else None
        )
        tracker_coordinate_activity = (
            (out.z_selected != 0)
            if out is not None and cfg.aux_variant == "sasa_release"
            else None
        )
        # The loss graph owns every tensor its backward still needs.  Dropping
        # the aggregate forward result here releases dead score/preselection
        # branches before backward instead of retaining them until the end of
        # the optimizer step.  Decoder/encoder materializations are likewise
        # retained only when a regularizer or backward formula saved them.
        del out, decoder, encoder

        if self.fwd is self.master:
            self.opt.zero_grad(set_to_none=True)
        parts["total"].backward()

        if self.fwd is not self.master:
            for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                if f.grad is None:
                    # Retained bf16 gradients historically became explicit
                    # zeros when a previously used parameter was absent from a
                    # later graph. Preserve its Adam/moment/weight-decay step.
                    if m.grad is not None:
                        m.grad.zero_()
                    continue
                if m.grad is None:
                    m.grad = torch.empty_like(m)
                m.grad.copy_(f.grad.detach())
                # The fp32 master now owns the gradient. Releasing the bf16
                # leaf allocation avoids zero-filling and retaining a second
                # full gradient set between backward passes.
                f.grad = None
        parameters = [p for p in self.master.parameters() if p.grad is not None]
        gradients = [p.grad for p in parameters]
        if not gradients:
            raise RuntimeError("training loss produced no parameter gradients")
        if cfg.gradient_clip_norm is not None:
            unclipped_grad_norm_t = torch.nn.utils.clip_grad_norm_(
                parameters,
                cfg.gradient_clip_norm,
            )
            clip_scale = (
                cfg.gradient_clip_norm / (unclipped_grad_norm_t + 1e-6)
            ).clamp(max=1.0)
            grad_norm_t = unclipped_grad_norm_t * clip_scale
        elif want_record:
            per_tensor_norms = [torch.linalg.vector_norm(g.float()) for g in gradients]
            unclipped_grad_norm_t = torch.linalg.vector_norm(
                torch.stack(per_tensor_norms)
            )
            grad_norm_t = unclipped_grad_norm_t
        else:
            unclipped_grad_norm_t = None
            grad_norm_t = None

        scalar_values: dict[str, float] | None = None
        if want_record:
            if observed is None:
                keep_fraction_t = torch.ones((), device=x.device, dtype=torch.float32)
            else:
                assert encoder_observed is not None
                keep_fraction_t = encoder_observed.sum() / observed.sum()
            assert grad_norm_t is not None and unclipped_grad_norm_t is not None
            scalar_names = [
                "rec",
                "total",
                "keep",
                "grad_norm",
                "unclipped_grad_norm",
            ]
            scalar_tensors = [
                parts["rec"].detach().float(),
                parts["total"].detach().float(),
                keep_fraction_t.detach().float(),
                grad_norm_t.detach().float(),
                unclipped_grad_norm_t.detach().float(),
            ]
            if "regularizer" in parts:
                scalar_names.append("regularizer")
                scalar_tensors.append(parts["regularizer"].detach().float())
            if l_aux is not None:
                scalar_names.append("aux")
                scalar_tensors.append(l_aux.detach().float())
            scalar_values = {
                name: float(value)
                for name, value in zip(
                    scalar_names,
                    torch.stack(scalar_tensors).cpu(),
                )
            }
            if not all(
                math.isfinite(scalar_values[name])
                for name in (
                    "rec",
                    "total",
                    "grad_norm",
                    "unclipped_grad_norm",
                )
            ):
                raise RuntimeError("non-finite loss/gradient/parameter/optimizer state")
        else:
            finite = (
                _all_finite((parts["rec"], parts["total"], unclipped_grad_norm_t))
                if unclipped_grad_norm_t is not None
                else _finite_gradients_with_l2_guard(
                    parts["rec"], parts["total"], gradients
                )
            )
            if not finite:
                raise RuntimeError("non-finite loss/gradient/parameter/optimizer state")

        # The load-bearing ordering: step on master -> retract master ->
        # regenerate the forward copy -> measure the post-cast residual.
        self.opt.step()
        projected_decoder = (self.step_idx + 1) % cfg.retract_every == 0
        if not _all_finite((tuple(self.master.parameters()), self.opt.state)):
            self.opt.zero_grad(set_to_none=True)
            raise RuntimeError(
                "optimizer produced non-finite parameter/state; refusing to "
                "continue (reload the last atomic checkpoint)"
            )
        self.sched.step()
        floor_hits_t: torch.Tensor | None = None
        projected_parameters: tuple[torch.Tensor, ...] = ()
        # ``step_idx`` is zero-based while ``retract_every`` is a cadence
        # in completed optimizer updates. Initialization applies the declared
        # constraint separately, so cadence 20 means updates 20, 40, ....
        if projected_decoder:
            floor_hits_t, projected_parameters = _project_decoder_(self.master)
        if projected_parameters and not _all_finite(projected_parameters):
            raise RuntimeError(
                "decoder projection produced non-finite parameters; refusing "
                "to continue (reload the last atomic checkpoint)"
            )
        if self.fwd is not self.master:
            with torch.no_grad():
                for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                    f.copy_(m)
                # theta is a frozen calibration buffer, not optimizer state.
                # The forward copy is created after checkpoint loading, so a
                # per-step copy only invalidates its CUDA validation cache.
        if cfg.aux_variant not in {"none", "fel"}:
            assert tracker_mask is not None
            self.tracker.update(
                tracker_mask,
                coordinate_activity=tracker_coordinate_activity,
            )
        self.accepted_tokens += int(x.shape[0])

        record: dict | None = None
        if want_record:
            assert scalar_values is not None
            record = {
                "step": self.step_idx,
                "rec": scalar_values["rec"],
                "total": scalar_values["total"],
                "lr": self.sched.get_last_lr()[0],
                "grad_norm": scalar_values["grad_norm"],
                "floor_hits": (int(floor_hits_t) if floor_hits_t is not None else 0),
                "encoder_site_keep_fraction": scalar_values["keep"],
            }
            if cfg.gradient_clip_norm is not None:
                record["grad_norm_unclipped"] = scalar_values["unclipped_grad_norm"]
            if "regularizer" in parts:
                record["regularizer"] = scalar_values["regularizer"]
            if l_aux is not None:
                record["aux"] = scalar_values["aux"]
        if log_step:
            assert record is not None
            record.update(self._diagnostics())
            self.history.append(record)
            if self._log_file is not None:
                self._log_file.write(json.dumps(record) + "\n")
                self._log_file.flush()
        self.step_idx += 1
        return record

    @torch.no_grad()
    def _diagnostics(self) -> dict:
        if self.cfg.aux_variant == "sasa_release":
            d = {
                "dead_frac_scalar_pass_window": float(
                    self.tracker.dead_coordinates(self.cfg.dead_window_passes)
                    .float()
                    .mean()
                )
            }
        elif self.cfg.aux_variant == "decoder_weighted_token_horizon":
            assert self.tracker.tokens_since_fired is not None
            d = {
                "dead_frac_token_horizon": float(
                    (self.tracker.tokens_since_fired >= self.cfg.dead_horizon_tokens)
                    .float()
                    .mean()
                )
            }
        elif self.cfg.aux_variant == "sasa":
            d = {
                "dead_frac_window": float(
                    (
                        self.tracker.frequency(self.cfg.dead_window_tokens)
                        <= self.cfg.dead_threshold
                    )
                    .float()
                    .mean()
                ),
            }
        elif self.cfg.aux_variant == "long_horizon":
            d = {
                "dead_frac_token_horizon": float(
                    self.tracker.dead(
                        "long_horizon",
                        threshold=self.cfg.dead_threshold,
                        window_tokens=self.cfg.dead_window_tokens,
                        horizon_tokens=self.cfg.dead_horizon_tokens,
                    )
                    .float()
                    .mean()
                )
            }
        else:
            d = {}
        # Existing diagnostic synchronizations are the periodic runtime gate;
        # reuse its one Gram scan for both the general constraint metric and
        # the specialization metric.
        if self.master.uses_stiefel_code_norm_decoded_energy:
            master_score_geometry = self.master.validate_decoded_energy_implementation()
            master_residual = float(master_score_geometry["gram_residual_max"])
            d["decoder_constraint_residual_master"] = master_residual
            d["decoded_energy_master_gram_residual"] = master_residual
        else:
            master_residual = _constraint_residual(self.master)
            if master_residual is not None:
                d["decoder_constraint_residual_master"] = master_residual
        if self.fwd is not self.master:
            if self.fwd.uses_stiefel_code_norm_decoded_energy:
                forward_score_geometry = (
                    self.fwd.validate_decoded_energy_implementation()
                )
                postcast_residual = float(forward_score_geometry["gram_residual_max"])
                d["decoder_constraint_residual_postcast"] = postcast_residual
                d["decoded_energy_postcast_gram_residual"] = postcast_residual
            else:
                postcast_residual = _constraint_residual(self.fwd)
                if postcast_residual is not None:
                    d["decoder_constraint_residual_postcast"] = postcast_residual
        shares = site_frobenius_shares(self.master.decoder_tensor()).detach()
        d["share_jump"] = float((shares - self._prev_shares).abs().max())
        self._prev_shares = shares.clone()
        return d

    # -- driving loop -------------------------------------------------------

    def fit(
        self, batches: Iterable[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
    ) -> list[dict]:
        for batch in batches:
            if self.step_idx >= self.cfg.total_steps:
                break
            if isinstance(batch, tuple):
                self.step(batch[0], observed=batch[1], materialize_record=False)
            else:
                self.step(batch, materialize_record=False)
        if self._log_file is not None:
            self._log_file.flush()
        return self.history

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        # Checkpoint boundaries are hard refusal points: a model outside the
        # declared score-geometry envelope must never become durable evidence.
        self.master.validate_decoded_energy_implementation()
        if self.fwd is not self.master:
            self.fwd.validate_decoded_energy_implementation()
        optimizer_state = self.opt.state_dict()
        if (
            validate_optimizer_state_config(
                optimizer_state,
                self.cfg,
                self.optimizer_kind,
            )
            != self._optimizer_contract
        ):
            raise RuntimeError("live optimizer implementation contract changed")
        _validate_optimizer_state_shapes(optimizer_state, self.opt)
        np_state = np.random.get_state()
        payload = {
            "model": self.master.state_dict(),
            "optimizer": optimizer_state,
            "scheduler": self.sched.state_dict(),
            "tracker": self.tracker.state_dict(),
            "step_idx": self.step_idx,
            "accepted_tokens": self.accepted_tokens,
            "data_cursor": copy.deepcopy(self.data_cursor),
            # Diagnostics affect the durable training report and therefore
            # belong to exact resume state just as much as weights and RNG.
            # ``_prev_shares`` is intentionally the last *logged* decoder
            # profile, not necessarily the profile at checkpoint time.
            "history": copy.deepcopy(self.history),
            "diagnostic_prev_shares": self._prev_shares.detach().clone(),
            "model_cfg": asdict(self.master.cfg),
            "train_cfg": asdict(self.cfg),
            "optimizer_kind": self.optimizer_kind,
            "run_binding": copy.deepcopy(self.run_binding),
            "rng": {
                "python": random.getstate(),
                "numpy": {
                    "bit_generator": np_state[0],
                    "state": torch.from_numpy(np_state[1].copy()),
                    "position": np_state[2],
                    "has_gauss": np_state[3],
                    "cached_gaussian": np_state[4],
                },
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "torch_mps": (
                    torch.mps.get_rng_state()
                    if torch.backends.mps.is_available()
                    else None
                ),
            },
        }
        path = Path(path)
        # Free-space check aborts *before* the write: the atomic
        # replace transiently doubles the footprint, and the freed space of
        # an overwritten checkpoint is deliberately not credited.
        nbytes = _payload_nbytes(payload)
        usage = shutil.disk_usage(path.parent if path.parent.name else ".")
        if usage.free - nbytes < CHECKPOINT_FREE_FLOOR_FRAC * usage.total:
            raise RuntimeError(
                f"checkpoint write would breach the "
                f"{CHECKPOINT_FREE_FLOOR_FRAC:.0%} free-space floor "
                f"({usage.free / 1e9:.1f} GB free, payload {nbytes / 1e9:.2f} GB)"
            )
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)  # atomic write-then-rename
        tmp.rename(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        expected_binding: dict | None = None,
    ) -> "Trainer":
        payload = torch.load(path, map_location=device, weights_only=True)
        model_payload = payload.get("model_cfg")
        if not isinstance(model_payload, dict):
            raise ValueError("checkpoint lacks model configuration")
        for identity in MODEL_IMPLEMENTATION_IDENTITY_FIELDS:
            if identity not in model_payload:
                raise ValueError(f"checkpoint lacks {identity} identity")
        train_payload = payload.get("train_cfg")
        if not isinstance(train_payload, dict):
            raise ValueError("checkpoint lacks train configuration")
        cfg = TrainConfig(
            **{
                **train_payload,
                "betas": tuple(train_payload["betas"]),
            }
        )
        optimizer_kind = payload.get("optimizer_kind")
        validate_optimizer_state_config(
            payload.get("optimizer"),
            cfg,
            optimizer_kind,
        )
        accepted_tokens = payload.get("accepted_tokens")
        if (
            not isinstance(accepted_tokens, int)
            or isinstance(accepted_tokens, bool)
            or accepted_tokens < 0
        ):
            raise ValueError("checkpoint lacks a valid exact accepted-token counter")
        step_idx = payload.get("step_idx")
        if not isinstance(step_idx, int) or isinstance(step_idx, bool) or step_idx < 0:
            raise ValueError("checkpoint lacks a valid exact step counter")
        stored_binding = payload.get("run_binding")
        if stored_binding is not None:
            validate_run_binding(
                stored_binding,
                {
                    "model_cfg": payload["model_cfg"],
                    "train_cfg": payload["train_cfg"],
                },
                keys=("model_cfg", "train_cfg"),
            )
        if expected_binding is not None:
            validate_run_binding(stored_binding, expected_binding)
        model_cfg = BSCConfig(**payload["model_cfg"])
        model = BlockCrosscoder(model_cfg).to(device)
        model.load_state_dict(payload["model"])
        trainer = cls(model, cfg, run_binding=stored_binding)
        if optimizer_kind != trainer.optimizer_kind:
            raise ValueError("checkpoint optimizer kind disagrees with constructor")
        if (
            validate_optimizer_state_config(
                payload["optimizer"],
                cfg,
                optimizer_kind,
            )
            != trainer._optimizer_contract
        ):
            raise ValueError("checkpoint optimizer group contract mismatch")
        _validate_optimizer_state_shapes(payload["optimizer"], trainer.opt)
        trainer.opt.load_state_dict(payload["optimizer"])
        if (
            validate_optimizer_state_config(
                trainer.opt.state_dict(),
                cfg,
                trainer.optimizer_kind,
            )
            != trainer._optimizer_contract
        ):
            raise ValueError("optimizer load changed its implementation contract")
        trainer.sched.load_state_dict(payload["scheduler"])
        trainer.tracker.load_state_dict(payload["tracker"])
        trainer.step_idx = step_idx
        trainer.accepted_tokens = accepted_tokens
        if cfg.aux_variant == "sasa":
            expected_history_tokens = min(
                accepted_tokens,
                cfg.dead_window_tokens,
            )
            if trainer.tracker.history_tokens != expected_history_tokens:
                raise ValueError(
                    "checkpoint dead tracker disagrees with accepted-token counter"
                )
        elif cfg.aux_variant == "long_horizon":
            if trainer.tracker.tokens_seen != accepted_tokens:
                raise ValueError(
                    "checkpoint dead tracker disagrees with accepted-token counter"
                )
        elif cfg.aux_variant == "sasa_release":
            if trainer.tracker.forward_passes != step_idx:
                raise ValueError(
                    "checkpoint dead tracker disagrees with exact step counter"
                )
        if cfg.aux_variant == "decoder_weighted_token_horizon":
            assert trainer.tracker.tokens_since_fired is not None
            if bool((trainer.tracker.tokens_since_fired > accepted_tokens).any()):
                raise ValueError(
                    "checkpoint dead tracker exceeds accepted-token counter"
                )
        trainer.data_cursor = dict(payload.get("data_cursor", {}))
        history = payload.get("history")
        previous_shares = payload.get("diagnostic_prev_shares")
        expected_share_shape = site_frobenius_shares(
            trainer.master.decoder_tensor()
        ).shape
        if (
            not isinstance(history, list)
            or any(not isinstance(item, dict) for item in history)
            or not torch.is_tensor(previous_shares)
            or previous_shares.shape != expected_share_shape
            or not bool(torch.isfinite(previous_shares).all())
        ):
            raise ValueError("checkpoint lacks valid exact diagnostic resume state")
        trainer.history = copy.deepcopy(history)
        trainer._prev_shares = (
            previous_shares.to(
                device=trainer.master.parameter_device,
                dtype=trainer.master.parameter_dtype,
            )
            .detach()
            .clone()
        )
        rng = payload.get("rng")
        if rng is not None:
            random.setstate(tuple(rng["python"]))
            np_rng = rng["numpy"]
            np.random.set_state(
                (
                    np_rng["bit_generator"],
                    np_rng["state"].cpu().numpy(),
                    np_rng["position"],
                    np_rng["has_gauss"],
                    np_rng["cached_gaussian"],
                )
            )
            torch.set_rng_state(rng["torch_cpu"].cpu())
            if torch.cuda.is_available() and rng["torch_cuda"]:
                # ``map_location=device`` moves these serialized CPU
                # ByteTensors onto CUDA along with model and optimizer state.
                # PyTorch's CUDA RNG setter nevertheless requires its state
                # tensors to reside on the CPU.
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in rng["torch_cuda"]]
                )
            if torch.backends.mps.is_available() and rng.get("torch_mps") is not None:
                torch.mps.set_rng_state(rng["torch_mps"].cpu())
        if trainer.fwd is not trainer.master:
            with torch.no_grad():
                for m, f in zip(trainer.master.parameters(), trainer.fwd.parameters()):
                    f.copy_(m)
        return trainer
