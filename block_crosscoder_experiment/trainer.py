"""Trainer for the block-sparse crosscoder.

Implements the load-bearing step ordering:

    optimizer step on fp32 master -> retract master decoders ->
    regenerate bf16 forward copy -> log post-cast Gram residual

with explicitly parameterized Adam or AdamW groups, independently declared
encoder/decoder/bias decay, frozen foreach/fused arithmetic, declared warmup
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

from .gram import gram_residual, retract_, site_frobenius_shares
from .model import BlockCrosscoder, BSCConfig, BSCOutput, bsc_loss

__all__ = [
    "TrainConfig",
    "DeadTracker",
    "Trainer",
    "aux_loss",
    "tensor_batches",
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
        if self.foreach is not False or self.fused is not False:
            raise ValueError("portable training requires foreach=False and fused=False")
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
    """Exact token-window frequency plus O(G) last-activation horizons.

    SASA's short frequency window stores boolean token masks and trims the
    oldest chunk at the exact token boundary. Long-horizon deadness does not
    retain a ``horizon x G`` matrix: one last-fire token index per block is
    sufficient and exact. ``max_tokens=0`` disables frequency storage for
    long-horizon-only auxiliary rules.
    """

    def __init__(
        self,
        n_blocks: int,
        capacity: int,
        device,
        *,
        max_tokens: int | None = None,
        block_dim: int = 1,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        self.n_blocks = int(n_blocks)
        self.block_dim = int(block_dim)
        if self.block_dim <= 0:
            raise ValueError("block_dim must be positive")
        self.capacity = int(capacity)
        self.max_tokens = max_tokens
        self.device = torch.device(device)
        self.chunks: list[torch.Tensor] = []
        self.tokens_seen = 0
        self.last_fire = torch.full(
            (n_blocks,), -1, dtype=torch.int64, device=self.device
        )
        # Minder's release increments a feature-age counter by the whole
        # batch size, then resets every feature selected anywhere in that
        # batch.  This is deliberately separate from ``last_fire``: the
        # latter preserves exact within-batch token positions for the adapted
        # block horizon, while the release counter is batch-boundary exact.
        self.tokens_since_fired = torch.zeros(
            n_blocks, dtype=torch.int64, device=self.device
        )
        self.coordinate_passes_since_fired = torch.zeros(
            n_blocks, self.block_dim, dtype=torch.int64, device=self.device
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
        accepted = mask.detach().to(device=self.device, dtype=torch.bool)
        any_fire = accepted.any(dim=0)
        self.tokens_since_fired += len(accepted)
        self.tokens_since_fired[any_fire] = 0
        if bool(any_fire.any()):
            reverse_offset = accepted.flip(0).to(torch.int8).argmax(dim=0)
            positions = self.tokens_seen + len(accepted) - 1 - reverse_offset
            self.last_fire[any_fire] = positions[any_fire]
        self.tokens_seen += len(accepted)

        # Exact SAELens rule: after each forward, increment every scalar
        # coordinate and reset those whose post-selection activation was
        # nonzero anywhere in the batch.  Signed negative values count as
        # firing because bool() is nonzero, matching the release trainer.
        self.coordinate_passes_since_fired += 1
        if coordinate_activity is None:
            coordinate_activity = accepted.unsqueeze(-1).expand(-1, -1, self.block_dim)
        if coordinate_activity.shape != (mask.shape[0], self.n_blocks, self.block_dim):
            raise ValueError(
                "coordinate_activity must have shape "
                f"[B, {self.n_blocks}, {self.block_dim}]"
            )
        did_fire = (
            coordinate_activity.detach()
            .to(device=self.device, dtype=torch.bool)
            .any(dim=0)
        )
        self.coordinate_passes_since_fired[did_fire] = 0
        self.forward_passes += 1

        if self.max_tokens != 0:
            self.chunks.append(accepted.clone())
            if self.max_tokens is None:
                while len(self.chunks) > self.capacity:
                    self.chunks.pop(0)
            else:
                excess = self.history_tokens - self.max_tokens
                while excess > 0 and self.chunks:
                    oldest = self.chunks[0]
                    if excess >= len(oldest):
                        excess -= len(oldest)
                        self.chunks.pop(0)
                    else:
                        self.chunks[0] = oldest[excess:].clone()
                        excess = 0

    @property
    def history_tokens(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    def frequency(self, window_tokens: int) -> torch.Tensor:
        """Per-block frequency over the last ``window_tokens`` accepted tokens.

        Stored boolean chunks are sliced at the exact token boundary, so batch
        size and partial final batches do not change the criterion.
        """
        if window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        if not self.chunks:
            return torch.zeros(self.n_blocks, device=self.device)
        remaining = min(window_tokens, self.history_tokens)
        total = torch.zeros(self.n_blocks, device=self.device)
        for chunk in reversed(self.chunks):
            take = min(remaining, len(chunk))
            total += chunk[-take:].sum(dim=0, dtype=torch.float32)
            remaining -= take
            if remaining == 0:
                break
        denominator = min(window_tokens, self.history_tokens)
        return total / max(1, denominator)

    def dead(
        self,
        variant: str,
        *,
        threshold: float,
        window_tokens: int,
        horizon_tokens: int,
    ) -> torch.Tensor:
        """Bool [G]. All-False until the token-denominated history is full."""
        G = self.n_blocks
        device = self.device
        if variant == "sasa":
            if self.history_tokens < window_tokens:
                return torch.zeros(G, dtype=torch.bool, device=device)
            return self.frequency(window_tokens) <= threshold
        if variant == "long_horizon":
            if self.tokens_seen < horizon_tokens:
                return torch.zeros(G, dtype=torch.bool, device=device)
            return self.last_fire < self.tokens_seen - horizon_tokens
        raise ValueError(f"no dead criterion for variant {variant!r}")

    def dead_coordinates(self, window_passes: int) -> torch.Tensor:
        """SAELens scalar dead mask, shape [G,b], evaluated before a forward."""
        if window_passes <= 0:
            raise ValueError("window_passes must be positive")
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
        if horizon_tokens <= 0:
            raise ValueError("horizon_tokens must be positive")
        if current_mask.ndim != 2 or current_mask.shape[1] != self.n_blocks:
            raise ValueError(f"current_mask must have shape [B, {self.n_blocks}]")
        projected = self.tokens_since_fired + int(current_mask.shape[0])
        projected = projected.clone()
        projected[current_mask.detach().to(self.device).any(dim=0)] = 0
        return projected >= horizon_tokens

    def state_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "tokens_seen": self.tokens_seen,
            "last_fire": self.last_fire,
            "tokens_since_fired": self.tokens_since_fired,
            "capacity": self.capacity,
            "max_tokens": self.max_tokens,
            "block_dim": self.block_dim,
            "coordinate_passes_since_fired": self.coordinate_passes_since_fired,
            "forward_passes": self.forward_passes,
        }

    def load_state_dict(self, state: dict) -> None:
        self.chunks = [
            chunk.to(device=self.device, dtype=torch.bool).clone()
            for chunk in state.get("chunks", [])
        ]
        self.tokens_seen = int(state.get("tokens_seen", self.history_tokens))
        self.last_fire = state.get("last_fire", self.last_fire).to(
            device=self.device, dtype=torch.int64
        )
        self.tokens_since_fired = state.get(
            "tokens_since_fired", self.tokens_since_fired
        ).to(device=self.device, dtype=torch.int64)
        self.capacity = int(state.get("capacity", self.capacity))
        self.max_tokens = state.get("max_tokens", self.max_tokens)
        stored_dim = int(state.get("block_dim", self.block_dim))
        if stored_dim != self.block_dim:
            raise ValueError("dead-tracker block_dim changed across resume")
        self.coordinate_passes_since_fired = state.get(
            "coordinate_passes_since_fired",
            self.coordinate_passes_since_fired,
        ).to(device=self.device, dtype=torch.int64)
        self.forward_passes = int(state.get("forward_passes", self.forward_passes))


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
    site_mask = (
        torch.ones(B, model.cfg.n_sites, 1, device=x.device, dtype=x.dtype)
        if observation_mask is None
        else observation_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    )

    def reconstruction(
        error: torch.Tensor,
        residual_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coord = model.coordinate_mask[:, 0, 0].to(error.device)
        masked = error.float() * coord * site_mask
        if reconstruction_loss == "mean_l2":
            denominator = site_mask.squeeze(-1).sum().clamp_min(1.0)
            return masked.norm(dim=-1).sum() / denominator
        if reconstruction_loss == "mean_squared":
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
            target = residual_target.float() * coord * site_mask
            observed_values = target.sum(dim=0) / site_mask.sum(dim=0).clamp_min(1.0)
            centered = (target - observed_values.unsqueeze(0)) * site_mask
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


def _all_finite(obj) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in _floating_tensors(obj))


def _project_decoder_(model: BlockCrosscoder) -> int:
    project = getattr(model, "project_decoder_", None)
    if project is not None:
        result = project()
        return 0 if result is None else int(result)
    return retract_(model.D.data, eig_floor=model.cfg.eig_floor)


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
        self.opt, self.optimizer_kind = build_optimizer(self.master, cfg)
        self.sched = LambdaLR(self.opt, _lr_factor(cfg))
        self.tracker = DeadTracker(
            model.cfg.n_blocks,
            capacity=128,
            device=next(model.parameters()).device,
            max_tokens=(cfg.dead_window_tokens if cfg.aux_variant == "sasa" else 0),
            block_dim=model.cfg.block_dim,
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
    ) -> dict:
        cfg = self.cfg
        fwd_param = next(self.fwd.parameters())
        x = x.to(device=fwd_param.device, dtype=fwd_param.dtype)
        if observed is None:
            observed = torch.ones(
                x.shape[0],
                self.master.cfg.n_sites,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            observed = observed.to(device=x.device, dtype=torch.bool)
            if observed.shape != (x.shape[0], self.master.cfg.n_sites):
                raise ValueError(
                    "observed must have shape "
                    f"[{x.shape[0]}, {self.master.cfg.n_sites}]"
                )
        if not bool(observed.any(dim=1).all()):
            raise ValueError("every token must have at least one true-observed site")
        encoder_observed = self._encoder_observation_mask(observed)
        log_step = self.step_idx % cfg.log_every == 0

        out = self.fwd(x, observed=encoder_observed)
        # The target mask is the true data-availability mask, not the stochastic
        # augmentation mask: hidden clean sites remain reconstruction targets.
        parts = bsc_loss(out, x, self.fwd, observation_mask=observed)

        l_aux = None
        if cfg.aux_variant != "none":
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

        self.opt.zero_grad(set_to_none=True)
        if self.fwd is not self.master:
            for p in self.fwd.parameters():
                p.grad = None
        parts["total"].backward()

        if self.fwd is not self.master:
            for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                if f.grad is None:
                    continue
                if m.grad is None:
                    m.grad = f.grad.detach().float()
                else:
                    m.grad.copy_(f.grad.detach())
        unclipped_grad_norm = math.sqrt(
            sum(
                float(p.grad.float().pow(2).sum())
                for p in self.master.parameters()
                if p.grad is not None
            )
        )
        if cfg.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.master.parameters() if p.grad is not None],
                cfg.gradient_clip_norm,
            )
        grad_norm = math.sqrt(
            sum(
                float(p.grad.float().pow(2).sum())
                for p in self.master.parameters()
                if p.grad is not None
            )
        )

        state_finite = _all_finite(tuple(self.master.parameters())) and _all_finite(
            self.opt.state
        )
        rec_val = float(parts["rec"].detach())
        if not (
            math.isfinite(grad_norm)
            and math.isfinite(rec_val)
            and math.isfinite(float(parts["total"].detach()))
            and state_finite
        ):
            raise RuntimeError("non-finite loss/gradient/parameter/optimizer state")

        # The load-bearing ordering: step on master -> retract master ->
        # regenerate the forward copy -> measure the post-cast residual.
        self.opt.step()
        if not (
            _all_finite(tuple(self.master.parameters())) and _all_finite(self.opt.state)
        ):
            self.opt.zero_grad(set_to_none=True)
            raise RuntimeError(
                "optimizer produced non-finite parameter/state; refusing to "
                "continue (reload the last atomic checkpoint)"
            )
        self.sched.step()
        floor_hits = 0
        # ``step_idx`` is zero-based while ``retract_every`` is a cadence
        # in completed optimizer updates. Initialization applies the declared
        # constraint separately, so cadence 20 means updates 20, 40, ....
        if (self.step_idx + 1) % cfg.retract_every == 0:
            floor_hits = _project_decoder_(self.master)
        if not _all_finite(tuple(self.master.parameters())):
            raise RuntimeError(
                "decoder projection produced non-finite parameters; refusing "
                "to continue (reload the last atomic checkpoint)"
            )
        if self.fwd is not self.master:
            with torch.no_grad():
                for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                    f.copy_(m)
                self.fwd.theta.copy_(self.master.theta)
        self.tracker.update(
            out.mask,
            coordinate_activity=(out.z_selected != 0),
        )
        self.accepted_tokens += int(x.shape[0])

        record = {
            "step": self.step_idx,
            "rec": rec_val,
            "total": float(parts["total"].detach()),
            "lr": self.sched.get_last_lr()[0],
            "grad_norm": grad_norm,
            "floor_hits": floor_hits,
            "encoder_site_keep_fraction": float(
                encoder_observed.sum() / observed.sum()
            ),
        }
        if cfg.gradient_clip_norm is not None:
            record["grad_norm_unclipped"] = unclipped_grad_norm
        if "regularizer" in parts:
            record["regularizer"] = float(parts["regularizer"].detach())
        if l_aux is not None:
            record["aux"] = float(l_aux.detach())
        if log_step:
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
            d = {
                "dead_frac_token_horizon": float(
                    (self.tracker.tokens_since_fired >= self.cfg.dead_horizon_tokens)
                    .float()
                    .mean()
                )
            }
        else:
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
        master_residual = _constraint_residual(self.master)
        if master_residual is not None:
            d["decoder_constraint_residual_master"] = master_residual
        if self.fwd is not self.master:
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
                self.step(batch[0], observed=batch[1])
            else:
                self.step(batch)
        if self._log_file is not None:
            self._log_file.flush()
        return self.history

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        np_state = np.random.get_state()
        payload = {
            "model": self.master.state_dict(),
            "optimizer": self.opt.state_dict(),
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
        cfg = TrainConfig(
            **{
                **payload["train_cfg"],
                "betas": tuple(payload["train_cfg"]["betas"]),
            }
        )
        model = BlockCrosscoder(model_cfg).to(device)
        model.load_state_dict(payload["model"])
        trainer = cls(model, cfg, run_binding=stored_binding)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.sched.load_state_dict(payload["scheduler"])
        trainer.tracker.load_state_dict(payload["tracker"])
        trainer.step_idx = payload["step_idx"]
        trainer.accepted_tokens = payload.get("accepted_tokens", 0)
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
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            if torch.backends.mps.is_available() and rng.get("torch_mps") is not None:
                torch.mps.set_rng_state(rng["torch_mps"].cpu())
        if trainer.fwd is not trainer.master:
            with torch.no_grad():
                for m, f in zip(trainer.master.parameters(), trainer.fwd.parameters()):
                    f.copy_(m)
        return trainer
