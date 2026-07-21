"""Block-sparse crosscoder: G blocks of width b, one shared code across sites.

All model mathematics lives in the store's declared per-site coordinates;
the model is agnostic to whether they are raw, scalar-normalized,
token-LayerNorm, or whitened. Inputs are batches x: [B, S, d].

Unfactorized parameter stacks are [S, G, b, d].  A declared novel arm may
instead use a low-rank site-axis factorization
``W[s,g,b,d] = sum_r A[s,r] B[r,g,b,d]`` for both encoder and decoder.  Every
consumer still receives the materialized tensor through ``encoder_tensor`` or
``decoder_tensor``; parameters are fp32 masters, and the bf16 forward copy is
the trainer's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch import nn

from .gram import (
    decoder_nuclear_penalty,
    init_decoder_stack,
    map_nuclear_penalty,
    normalize_block_frobenius_,
    project_block_frobenius_,
    project_latent_rows_,
    qr_retract_,
    retract_,
)

__all__ = [
    "BSCConfig",
    "BSCOutput",
    "BlockCrosscoder",
    "SignedStreamingScoreQuantile",
    "StreamingScoreQuantile",
    "batch_topk_mask",
    "token_topk_mask",
    "bsc_loss",
]


class StreamingScoreQuantile:
    """Bounded-memory pooled-score quantile for full-split calibration.

    A fixed log-spaced histogram over the selection-score range: scores
    are non-negative block norms, so bins span [lo, hi] geometrically
    with dedicated underflow (score <= lo, including exact zeros) and
    overflow bins. Counts are int64 — accumulation is deterministic and
    batch-order independent, unlike any floating-point running sum. Peak
    memory is O(n_bins) (~8 MB at the default 2^20), independent of the
    calibration token count; per-batch work is one bucketize + bincount
    on the scores' device.

    Resolution: one bin = hi/lo spread over n_bins geometrically =
    ~3.1e-5 relative width at the defaults — far inside the
    |Δ avg-blocks| <= 0.1 validation gate. The quantile returns the
    geometric midpoint of the crossing bin.

    Non-finite scores raise: a NaN/inf selection score is a bug upstream
    and silence here would launder it into theta.
    """

    def __init__(
        self,
        n_bins: int = 1 << 20,
        lo: float = 1e-9,
        hi: float = 1e5,
        device: torch.device | str = "cpu",
    ) -> None:
        if not (0 < lo < hi):
            raise ValueError("need 0 < lo < hi")
        self.lo, self.hi, self.n_bins = float(lo), float(hi), int(n_bins)
        self.edges = torch.logspace(
            math.log10(lo),
            math.log10(hi),
            n_bins + 1,
            dtype=torch.float32,
            device=device,
        )
        # counts[0] = underflow (<= lo), counts[1..n_bins] = bins,
        # counts[n_bins+1] = overflow (> hi).
        self.counts = torch.zeros(n_bins + 2, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        s = scores.detach().flatten().float()
        if not torch.isfinite(s).all():
            raise ValueError("non-finite selection scores in calibration batch")
        idx = torch.bucketize(s, self.edges, right=False)
        self.counts += torch.bincount(idx, minlength=self.n_bins + 2)

    def quantile(self, q: float) -> float:
        n = int(self.counts.sum())
        if n == 0:
            raise ValueError("no scores accumulated")
        target = min(max(int(round(q * n)), 1), n)
        cum = torch.cumsum(self.counts, dim=0)
        bin_idx = int(torch.searchsorted(cum, torch.tensor(target), right=False))
        if bin_idx == 0:
            return self.lo
        if bin_idx >= self.n_bins + 1:
            return self.hi
        lo_edge = float(self.edges[bin_idx - 1])
        hi_edge = float(self.edges[bin_idx])
        return math.sqrt(lo_edge * hi_edge)


class SignedStreamingScoreQuantile:
    """Bounded-memory quantile for signed selection scores.

    Isolated reconstruction gain is intentionally signed: a candidate whose
    decoded write points away from the observed input must rank below zero,
    not be clipped into a tie.  Two symmetric log grids preserve relative
    resolution away from zero, with an explicit zero boundary for the central
    bins.  Counts remain deterministic, order-independent int64 values.
    """

    def __init__(
        self,
        bins_per_sign: int = 1 << 19,
        lo: float = 1e-9,
        hi: float = 1e5,
        device: torch.device | str = "cpu",
    ) -> None:
        if bins_per_sign <= 0 or not (0 < lo < hi):
            raise ValueError("need positive bins_per_sign and 0 < lo < hi")
        self.lo = float(lo)
        self.hi = float(hi)
        magnitude = torch.logspace(
            math.log10(lo),
            math.log10(hi),
            int(bins_per_sign) + 1,
            dtype=torch.float32,
            device=device,
        )
        self.edges = torch.cat(
            (-magnitude.flip(0), torch.zeros(1, device=device), magnitude)
        )
        self.counts = torch.zeros(
            len(self.edges) + 1,
            dtype=torch.int64,
            device=device,
        )

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        values = scores.detach().flatten().float()
        if not torch.isfinite(values).all():
            raise ValueError("non-finite selection scores in calibration batch")
        indices = torch.bucketize(
            values,
            self.edges,
            right=False,
        )
        self.counts += torch.bincount(indices, minlength=len(self.edges) + 1)

    def quantile(self, q: float) -> float:
        count = int(self.counts.sum())
        if count == 0:
            raise ValueError("no scores accumulated")
        target = min(max(int(round(q * count)), 1), count)
        cumulative = torch.cumsum(self.counts, dim=0)
        bin_index = int(
            torch.searchsorted(cumulative, torch.tensor(target), right=False)
        )
        if bin_index == 0:
            return -self.hi
        if bin_index >= len(self.edges):
            return self.hi
        lower = float(self.edges[bin_index - 1])
        upper = float(self.edges[bin_index])
        if lower <= 0.0 <= upper:
            return 0.0
        magnitude_midpoint = math.sqrt(abs(lower * upper))
        return -magnitude_midpoint if upper < 0.0 else magnitude_midpoint


@dataclass
class BSCConfig:
    n_blocks: int  # G
    block_dim: int  # b
    n_sites: int  # S
    d_model: int  # d
    k: float  # average active blocks/token (BatchTopK budget; fractional OK)
    lambda_regularizer: float = 0.0
    eig_floor: float = 1e-6  # retraction eigenvalue floor
    sv_eps: float = 1e-8  # eps inside sqrt(eig + eps)
    seed: int = 0
    selection: str = "batch_topk"  # batch_topk | token_topk | threshold | dense
    # Paper implementations do not disclose an exact tie rule.  Binding the
    # cutoff fill to candidate index makes zero/ReLU ties reproducible without
    # changing the score ordering away from the cutoff.
    selector_tie_break: str = "lowest_flat_index_at_cutoff"
    encoder_mode: str = "untied"  # untied | tied
    encoder_bias: bool = False
    code_activation: str = "signed"  # signed | relu | group_soft_threshold
    selection_score: str = (
        "code_norm"  # plus decoder_weighted/decoded_energy/loss decrease
    )
    decoder_constraint: str = "gram"  # plus QR, ball/equality Frobenius, row-unit, free
    encoder_constraint: str = "none"  # none | unit_latent
    regularizer: str | None = None  # plus map nuclear, crosscoder L1, group L21
    # Cross-site topology. ``site_dims`` permits rectangular sites inside a
    # padded [B,S,max(d_s)] tensor; padded coordinates are structurally zero.
    site_dims: tuple[int, ...] | None = None
    # ``availability_rescaled_sum`` preserves the full-site summed encoder's
    # scale under clean-target site masking: S / n_visible times the visible
    # site sum.  It is an exact sum when every site is present.
    encoder_fusion: str = "sum"  # sum | mean | availability_rescaled_sum | source
    source_site: int = 0
    # Paper/code objective forks must be explicit rather than inherited from
    # the runtime environment.
    reconstruction_loss: str = (
        "mean_squared"  # mean_squared | squared_l2 | mean_l2 | mean_l1
    )
    decoder_norm_geometry: str = "sum_l2"  # sum_l2 | concat_l2
    decoder_bias: bool = True
    decoder_bias_init: str = "zero"  # zero | data_mean | geometric_median
    apply_decoder_bias_to_input: bool = False
    decoder_init_distribution: str = "gaussian_std_inverse_sqrt_d"
    decoder_init_preconditioning: str = "concatenated_gram_retraction"
    decoder_init_operation_order: str = (
        "gaussian_precondition_mask_rescale_then_declared_constraint"
    )
    encoder_init: str = "decoder_transpose"  # decoder_transpose | independent
    encoder_scale_init: float = 1.0  # tied global gamma; inert for untied encoders
    identical_site_init: bool = False
    group_lasso_target_k: float | None = None
    group_threshold_scope: str = "per_block"  # per_block | shared_scalar
    group_threshold_parameterization: str = "softplus"  # softplus | exp
    group_threshold_raw_init: float | None = None
    group_threshold_effective_init: float = 0.1
    map_nuclear_reduction: str = "mean_normalized"  # mean_normalized | sum_blocks
    # Novel FMX-inspired adaptation: factor only the site/layer axis while
    # retaining the BSC block coordinate as the sparse unit. ``None`` is the
    # full unfactorized control. The factorized arm is deliberately narrow;
    # see the validation below.
    site_rank: int | None = None

    def __post_init__(self) -> None:
        if self.selection not in {"batch_topk", "token_topk", "threshold", "dense"}:
            raise ValueError(
                "selection must be batch_topk, token_topk, threshold, or dense"
            )
        if self.selector_tie_break != "lowest_flat_index_at_cutoff":
            raise ValueError("selector_tie_break must be lowest_flat_index_at_cutoff")
        if self.encoder_mode not in {"untied", "tied"}:
            raise ValueError("encoder_mode must be untied or tied")
        if self.code_activation not in {"signed", "relu", "group_soft_threshold"}:
            raise ValueError(
                "code_activation must be signed, relu, or group_soft_threshold"
            )
        if self.selection_score not in {
            "code_norm",
            "decoder_weighted",
            "decoded_energy",
            "isolated_loss_decrease",
        }:
            raise ValueError(
                "selection_score must be code_norm, decoder_weighted, or "
                "decoded_energy, or isolated_loss_decrease"
            )
        if self.decoder_constraint not in {
            "gram",
            "qr",
            "frobenius",
            "unit_frobenius",
            "unit_latent",
            "free",
        }:
            raise ValueError(
                "decoder_constraint must be gram, qr, frobenius, unit_frobenius, "
                "unit_latent, or free"
            )
        if self.encoder_constraint not in {"none", "unit_latent"}:
            raise ValueError("encoder_constraint must be none or unit_latent")
        if self.site_dims is None:
            self.site_dims = (self.d_model,) * self.n_sites
        else:
            self.site_dims = tuple(int(v) for v in self.site_dims)
        if len(self.site_dims) != self.n_sites:
            raise ValueError("site_dims must have one entry per site")
        if any(v <= 0 or v > self.d_model for v in self.site_dims):
            raise ValueError("site_dims entries must be in [1, d_model]")
        if self.encoder_fusion not in {
            "sum",
            "mean",
            "availability_rescaled_sum",
            "source",
        }:
            raise ValueError(
                "encoder_fusion must be sum, mean, availability_rescaled_sum, or source"
            )
        if not 0 <= self.source_site < self.n_sites:
            raise ValueError("source_site is outside the site range")
        if self.reconstruction_loss not in {
            "mean_squared",
            "squared_l2",
            "mean_l2",
            "mean_l1",
        }:
            raise ValueError(
                "reconstruction_loss must be mean_squared, squared_l2, mean_l2, or mean_l1"
            )
        if self.decoder_norm_geometry not in {"sum_l2", "concat_l2"}:
            raise ValueError("decoder_norm_geometry must be sum_l2 or concat_l2")
        if self.decoder_init_distribution != "gaussian_std_inverse_sqrt_d":
            raise ValueError("unsupported decoder initialization distribution")
        valid_init_orders = {
            "concatenated_gram_retraction": (
                "gaussian_precondition_mask_rescale_then_declared_constraint"
            ),
            "none": "gaussian_mask_rescale_then_declared_constraint",
        }
        expected_order = valid_init_orders.get(self.decoder_init_preconditioning)
        if expected_order is None:
            raise ValueError("unsupported decoder initialization preconditioning")
        if self.decoder_init_operation_order != expected_order:
            raise ValueError(
                "decoder initialization operation order does not match its "
                "declared preconditioning"
            )
        if self.decoder_bias_init not in {"zero", "data_mean", "geometric_median"}:
            raise ValueError(
                "decoder_bias_init must be zero, data_mean, or geometric_median"
            )
        if self.decoder_bias_init != "zero" and not self.decoder_bias:
            raise ValueError(
                "a disabled decoder bias cannot use a data-derived initializer"
            )
        if self.apply_decoder_bias_to_input and not self.decoder_bias:
            raise ValueError("pre-encoder decoder-bias centering requires decoder_bias")
        if self.encoder_init not in {"decoder_transpose", "independent"}:
            raise ValueError("encoder_init must be decoder_transpose or independent")
        if self.encoder_scale_init <= 0:
            raise ValueError("encoder_scale_init must be positive")
        if self.group_lasso_target_k is not None and self.group_lasso_target_k <= 0:
            raise ValueError("group_lasso_target_k must be positive")
        if self.group_threshold_scope not in {"per_block", "shared_scalar"}:
            raise ValueError("group_threshold_scope must be per_block or shared_scalar")
        if self.group_threshold_parameterization not in {"softplus", "exp"}:
            raise ValueError("group_threshold_parameterization must be softplus or exp")
        if self.group_threshold_effective_init <= 0 or not math.isfinite(
            self.group_threshold_effective_init
        ):
            raise ValueError(
                "group_threshold_effective_init must be finite and positive"
            )
        if self.group_threshold_raw_init is not None:
            if not math.isfinite(self.group_threshold_raw_init):
                raise ValueError("group_threshold_raw_init must be finite")
            implied = (
                (
                    math.log1p(math.exp(self.group_threshold_raw_init))
                    if self.group_threshold_raw_init <= 20.0
                    else self.group_threshold_raw_init
                )
                if self.group_threshold_parameterization == "softplus"
                else math.exp(self.group_threshold_raw_init)
            )
            if not math.isclose(
                implied,
                self.group_threshold_effective_init,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "raw and effective group-threshold initializers disagree"
                )
        if self.map_nuclear_reduction not in {"mean_normalized", "sum_blocks"}:
            raise ValueError(
                "map_nuclear_reduction must be mean_normalized or sum_blocks"
            )
        if self.regularizer is None:
            self.regularizer = "none"
        if self.regularizer not in {
            "none",
            "map_nuclear",
            "decoder_nuclear",
            "crosscoder_l1",
            "group_l21",
        }:
            raise ValueError("unknown regularizer")
        if self.regularizer == "none" and self.lambda_regularizer > 0:
            raise ValueError(
                "positive lambda_regularizer requires an explicit supported regularizer"
            )
        if self.regularizer == "crosscoder_l1" and self.code_activation != "relu":
            raise ValueError("crosscoder_l1 requires relu codes")
        if (
            self.regularizer == "group_l21"
            and self.code_activation != "group_soft_threshold"
        ):
            raise ValueError("group_l21 requires group_soft_threshold codes")
        if self.selection == "dense" and self.code_activation != "relu":
            if self.code_activation != "group_soft_threshold":
                raise ValueError(
                    "dense selection requires relu or group_soft_threshold codes"
                )
        if (
            self.selection_score == "decoder_weighted"
            and self.code_activation != "relu"
        ):
            raise ValueError("decoder_weighted selection is the ReLU crosscoder bridge")
        if self.selection_score == "isolated_loss_decrease" and (
            self.decoder_bias
            or self.reconstruction_loss not in {"mean_squared", "squared_l2"}
        ):
            raise ValueError(
                "isolated_loss_decrease requires a bias-free quadratic "
                "reconstruction objective"
            )
        if self.site_rank is not None:
            if (
                isinstance(self.site_rank, bool)
                or int(self.site_rank) != self.site_rank
            ):
                raise ValueError("site_rank must be an integer")
            self.site_rank = int(self.site_rank)
            if not 1 <= self.site_rank <= self.n_sites:
                raise ValueError("site_rank must be in [1, n_sites]")
            if len(set(self.site_dims)) != 1:
                raise ValueError("site-axis factorization requires equal site widths")
            if self.encoder_mode != "untied":
                raise ValueError("site-axis factorization requires an untied encoder")
            if self.decoder_constraint != "free":
                raise ValueError(
                    "site-axis factorization requires a free decoder; constrained "
                    "full-tensor projection is not exact in factor space"
                )
            if self.encoder_constraint != "none":
                raise ValueError(
                    "site-axis factorization requires encoder_constraint='none'"
                )

    @property
    def n_latents(self) -> int:
        return self.n_blocks * self.block_dim


class BSCOutput(NamedTuple):
    xhat: torch.Tensor  # [B, S, d] reconstruction in declared coordinates
    z: torch.Tensor  # [B, G, b] pre-selection code
    z_selected: torch.Tensor  # [B, G, b] post-selection code (masked)
    scores: torch.Tensor  # [B, G] selection scores p_g = ||z_g||
    mask: torch.Tensor  # [B, G] bool, selected blocks


def _site_axis_factorize(
    tensor: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic truncated SVD of the site unfolding.

    Returning an identity site factor at ``rank == S`` makes the full-rank
    factorized control reconstruct the initialized tensor bit-for-bit.  Lower
    ranks use the best Frobenius approximation of the site unfolding.
    """
    n_sites, n_blocks, block_dim, d_model = tensor.shape
    if rank == n_sites:
        return (
            torch.eye(n_sites, dtype=tensor.dtype, device=tensor.device),
            tensor.detach().clone(),
        )
    # Initialization is seed-bound. Run the only decomposition on CPU even
    # when the requested model device is CUDA/MPS so accelerator SVD kernels
    # cannot introduce a device-specific initialization fork.
    matrix = tensor.detach().cpu().reshape(n_sites, -1).float()
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    site = left[:, :rank].to(device=tensor.device, dtype=tensor.dtype)
    core = (
        (singular[:rank, None] * right[:rank])
        .reshape(rank, n_blocks, block_dim, d_model)
        .to(device=tensor.device, dtype=tensor.dtype)
    )
    return site, core


def batch_topk_mask(scores: torch.Tensor, k: float) -> torch.Tensor:
    """BatchTopK over blocks: keep the top round(k*B) block-activations
    batch-wide. Fractional k sets the budget below one block per token —
    the under-provisioned regime the capture sweep probes.

    Per-token counts vary by design; only the batch total is pinned.
    scores: [B, G]  ->  bool mask [B, G]
    """
    B, G = scores.shape
    n_keep = min(int(round(k * B)), B * G)
    if n_keep == 0:
        return torch.zeros(B, G, dtype=torch.bool, device=scores.device)
    if n_keep == B * G:
        return torch.ones(B, G, dtype=torch.bool, device=scores.device)
    flat = scores.reshape(-1)
    cutoff = flat.topk(n_keep, sorted=False).values.min()
    strictly_above = flat > cutoff
    remaining = n_keep - strictly_above.sum()
    tied = flat == cutoff
    # Row-major flattening is the declared candidate index for BatchTopK.
    tie_rank = tied.to(torch.int32).cumsum(dim=0, dtype=torch.int32)
    mask = strictly_above | (tied & (tie_rank <= remaining))
    return mask.view(B, G)


def token_topk_mask(scores: torch.Tensor, k: float) -> torch.Tensor:
    """Per-token block TopK used by the published BSF and SASA recipes."""
    B, G = scores.shape
    n_keep = min(max(int(round(k)), 0), G)
    if n_keep == 0:
        return torch.zeros(B, G, dtype=torch.bool, device=scores.device)
    if n_keep == G:
        return torch.ones(B, G, dtype=torch.bool, device=scores.device)
    cutoff = (
        scores.topk(n_keep, dim=1, sorted=False).values.min(dim=1, keepdim=True).values
    )
    strictly_above = scores > cutoff
    remaining = n_keep - strictly_above.sum(dim=1, keepdim=True)
    tied = scores == cutoff
    # Within each token, block index is the declared candidate index.
    tie_rank = tied.to(torch.int32).cumsum(dim=1, dtype=torch.int32)
    return strictly_above | (tied & (tie_rank <= remaining))


class BlockCrosscoder(nn.Module):
    """Gram-constrained block-sparse crosscoder.

    encode:  z_g = sum_s E_g^s x^s          (summed per-site maps, untied)
    select:  BatchTopK on p_g = ||z_g||     (exact contribution energy under
                                             the Gram constraint)
    decode:  xhat^s = c^s + sum_{g active} D_g^s^T z_g
    """

    E: nn.Parameter | None  # [S,G,b,d], absent when tied or factorized
    D: nn.Parameter | None  # [S,G,b,d], absent when factorized
    E_site: nn.Parameter | None  # [S,R]
    E_core: nn.Parameter | None  # [R,G,b,d]
    D_site: nn.Parameter | None  # [S,R]
    D_core: nn.Parameter | None  # [R,G,b,d]
    c: nn.Parameter  # [S, d]
    theta: torch.Tensor  # scalar buffer, inference selection threshold

    def __init__(
        self,
        cfg: BSCConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        # Own copy: k is mutated in place by budget annealing, and an
        # aliased caller config would leak that mutation.
        self.cfg = replace(cfg)
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        D = init_decoder_stack(
            cfg.n_sites,
            cfg.n_blocks,
            cfg.block_dim,
            cfg.d_model,
            generator=gen,
            preconditioning=cfg.decoder_init_preconditioning,
        )
        if device is not None:
            D = D.to(device)
        coordinate_mask = torch.arange(cfg.d_model, device=D.device).view(1, 1, 1, -1)
        coordinate_mask = coordinate_mask < torch.tensor(
            cfg.site_dims, device=D.device
        ).view(cfg.n_sites, 1, 1, 1)
        self.register_buffer("coordinate_mask", coordinate_mask)
        self._has_padded_coordinates = any(
            width != cfg.d_model for width in cfg.site_dims
        )
        D.mul_(coordinate_mask)
        if cfg.identical_site_init:
            if len(set(cfg.site_dims)) != 1:
                raise ValueError("identical site initialization requires equal widths")
            D.copy_(D[:1].expand_as(D))
        # Structural masking, identical-site copying, and source norm
        # initialization all change the block Gram. Enforce
        # the declared constraint only after those operations, so the very
        # first forward has the same geometry as every post-step forward.
        if cfg.decoder_constraint == "gram":
            retract_(D, eig_floor=cfg.eig_floor)
        elif cfg.decoder_constraint == "qr":
            qr_retract_(D)
        elif cfg.decoder_constraint == "frobenius":
            project_block_frobenius_(D)
        elif cfg.decoder_constraint == "unit_frobenius":
            normalize_block_frobenius_(D)
        elif cfg.decoder_constraint == "unit_latent":
            project_latent_rows_(D)
        D.mul_(coordinate_mask)
        if cfg.site_rank is None:
            self.D = nn.Parameter(D)
            self.register_parameter("D_site", None)
            self.register_parameter("D_core", None)
        else:
            decoder_site, decoder_core = _site_axis_factorize(D, cfg.site_rank)
            self.register_parameter("D", None)
            self.D_site = nn.Parameter(decoder_site)
            self.D_core = nn.Parameter(decoder_core)
        # Transpose-tied at init only (Fel App. D convention); encoder scale
        # is norm-calibrated on a data batch via calibrate_encoder_scale_.
        if cfg.encoder_mode == "untied":
            if cfg.encoder_init == "decoder_transpose":
                encoder = D.detach().clone()
            else:
                encoder = init_decoder_stack(
                    cfg.n_sites,
                    cfg.n_blocks,
                    cfg.block_dim,
                    cfg.d_model,
                    generator=gen,
                    preconditioning="none",
                ).to(D.device)
                encoder.mul_(coordinate_mask)
                if cfg.identical_site_init:
                    encoder.copy_(encoder[:1].expand_as(encoder))
            if cfg.site_rank is None:
                self.E = nn.Parameter(encoder)
                self.register_parameter("E_site", None)
                self.register_parameter("E_core", None)
                if cfg.encoder_constraint == "unit_latent":
                    project_latent_rows_(self.E.data)
            else:
                encoder_site, encoder_core = _site_axis_factorize(
                    encoder, cfg.site_rank
                )
                self.register_parameter("E", None)
                self.E_site = nn.Parameter(encoder_site)
                self.E_core = nn.Parameter(encoder_core)
            self.register_parameter("log_gamma", None)
        else:
            self.register_parameter("E", None)
            self.register_parameter("E_site", None)
            self.register_parameter("E_core", None)
            self.log_gamma = nn.Parameter(
                torch.tensor(math.log(cfg.encoder_scale_init), device=D.device)
            )
        if cfg.encoder_bias:
            self.a = nn.Parameter(
                torch.zeros(cfg.n_blocks, cfg.block_dim, device=D.device)
            )
        else:
            self.register_parameter("a", None)
        if cfg.code_activation == "group_soft_threshold":
            # Positive learned thresholds.  Paper and release bridges choose
            # the transform and initializer explicitly in their resolved
            # decision records instead of inheriting an implementation default.
            threshold_shape = (
                () if cfg.group_threshold_scope == "shared_scalar" else (cfg.n_blocks,)
            )
            if cfg.group_threshold_raw_init is not None:
                raw_threshold = cfg.group_threshold_raw_init
            elif cfg.group_threshold_parameterization == "softplus":
                effective = cfg.group_threshold_effective_init
                raw_threshold = (
                    effective if effective > 20.0 else math.log(math.expm1(effective))
                )
            else:
                raw_threshold = math.log(cfg.group_threshold_effective_init)
            self.log_threshold = nn.Parameter(
                torch.full(threshold_shape, raw_threshold, device=D.device)
            )
        else:
            self.register_parameter("log_threshold", None)
        self.c = nn.Parameter(
            torch.zeros(cfg.n_sites, cfg.d_model, device=D.device),
            requires_grad=cfg.decoder_bias,
        )
        # Inference threshold theta: fit on the calibration split, frozen and
        # serialized with the codec. NaN until calibrated.
        self.register_buffer("theta", torch.tensor(float("nan")))

    # -- core ops ---------------------------------------------------------

    def decoder_tensor(self) -> torch.Tensor:
        """Structured decoder used by every forward/objective path."""
        if self.cfg.site_rank is None:
            assert self.D is not None
            decoder = self.D
        else:
            assert self.D_site is not None and self.D_core is not None
            decoder = torch.einsum("sr,rgbd->sgbd", self.D_site, self.D_core)
        if self._has_padded_coordinates:
            return decoder * self.coordinate_mask
        return decoder

    def encoder_tensor(self) -> torch.Tensor:
        if self.cfg.site_rank is not None:
            assert self.E_site is not None and self.E_core is not None
            encoder = torch.einsum("sr,rgbd->sgbd", self.E_site, self.E_core)
        elif self.E is None:
            assert self.D is not None and self.log_gamma is not None
            encoder = self.D * self.log_gamma.exp()
        else:
            encoder = self.E
        if self._has_padded_coordinates:
            return encoder * self.coordinate_mask
        return encoder

    def _site_observation_mask(
        self,
        x: torch.Tensor,
        observed: torch.Tensor | None = None,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        cfg = self.cfg
        if observed is None:
            keep = torch.ones(
                x.shape[0], cfg.n_sites, 1, dtype=x.dtype, device=x.device
            )
        else:
            if observed.shape != (x.shape[0], cfg.n_sites):
                raise ValueError(
                    f"observed must have shape [{x.shape[0]},{cfg.n_sites}]"
                )
            keep = observed.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            if validate and not bool(keep.sum(dim=1).gt(0).all()):
                raise ValueError("every token must observe at least one site")
        if cfg.encoder_fusion == "source":
            source_observed = keep[:, cfg.source_site : cfg.source_site + 1].clone()
            keep.zero_()
            keep[:, cfg.source_site : cfg.source_site + 1] = source_observed
        if validate and not bool(keep.sum(dim=1).gt(0).all()):
            raise ValueError("encoder fusion has no observed source site for a token")
        return keep

    def _encode_with_tensor(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode with an already materialized structured encoder.

        A factorized or tied model otherwise rebuilt its full site tensor once
        for encoding and again in later forward stages.  Keeping the tensor
        local to one forward preserves autograd while avoiding those duplicate
        materializations.
        """
        cfg = self.cfg
        if x.ndim != 3 or x.shape[1:] != (cfg.n_sites, cfg.d_model):
            raise ValueError(
                f"expected [B,{cfg.n_sites},{cfg.d_model}], got {tuple(x.shape)}"
            )
        if self._has_padded_coordinates:
            x = x * self.coordinate_mask[:, 0, 0].to(x.dtype)
        if cfg.apply_decoder_bias_to_input:
            x = x - self.c.to(x.dtype).unsqueeze(0)
        # The dominant real-model path observes every site and uses literal
        # sum fusion.  Its observation mask is algebraically all ones, so do
        # not allocate it or stream the complete activation batch through a
        # no-op multiplication.
        if observed is None and cfg.encoder_fusion == "sum":
            keep = None
        else:
            keep = self._site_observation_mask(
                x,
                observed,
                validate=validate_observed,
            )
            x = x * keep
        W = encoder.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)
        per_site = torch.bmm(x.transpose(0, 1), W.transpose(1, 2))
        z = per_site.sum(dim=0)
        if cfg.encoder_fusion == "mean":
            assert keep is not None
            z = z / keep.sum(dim=1)
        elif cfg.encoder_fusion == "availability_rescaled_sum":
            assert keep is not None
            z = z * (cfg.n_sites / keep.sum(dim=1))
        z = z.view(-1, cfg.n_blocks, cfg.block_dim)
        if self.a is not None:
            z = z + self.a
        if cfg.code_activation == "relu":
            z = torch.relu(z)
        elif cfg.code_activation == "group_soft_threshold":
            norm = z.norm(dim=-1, keepdim=True)
            threshold = (
                torch.nn.functional.softplus(self.log_threshold)
                if self.cfg.group_threshold_parameterization == "softplus"
                else torch.exp(self.log_threshold)
            )
            threshold = (
                threshold.view(1, 1, 1)
                if threshold.ndim == 0
                else threshold.view(1, -1, 1)
            )
            z = z * torch.relu(1.0 - threshold / norm.clamp_min(1e-12))
        return z, keep

    def encode(
        self,
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, S, d] -> z: [B, G, b]."""
        z, _ = self._encode_with_tensor(x, self.encoder_tensor(), observed=observed)
        return z

    def scores(
        self,
        z: torch.Tensor,
        *,
        x: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
        _decoder: torch.Tensor | None = None,
        _observation_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Configured sparse-event score.

        Gram-constrained BSCs use the block norm, which is exactly isolated
        decoded energy.  Minder's scalar BatchTopK crosscoder instead uses a
        ReLU activation multiplied by the sum of its site decoder norms.
        """
        if self.cfg.selection_score in {"code_norm", "decoder_weighted"}:
            score = z.norm(dim=-1)
        if self.cfg.selection_score == "decoder_weighted":
            D = (
                self.decoder_tensor() if _decoder is None else _decoder
            ).float()
            per_site = D.pow(2).sum(dim=(2, 3)).sqrt()
            if self.cfg.decoder_norm_geometry == "sum_l2":
                site_norms = per_site.sum(dim=0)
            else:
                site_norms = per_site.pow(2).sum(dim=0).sqrt()
            score = score * site_norms.to(score.dtype).unsqueeze(0)
        elif self.cfg.selection_score == "decoded_energy":
            # Isolated decoded contribution energy. Unlike code_norm this is
            # invariant to reciprocal within-block encoder/decoder gauges, and
            # unlike decoder_weighted it preserves vector-coordinate geometry.
            D = (
                self.decoder_tensor() if _decoder is None else _decoder
            ).float()
            gram = torch.einsum("sgbd,sgcd->gbc", D, D)
            energy_sq = torch.einsum("ngb,gbc,ngc->ng", z.float(), gram, z.float())
            score = energy_sq.clamp_min(0).sqrt().to(z.dtype)
        elif self.cfg.selection_score == "isolated_loss_decrease":
            if x is None:
                raise ValueError(
                    "isolated_loss_decrease scoring requires the observed input"
                )
            if x.ndim != 3 or x.shape[1:] != (
                self.cfg.n_sites,
                self.cfg.d_model,
            ):
                raise ValueError(
                    "isolated_loss_decrease input shape does not match the model"
                )
            # Exact squared-error decrease from adding one block to the zero
            # reconstruction: 2 <r_O, y_g,O> - ||y_g,O||^2.  Work in decoder
            # coordinates and materialize at most [batch, groups, block], never
            # the prohibitive [batch, groups, sites, d_model] contribution
            # tensor. Hidden clean targets are excluded by ``keep``.
            keep = (
                self._site_observation_mask(x, observed)
                if _observation_keep is None
                else _observation_keep
            )
            coordinate_mask = self.coordinate_mask[:, 0, 0].to(x.dtype)
            residual = x * keep * coordinate_mask
            decoder = (
                self.decoder_tensor() if _decoder is None else _decoder
            ).float()
            code = z.float()
            projected = torch.zeros_like(code)
            energy_sq = torch.zeros(z.shape[:2], dtype=torch.float32, device=z.device)
            for site in range(self.cfg.n_sites):
                projected = projected + torch.einsum(
                    "nd,gbd->ngb",
                    residual[:, site].float(),
                    decoder[site],
                )
                site_gram = torch.einsum("gbd,gcd->gbc", decoder[site], decoder[site])
                site_energy = torch.einsum("ngb,gbc,ngc->ng", code, site_gram, code)
                energy_sq = (
                    energy_sq + keep[:, site, 0].float().unsqueeze(1) * site_energy
                )
            score = (2.0 * (projected * code).sum(dim=-1) - energy_sq).to(z.dtype)
        return score

    def _select_scores(
        self,
        p: torch.Tensor,
        *,
        mode: str,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mode == "topk":
            if self.cfg.selection == "batch_topk":
                return batch_topk_mask(p, self.cfg.k)
            if self.cfg.selection == "token_topk":
                return token_topk_mask(p, self.cfg.k)
            if self.cfg.selection == "threshold":
                if torch.isnan(self.theta):
                    raise RuntimeError("training threshold not calibrated")
                return p > self.theta
            if z is None:
                raise ValueError(
                    "dense training selection requires post-activation codes"
                )
            # Dense ReLU and learned group-threshold methods train on every
            # nonzero post-activation code.  Their endpoint ranking score is
            # deliberately independent: in particular, a signed isolated-loss
            # score may be negative without silently adding an undeclared hard
            # score>0 gate to learned group shrinkage.
            return z.norm(dim=-1) > 0
        if mode == "threshold":
            if torch.isnan(self.theta):
                raise RuntimeError("inference threshold not calibrated")
            return p > self.theta
        raise ValueError(f"unknown selection mode {mode!r}")

    def select(
        self,
        z: torch.Tensor,
        *,
        mode: str = "topk",
        x: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the active-group mask ``[B, G]``.

        ``topk`` means the declared training rule: token/BatchTopK for hard
        selectors and nonzero post-activation support for dense ReLU or learned
        group shrinkage. ``threshold`` always means inference against the
        frozen calibrated endpoint-score threshold.
        """
        return self._select_scores(
            self.scores(z, x=x, observed=observed),
            mode=mode,
            z=z,
        )

    def decode(
        self,
        z_selected: torch.Tensor,
        *,
        add_bias: bool = True,
        _decoder: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z_selected: [B, G, b] -> xhat: [B, S, d]. AuxK residual
        reconstruction decodes without the bias (add_bias=False)."""
        cfg = self.cfg
        decoder = self.decoder_tensor() if _decoder is None else _decoder
        Wd = decoder.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)
        flat = z_selected.reshape(-1, cfg.n_latents)
        # [B, G*b] @ [S, G*b, d] broadcasts to [S, B, d].
        xhat = torch.matmul(flat, Wd)
        if add_bias and cfg.decoder_bias:
            xhat = xhat + self.c.unsqueeze(1)
        xhat = xhat.transpose(0, 1)
        if self._has_padded_coordinates:
            xhat = xhat * self.coordinate_mask[:, 0, 0].to(xhat.dtype)
        return xhat

    def forward(
        self,
        x: torch.Tensor,
        *,
        mode: str = "topk",
        observed: torch.Tensor | None = None,
    ) -> BSCOutput:
        out, _, _ = self.forward_with_materialized(
            x,
            mode=mode,
            observed=observed,
        )
        return out

    def forward_with_materialized(
        self,
        x: torch.Tensor,
        *,
        mode: str = "topk",
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        _decoder: torch.Tensor | None = None,
        _encoder: torch.Tensor | None = None,
    ) -> tuple[BSCOutput, torch.Tensor, torch.Tensor]:
        """Forward plus the exact structured weights used by that pass.

        The trainer consumes these tensors for regularizers, avoiding another
        full site-axis materialization.  Public callers can continue to use
        :meth:`forward` and receive the unchanged ``BSCOutput`` contract.
        """
        decoder = self.decoder_tensor() if _decoder is None else _decoder
        if _encoder is None:
            if self.cfg.encoder_mode == "tied":
                assert self.log_gamma is not None
                encoder = decoder * self.log_gamma.exp()
            else:
                encoder = self.encoder_tensor()
        else:
            encoder = _encoder
        z, keep = self._encode_with_tensor(
            x,
            encoder,
            observed=observed,
            validate_observed=validate_observed,
        )
        scores = self.scores(
            z,
            x=x,
            observed=observed,
            _decoder=decoder,
            _observation_keep=keep,
        )
        mask = self._select_scores(scores, mode=mode, z=z)
        z_selected = z * mask.unsqueeze(-1)
        xhat = self.decode(z_selected, _decoder=decoder)
        return BSCOutput(xhat, z, z_selected, scores, mask), decoder, encoder

    # -- init calibration --------------------------------------------------

    @torch.no_grad()
    def initialize_decoder_bias_(
        self,
        x: torch.Tensor,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        """Apply the declared data-derived reconstruction-bias initializer.

        ``geometric_median`` uses deterministic Weiszfeld iterations per site
        on the supplied initialization split.  The caller is responsible for
        binding that split and its row digest in the run manifest.
        """

        if not self.cfg.decoder_bias:
            self.c.zero_()
            return
        if x.ndim != 3 or x.shape[1:] != (self.cfg.n_sites, self.cfg.d_model):
            raise ValueError(
                f"expected [B,{self.cfg.n_sites},{self.cfg.d_model}], got {tuple(x.shape)}"
            )
        values = x.to(device=self.c.device, dtype=torch.float32)
        for site, width in enumerate(self.cfg.site_dims):
            sample = values[:, site, :width]
            if self.cfg.decoder_bias_init == "zero":
                center = torch.zeros(width, device=sample.device)
            elif self.cfg.decoder_bias_init == "data_mean":
                center = sample.mean(dim=0)
            else:
                center = sample.mean(dim=0)
                for _ in range(max_iterations):
                    distance = (sample - center).norm(dim=1).clamp_min(1e-12)
                    updated = (sample / distance[:, None]).sum(dim=0) / (
                        1.0 / distance
                    ).sum()
                    if (updated - center).norm() <= tolerance * (1.0 + center.norm()):
                        center = updated
                        break
                    center = updated
            self.c[site, :width].copy_(center)
            self.c[site, width:].zero_()

    @torch.no_grad()
    def calibrate_encoder_scale_(
        self, x: torch.Tensor, *, per_block: bool = True, eps: float = 1e-12
    ) -> None:
        """Scale the encoder so initial selection scores are comparable
        across blocks. Fel App. D prescribes
        transpose-tied init with encoder scale calibration in broad terms;
        the per-block median equalization here is BSC-specific. Preserves
        the global scale the tied Gram-constrained init already gives.
        """
        p = self.scores(self.encode(x), x=x)  # [B, G]
        if self.cfg.encoder_mode == "tied":
            # Fel Grassmannian has one learned gamma, not per-block scales.
            return
        mean_p = p.mean(dim=0).clamp_min(eps)  # [G]
        if self.cfg.site_rank is not None and per_block:
            raise ValueError(
                "per-block encoder calibration is not representable by a "
                "site-axis factorization; use the global calibration"
            )
        if per_block:
            scale = mean_p.median() / mean_p  # [G]
            assert self.E is not None
            self.E.mul_(scale.view(1, -1, 1, 1))
        else:
            self.scale_encoder_(float(mean_p.median() / mean_p.mean()))
        if self.E is not None:
            self.E.mul_(self.coordinate_mask)

    @torch.no_grad()
    def scale_encoder_(self, multiplier: float) -> None:
        """Apply a global encoder scale in either parameterization."""
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("encoder multiplier must be finite and positive")
        if self.cfg.encoder_mode == "tied":
            assert self.log_gamma is not None
            self.log_gamma.add_(math.log(multiplier))
        elif self.cfg.site_rank is not None:
            assert self.E_core is not None
            self.E_core.mul_(multiplier)
        else:
            assert self.E is not None
            self.E.mul_(multiplier)
            self.E.mul_(self.coordinate_mask)

    @property
    def parameter_device(self) -> torch.device:
        if self.D is not None:
            return self.D.device
        assert self.D_core is not None
        return self.D_core.device

    @property
    def parameter_dtype(self) -> torch.dtype:
        if self.D is not None:
            return self.D.dtype
        assert self.D_core is not None
        return self.D_core.dtype

    @torch.no_grad()
    def project_decoder_(self) -> int:
        """Apply the configured decoder constraint after an optimizer step."""
        if self.cfg.site_rank is not None:
            # BSCConfig permits only the free-decoder factorized arm.  There
            # is no exact factor-space equivalent of the full-tensor Stiefel
            # or norm projections, so silently materializing/projecting and
            # refactorizing would change the optimizer state and is forbidden.
            bad = 0
        else:
            assert self.D is not None
            if self._has_padded_coordinates:
                self.D.data.mul_(self.coordinate_mask)
            if self.cfg.decoder_constraint == "gram":
                bad = retract_(self.D.data, eig_floor=self.cfg.eig_floor)
            elif self.cfg.decoder_constraint == "qr":
                bad = qr_retract_(self.D.data)
            elif self.cfg.decoder_constraint == "frobenius":
                bad = project_block_frobenius_(self.D.data)
            elif self.cfg.decoder_constraint == "unit_frobenius":
                bad = normalize_block_frobenius_(self.D.data)
            elif self.cfg.decoder_constraint == "unit_latent":
                bad = project_latent_rows_(self.D.data)
            else:
                bad = 0
            if self._has_padded_coordinates:
                self.D.data.mul_(self.coordinate_mask)
            if self.E is not None:
                if self._has_padded_coordinates:
                    self.E.data.mul_(self.coordinate_mask)
                if self.cfg.encoder_constraint == "unit_latent":
                    project_latent_rows_(self.E.data)
                    if self._has_padded_coordinates:
                        self.E.data.mul_(self.coordinate_mask)
        if not self.cfg.decoder_bias:
            self.c.data.zero_()
        elif self._has_padded_coordinates:
            self.c.data.mul_(self.coordinate_mask[:, 0, 0])
        return bad

    @torch.no_grad()
    def fit_threshold_(
        self, batches, target_avg_blocks: float, *, method: str = "exact"
    ) -> float:
        """Fit the frozen inference threshold theta on the calibration
        split so the average active-block count hits the preregistered
        target: mean count = G * P(p > theta), so theta is the
        (1 - target/G) quantile of the pooled score distribution.

        method="exact": kthvalue over host-accumulated scores (not
        torch.quantile, which caps at ~16M elements). At G=4096 the
        pooled score matrix is already ~8.6 GB at a modest pilot slice and
        cannot sit next to the model on a 24 GB card; the scalar pilot arm
        also exhausted 61 GB host RAM. Kept as the validation reference.

        method="streaming": bounded-memory log-histogram quantile — the
        real-campaign path for every declared calibration split at large
        dictionary width, including any G >= 8192 configuration. Nonnegative
        scores use one positive log grid; isolated loss decrease uses symmetric
        signed log grids with an explicit zero boundary. Both are deterministic
        (batch-order independent, int64 counts); resolution ~3e-5 relative in
        theta away from zero. Validation gate vs exact:
        |Δ avg-blocks| <= 0.1.
        """
        q = 1.0 - target_avg_blocks / self.cfg.n_blocks
        if method == "streaming":
            histogram_type = (
                SignedStreamingScoreQuantile
                if self.cfg.selection_score == "isolated_loss_decrease"
                else StreamingScoreQuantile
            )
            hist = histogram_type(device=self.parameter_device)
            for x in batches:
                value = x.to(self.parameter_device, self.parameter_dtype)
                hist.update(self.scores(self.encode(value), x=value))
            theta = hist.quantile(q)
        elif method == "exact":
            score_batches = []
            for x in batches:
                value = x.to(self.parameter_device, self.parameter_dtype)
                score_batches.append(
                    self.scores(self.encode(value), x=value).flatten().float().cpu()
                )
            scores = torch.cat(score_batches)
            n = scores.numel()
            idx = min(max(int(round(q * n)), 1), n)
            theta = float(scores.kthvalue(idx).values)
        else:
            raise ValueError("method must be 'exact' or 'streaming'")
        self.theta.fill_(theta)
        return theta


def bsc_loss(
    out: BSCOutput,
    x: torch.Tensor,
    model: BlockCrosscoder,
    observation_mask: torch.Tensor | None = None,
    *,
    decoder: torch.Tensor | None = None,
    encoder: torch.Tensor | None = None,
    validate_observation_mask: bool = True,
) -> dict[str, torch.Tensor]:
    """Pinned reductions so lambda and alpha transfer across configs.

    L_rec = mean over tokens, sites, dims of the squared residual.
    L_aux  lives in the trainer (AuxK needs cross-step frequency state).

    Reductions run in fp32 regardless of forward dtype — a bf16 mean over
    millions of elements loses the precision the comparisons need.
    """
    cfg = model.cfg
    coord = model.coordinate_mask[:, 0, 0].to(x.device)
    all_observed = observation_mask is None
    if observation_mask is None:
        observed = None
    else:
        if observation_mask.shape != (x.shape[0], cfg.n_sites):
            raise ValueError(
                f"observation_mask must have shape [{x.shape[0]}, {cfg.n_sites}]"
            )
        observed = observation_mask.to(device=x.device, dtype=torch.bool)
        if validate_observation_mask and not bool(observed.any()):
            raise ValueError("observation_mask excludes the entire batch")

    def reconstruction(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        residual = pred.float() - target.float()
        if model._has_padded_coordinates:
            residual = residual * coord
        if all_observed:
            if cfg.reconstruction_loss == "mean_l2":
                return residual.norm(dim=-1).sum() / (target.shape[0] * cfg.n_sites)
            if cfg.reconstruction_loss == "mean_l1":
                return residual.abs().sum(dim=-1).sum() / (
                    target.shape[0] * cfg.n_sites
                )
            if cfg.reconstruction_loss == "squared_l2":
                return residual.pow(2).sum() / target.shape[0]
            denominator = target.shape[0] * sum(cfg.site_dims)
            return residual.pow(2).sum() / denominator
        assert observed is not None
        site_mask = observed.to(torch.float32).unsqueeze(-1)
        residual = residual * site_mask
        if cfg.reconstruction_loss == "mean_l2":
            # Released Minder code minimizes the mean Euclidean norm per
            # example/site, not the squared objective written in the papers.
            return residual.norm(dim=-1).sum() / observed.sum()
        if cfg.reconstruction_loss == "mean_l1":
            return residual.abs().sum(dim=-1).sum() / observed.sum()
        if cfg.reconstruction_loss == "squared_l2":
            return residual.pow(2).sum() / target.shape[0]
        denominator = (
            (observed.to(torch.float32).unsqueeze(-1) * coord).sum().clamp_min(1.0)
        )
        return residual.pow(2).sum() / denominator

    l_rec = reconstruction(x, out.xhat)
    total = l_rec
    parts: dict[str, torch.Tensor] = {"rec": l_rec}
    if cfg.lambda_regularizer > 0 and cfg.regularizer != "none":
        D = model.decoder_tensor() if decoder is None else decoder
        if cfg.regularizer == "map_nuclear":
            E = model.encoder_tensor() if encoder is None else encoder
            reg = map_nuclear_penalty(D, E, eps=cfg.sv_eps)
            if cfg.map_nuclear_reduction == "sum_blocks":
                reg = reg * cfg.n_blocks * cfg.block_dim
        elif cfg.regularizer == "decoder_nuclear":
            reg = decoder_nuclear_penalty(D, eps=cfg.sv_eps)
        elif cfg.regularizer == "crosscoder_l1":
            # Anthropic's sitewise decoder-norm-weighted activation L1.
            # For the paper-faithful bridge b=1; the Frobenius extension is
            # well-defined for blocks but is not claimed as their objective.
            per_site = D.float().pow(2).sum(dim=(2, 3)).sqrt()
            if cfg.decoder_norm_geometry == "sum_l2":
                site_cost = per_site.sum(dim=0)
            else:
                site_cost = per_site.pow(2).sum(dim=0).sqrt()
            reg = (out.scores.float() * site_cost.unsqueeze(0)).sum(dim=1).mean()
        elif cfg.regularizer == "group_l21":
            # Fel Group-Lasso BSF: mean over examples of the sum of activated
            # block norms.  The learned group soft threshold lives in encode.
            reg = out.z.float().norm(dim=-1).sum(dim=1).mean()
            if (
                cfg.group_lasso_target_k is not None
                and float(out.mask.float().sum(dim=1).mean())
                <= cfg.group_lasso_target_k
            ):
                reg = reg * 0.0
        else:  # guarded by BSCConfig
            raise AssertionError(cfg.regularizer)
        parts["regularizer"] = reg
        total = total + cfg.lambda_regularizer * reg
    parts["total"] = total
    return parts
