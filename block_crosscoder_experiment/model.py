"""Block-sparse crosscoder: G blocks of width b, one shared code across sites.

All model mathematics lives in the store's declared per-site coordinates;
the model is agnostic to whether they are raw, scalar-normalized,
token-LayerNorm, or whitened. Inputs are batches x: [B, S, d].

Unfactorized parameter stacks are [S, G, b, d].  A declared novel arm may
instead use a low-rank site-axis factorization
``W[s,g,b,d] = sum_r A[s,r] B[r,g,b,d]`` for both encoder and decoder. The
logical core is stored in contraction-ready physical layouts: encoder
``[R*d,G*b]`` and decoder ``[G*b,R*d]``. Its canonical forward contracts
those parameters directly in rank space;
``encoder_tensor`` and ``decoder_tensor`` remain explicit materialized-oracle
surfaces.  Parameters are fp32 masters, and the bf16 forward copy is the
trainer's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache
from typing import NamedTuple

import torch
from torch import nn

from .gram import (
    _cholesky_qr_retract_count_tensor_,
    _normalize_block_frobenius_count_tensor_,
    _project_block_frobenius_count_tensor_,
    _project_latent_rows_count_tensor_,
    _qr_retract_count_tensor_,
    _retract_count_tensor_,
    decoder_nuclear_penalty,
    factorized_decoder_nuclear_penalty,
    factorized_map_nuclear_penalty,
    gram_residual,
    init_decoder_stack,
    map_nuclear_penalty,
)
from .runtime_limits import (
    CODE_NORM_CUDA_IMPLEMENTATION,
    CODE_NORM_IMPLEMENTATIONS,
    CUDA_CODE_NORM_MIN_OUTPUTS,
    DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
    DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    DECODER_RETRACTION_IMPLEMENTATIONS,
    DECODER_RETRACTION_NOT_APPLICABLE,
    DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
    DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
    CUDA_SPARSE_DECODE_DENSITY_DENOMINATOR,
    CUDA_SPARSE_DECODE_MIN_BATCH,
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_IMPLEMENTATIONS,
    DECODED_ENERGY_MASTER_GRAM_RESIDUAL_MAX,
    DECODED_ENERGY_POSTCAST_GRAM_RESIDUAL_MAX,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
    FACTORIZED_EXECUTION_IMPLEMENTATIONS,
    FACTORIZED_EXECUTION_NOT_APPLICABLE,
    ISOLATED_LOSS_EXACT_IMPLEMENTATION,
    ISOLATED_LOSS_IMPLEMENTATIONS,
    ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
    MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    MAP_NUCLEAR_IMPLEMENTATIONS,
    SPARSE_DECODE_CUDA_IMPLEMENTATION,
    SPARSE_DECODE_IMPLEMENTATIONS,
    decoded_energy_code_norm_eligible,
    isolated_loss_mapped_eligible,
)

__all__ = [
    "BSCConfig",
    "BSCOutput",
    "BSCSelection",
    "BlockCrosscoder",
    "ISOLATED_LOSS_EXACT_IMPLEMENTATION",
    "ISOLATED_LOSS_MAPPED_IMPLEMENTATION",
    "SignedStreamingScoreQuantile",
    "StreamingScoreQuantile",
    "batch_topk_mask",
    "token_topk_mask",
    "bsc_reconstruction_loss",
    "bsc_loss",
    "isolated_loss_mapped_eligible",
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
        bin_idx = int(
            torch.searchsorted(
                cum,
                torch.tensor(target, dtype=cum.dtype, device=cum.device),
                right=False,
            )
        )
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
            torch.searchsorted(
                cumulative,
                torch.tensor(
                    target,
                    dtype=cumulative.dtype,
                    device=cumulative.device,
                ),
                right=False,
            )
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
    # The scientific norm is unchanged.  This identity binds only the
    # bitwise-equivalent guarded CUDA execution schedule.
    code_norm_implementation: str = CODE_NORM_CUDA_IMPLEMENTATION
    # The scientific score name remains decoded_energy.  This separate,
    # serialized implementation identity permits the Stiefel equality
    # ||D_g^T z_g||_sites = ||z_g|| only under the guarded carrier below.
    decoded_energy_implementation: str = DECODED_ENERGY_EXACT_IMPLEMENTATION
    # Isolated loss decrease remains the signed scientific score.  The mapped
    # free-decoder implementation changes only its contraction schedule and is
    # serialized separately so orchestration cannot select it implicitly.
    isolated_loss_decrease_implementation: str = ISOLATED_LOSS_EXACT_IMPLEMENTATION
    # Resolved into an explicit serialized identity in ``__post_init__``.  A
    # None constructor default preserves ergonomic direct construction without
    # permitting checkpoints or orchestration to omit the realized algorithm.
    decoder_retraction_implementation: str | None = None
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
    # The canonical factorized forward remains compact through encode, score,
    # and decode.  The materialized implementation is an explicit release
    # oracle, never an ambient device-dependent fallback.
    factorized_execution_implementation: str | None = None
    sparse_decode_implementation: str = SPARSE_DECODE_CUDA_IMPLEMENTATION
    map_nuclear_implementation: str = MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION

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
        if self.decoded_energy_implementation not in DECODED_ENERGY_IMPLEMENTATIONS:
            raise ValueError(
                "decoded_energy_implementation must be exact_decoder_gram_v1 "
                "or stiefel_code_norm_bounded_v1"
            )
        if (
            self.isolated_loss_decrease_implementation
            not in ISOLATED_LOSS_IMPLEMENTATIONS
        ):
            raise ValueError(
                "isolated_loss_decrease_implementation must be "
                "exact_site_gram_quadratic_v1 or "
                "mapped_free_decoder_quadratic_v1"
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
        if self.decoder_retraction_implementation is None:
            self.decoder_retraction_implementation = (
                DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
                if self.decoder_constraint == "qr"
                else DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION
                if self.decoder_constraint == "gram"
                else DECODER_RETRACTION_NOT_APPLICABLE
            )
        if (
            self.decoder_retraction_implementation
            not in DECODER_RETRACTION_IMPLEMENTATIONS
        ):
            raise ValueError("unknown decoder_retraction_implementation")
        if self.decoder_constraint == "qr":
            if self.decoder_retraction_implementation not in {
                DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
                DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
            }:
                raise ValueError(
                    "QR decoder constraint requires a QR retraction implementation"
                )
        elif self.decoder_constraint == "gram":
            if self.decoder_retraction_implementation not in {
                DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
                DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
            }:
                raise ValueError(
                    "Gram decoder constraint requires symmetric-polar retraction"
                )
        elif (
            self.decoder_retraction_implementation != DECODER_RETRACTION_NOT_APPLICABLE
        ):
            raise ValueError(
                "non-Stiefel decoder constraint requires not-applicable retraction"
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
        if (
            self.decoder_constraint in {"gram", "qr"}
            and sum(self.site_dims) < self.block_dim
        ):
            raise ValueError(
                "Stiefel decoder requires at least block_dim active coordinates"
            )
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
        if (
            self.isolated_loss_decrease_implementation
            == ISOLATED_LOSS_MAPPED_IMPLEMENTATION
            and not isolated_loss_mapped_eligible(
                selection_score=self.selection_score,
                decoder_constraint=self.decoder_constraint,
                decoder_bias=self.decoder_bias,
                reconstruction_loss=self.reconstruction_loss,
            )
        ):
            raise ValueError(
                "mapped isolated-loss decrease requires "
                "isolated_loss_decrease scoring and a free decoder"
            )
        if self.factorized_execution_implementation is None:
            self.factorized_execution_implementation = (
                FACTORIZED_EXECUTION_NOT_APPLICABLE
                if self.site_rank is None
                else (
                    FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION
                    if self.site_rank in {1, 2}
                    and self.regularizer in {"map_nuclear", "decoder_nuclear"}
                    else FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
                )
            )
        if (
            self.factorized_execution_implementation
            not in FACTORIZED_EXECUTION_IMPLEMENTATIONS
        ):
            raise ValueError("unknown factorized_execution_implementation")
        if self.code_norm_implementation not in CODE_NORM_IMPLEMENTATIONS:
            raise ValueError("unknown code_norm_implementation")
        if self.sparse_decode_implementation not in SPARSE_DECODE_IMPLEMENTATIONS:
            raise ValueError("unknown sparse_decode_implementation")
        if self.map_nuclear_implementation not in MAP_NUCLEAR_IMPLEMENTATIONS:
            raise ValueError("unknown map_nuclear_implementation")
        if self.site_rank is None:
            if (
                self.factorized_execution_implementation
                != FACTORIZED_EXECUTION_NOT_APPLICABLE
            ):
                raise ValueError(
                    "unfactorized model requires not-applicable factorized execution"
                )
        else:
            if (
                self.factorized_execution_implementation
                == FACTORIZED_EXECUTION_NOT_APPLICABLE
            ):
                raise ValueError(
                    "site-axis factorization requires a factorized execution "
                    "implementation"
                )
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
            factor_regularizer_eligible = self.site_rank in {
                1,
                2,
            } and self.regularizer in {"map_nuclear", "decoder_nuclear"}
            if (
                self.factorized_execution_implementation
                == FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION
                and not factor_regularizer_eligible
            ):
                raise ValueError(
                    "factor-regularizer execution requires site rank 1/2 and "
                    "map_nuclear or decoder_nuclear"
                )
            if (
                self.factorized_execution_implementation
                == FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
                and factor_regularizer_eligible
            ):
                raise ValueError(
                    "rank-1/2 factorized nuclear regularization requires the v4 "
                    "factor-regularizer execution identity"
                )
        if (
            self.decoded_energy_implementation
            == DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
            and not decoded_energy_code_norm_eligible(
                selection_score=self.selection_score,
                decoder_constraint=self.decoder_constraint,
                training_selector=self.selection,
                site_rank=self.site_rank,
                # BSCConfig does not own optimizer cadence.  Trainer repeats
                # the complete check with its serialized TrainConfig.
                retract_every=1,
            )
        ):
            raise ValueError(
                "stiefel code-norm decoded energy requires decoded_energy, "
                "a Gram/QR unfactorized decoder, and a hard TopK selector"
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


class BSCSelection(NamedTuple):
    """Forward state through hard selection, before dense reconstruction."""

    z: torch.Tensor  # [B, G, b] pre-selection code
    z_selected: torch.Tensor  # [B, G, b] post-selection code (masked)
    scores: torch.Tensor  # [B, G] endpoint-selection scores
    mask: torch.Tensor  # [B, G] bool, selected blocks


class _ScoreGeometry(NamedTuple):
    """Frozen decoder-only score tensors bound to one materialization."""

    decoder_key: tuple[object, ...]
    decoder_weight: torch.Tensor | None
    decoder_gram: torch.Tensor | None
    site_decoder_gram: torch.Tensor | None
    isolated_loss_decoder_map: torch.Tensor | None
    isolated_loss_all_site_gram: torch.Tensor | None


class _FrozenEncoderSites(NamedTuple):
    """Batch-local frozen per-site encoder contractions."""

    input_key: tuple[object, ...]
    encoder_key: tuple[object, ...]
    preprocessing_key: tuple[object, ...]
    postprocess_key: tuple[object, ...]
    values: torch.Tensor  # [S, B, G*b]


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


def _pack_encoder_factor_core(core: torch.Tensor) -> torch.Tensor:
    """Store logical ``[R,G,b,d]`` in encode-GEMM order ``[R*d,G*b]``."""

    rank, groups, block_dim, d_model = core.shape
    return (
        core.permute(0, 3, 1, 2)
        .reshape(rank * d_model, groups * block_dim)
        .contiguous()
    )


def _pack_full_encoder(encoder: torch.Tensor) -> torch.Tensor:
    """Store logical ``[S,G,b,d]`` in encode-GEMM order ``[S*d,G*b]``."""

    sites, groups, block_dim, d_model = encoder.shape
    return (
        encoder.permute(0, 3, 1, 2)
        .reshape(sites * d_model, groups * block_dim)
        .contiguous()
    )


def _unpack_full_encoder(
    packed: torch.Tensor,
    *,
    sites: int,
    groups: int,
    block_dim: int,
    d_model: int,
) -> torch.Tensor:
    """Logical ``[S,G,b,d]`` view of a packed full encoder map."""

    return packed.view(sites, d_model, groups, block_dim).permute(0, 2, 3, 1)


def _eager_scaled_tied_encoder_map(
    decoder: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Scale and pack a tied decoder into the flattened encoder layout."""

    return _pack_full_encoder(decoder * gamma)


@cache
def _compiled_cuda_scaled_tied_encoder_map():
    return torch.compile(
        _eager_scaled_tied_encoder_map,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
    )


_CUDA_TIED_ENCODER_PACK_MIN_ELEMENTS = 1 << 20


def _pack_decoder_factor_core(core: torch.Tensor) -> torch.Tensor:
    """Store logical ``[R,G,b,d]`` in decode-GEMM order ``[G*b,R*d]``."""

    rank, groups, block_dim, d_model = core.shape
    return (
        core.permute(1, 2, 0, 3)
        .reshape(groups * block_dim, rank * d_model)
        .contiguous()
    )


# The smallest Phase-2 selector pool is 2048 optimizer tokens by 2048 blocks,
# four times this fixed gate. Keep smaller CUDA calls eager so calibration
# tails, Phase-1 cells, and tests do not pay Inductor compilation churn. The
# complete existing TopK/comparison/integer tie-finalization interior is
# compiled without changing the policy or operation order.
_CUDA_SELECTOR_FUSION_MIN_ELEMENTS = 1 << 20


def _eager_batch_topk_interior(
    scores: torch.Tensor,
    n_keep: int,
) -> torch.Tensor:
    B, G = scores.shape
    flat = scores.reshape(-1)
    cutoff = flat.topk(n_keep, sorted=False).values.min()
    strictly_above = flat > cutoff
    remaining = n_keep - strictly_above.sum()
    tied = flat == cutoff
    # Row-major flattening is the declared candidate index for BatchTopK.
    tie_rank = tied.to(torch.int32).cumsum(dim=0, dtype=torch.int32)
    mask = strictly_above | (tied & (tie_rank <= remaining))
    return mask.view(B, G)


def _eager_token_topk_interior(
    scores: torch.Tensor,
    n_keep: int,
) -> torch.Tensor:
    cutoff = (
        scores.topk(n_keep, dim=1, sorted=False).values.min(dim=1, keepdim=True).values
    )
    strictly_above = scores > cutoff
    remaining = n_keep - strictly_above.sum(dim=1, keepdim=True)
    tied = scores == cutoff
    # Within each token, block index is the declared candidate index.
    tie_rank = tied.to(torch.int32).cumsum(dim=1, dtype=torch.int32)
    return strictly_above | (tied & (tie_rank <= remaining))


@cache
def _compiled_cuda_batch_topk_interior():
    return torch.compile(
        _eager_batch_topk_interior,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
    )


@cache
def _compiled_cuda_token_topk_interior():
    return torch.compile(
        _eager_token_topk_interior,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
    )


def _batch_topk_interior(scores: torch.Tensor, n_keep: int) -> torch.Tensor:
    if scores.is_cuda and scores.numel() >= _CUDA_SELECTOR_FUSION_MIN_ELEMENTS:
        return _compiled_cuda_batch_topk_interior()(scores, n_keep)
    return _eager_batch_topk_interior(scores, n_keep)


def _token_topk_interior(scores: torch.Tensor, n_keep: int) -> torch.Tensor:
    if scores.is_cuda and scores.numel() >= _CUDA_SELECTOR_FUSION_MIN_ELEMENTS:
        return _compiled_cuda_token_topk_interior()(scores, n_keep)
    return _eager_token_topk_interior(scores, n_keep)


def _assert_finite_selector_scores(scores: torch.Tensor) -> None:
    finite = torch.isfinite(scores).all()
    if scores.is_cuda:
        # Keep the hot path asynchronous: the device-side assertion is
        # ordered before TopK without a scalar host synchronization.
        torch._assert_async(finite, "selector scores must be finite")
    elif not bool(finite):
        raise ValueError("selector scores must be finite")


def batch_topk_mask(scores: torch.Tensor, k: float) -> torch.Tensor:
    """BatchTopK over blocks: keep the top round(k*B) block-activations
    batch-wide. Fractional k sets the budget below one block per token —
    the under-provisioned regime the capture sweep probes.

    Per-token counts vary by design; only the batch total is pinned.
    scores: [B, G]  ->  bool mask [B, G]
    """
    _assert_finite_selector_scores(scores)
    B, G = scores.shape
    n_keep = min(int(round(k * B)), B * G)
    if n_keep == 0:
        return torch.zeros(B, G, dtype=torch.bool, device=scores.device)
    if n_keep == B * G:
        return torch.ones(B, G, dtype=torch.bool, device=scores.device)
    return _batch_topk_interior(scores, n_keep)


def token_topk_mask(scores: torch.Tensor, k: float) -> torch.Tensor:
    """Per-token block TopK used by the published BSF and SASA recipes."""
    _assert_finite_selector_scores(scores)
    B, G = scores.shape
    n_keep = min(max(int(round(k)), 0), G)
    if n_keep == 0:
        return torch.zeros(B, G, dtype=torch.bool, device=scores.device)
    if n_keep == G:
        return torch.ones(B, G, dtype=torch.bool, device=scores.device)
    return _token_topk_interior(scores, n_keep)


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
    E_core: nn.Parameter | None  # physical [R*d,G*b]
    D_site: nn.Parameter | None  # [S,R]
    D_core: nn.Parameter | None  # physical [G*b,R*d]
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
            _retract_count_tensor_(
                D,
                eig_floor=cfg.eig_floor,
                implementation=cfg.decoder_retraction_implementation,
            )
        elif cfg.decoder_constraint == "qr":
            if (
                cfg.decoder_retraction_implementation
                == DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
            ):
                _cholesky_qr_retract_count_tensor_(D)
            else:
                assert (
                    cfg.decoder_retraction_implementation
                    == DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION
                )
                _qr_retract_count_tensor_(D)
        elif cfg.decoder_constraint == "frobenius":
            _project_block_frobenius_count_tensor_(D)
        elif cfg.decoder_constraint == "unit_frobenius":
            _normalize_block_frobenius_count_tensor_(D)
        elif cfg.decoder_constraint == "unit_latent":
            _project_latent_rows_count_tensor_(D)
        D.mul_(coordinate_mask)
        if cfg.site_rank is None:
            self.D = nn.Parameter(D)
            self.register_parameter("D_site", None)
            self.register_parameter("D_core", None)
        else:
            decoder_site, decoder_core = _site_axis_factorize(D, cfg.site_rank)
            self.register_parameter("D", None)
            self.D_site = nn.Parameter(decoder_site)
            self.D_core = nn.Parameter(_pack_decoder_factor_core(decoder_core))
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
                if cfg.encoder_constraint == "unit_latent":
                    _project_latent_rows_count_tensor_(encoder)
                self.E = nn.Parameter(_pack_full_encoder(encoder))
                self.register_parameter("E_site", None)
                self.register_parameter("E_core", None)
            else:
                encoder_site, encoder_core = _site_axis_factorize(
                    encoder, cfg.site_rank
                )
                self.register_parameter("E", None)
                self.E_site = nn.Parameter(encoder_site)
                self.E_core = nn.Parameter(_pack_encoder_factor_core(encoder_core))
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
        self.register_buffer("theta", torch.tensor(float("nan"), device=D.device))
        # Device scalar validation synchronizes CUDA. Cache only the exact
        # tensor mutation/device identity which passed; state loading, fill_,
        # deepcopy, and module conversion all invalidate at least one key
        # component without adding any serialized state.
        self._validated_theta_key: tuple[object, ...] | None = None

    # -- core ops ---------------------------------------------------------

    def decoder_tensor(self) -> torch.Tensor:
        """Materialize the structured decoder for an explicit oracle/objective."""
        if self.cfg.site_rank is None:
            assert self.D is not None
            decoder = self.D
        else:
            assert self.D_site is not None and self.D_core is not None
            decoder = torch.einsum(
                "sr,rgbd->sgbd",
                self.D_site,
                self._decoder_factor_core_tensor(),
            )
        if self._has_padded_coordinates:
            return decoder * self.coordinate_mask
        return decoder

    @property
    def uses_stiefel_code_norm_decoded_energy(self) -> bool:
        return (
            self.cfg.decoded_energy_implementation
            == DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
        )

    @property
    def uses_direct_factorized_execution(self) -> bool:
        return (
            self.cfg.site_rank is not None
            and self.cfg.factorized_execution_implementation
            in {
                FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
                FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
            }
        )

    @property
    def uses_factorized_nuclear_regularizers(self) -> bool:
        return (
            self.cfg.factorized_execution_implementation
            == FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION
        )

    @property
    def uses_mapped_isolated_loss_decrease(self) -> bool:
        return (
            self.cfg.isolated_loss_decrease_implementation
            == ISOLATED_LOSS_MAPPED_IMPLEMENTATION
        )

    @torch.no_grad()
    def validate_decoded_energy_implementation(self) -> dict[str, object]:
        """Validate the effective decoder geometry used by the fast score.

        This is intentionally called at initialization, diagnostics,
        checkpoint boundaries, and trained-model load—not every step, where
        rebuilding the Gram would erase the specialization's measured win.
        """

        if not self.uses_stiefel_code_norm_decoded_energy:
            return {
                "applicable": False,
                "implementation": self.cfg.decoded_energy_implementation,
                "passed": True,
            }
        if self.parameter_dtype not in {torch.float32, torch.bfloat16}:
            raise RuntimeError(
                "stiefel code-norm decoded energy requires fp32 master or "
                "bf16 forward precision"
            )
        residual = float(gram_residual(self.decoder_tensor().float()).max())
        postcast = self.parameter_dtype == torch.bfloat16
        limit = (
            DECODED_ENERGY_POSTCAST_GRAM_RESIDUAL_MAX
            if postcast
            else DECODED_ENERGY_MASTER_GRAM_RESIDUAL_MAX
        )
        passed = math.isfinite(residual) and residual <= limit
        record: dict[str, object] = {
            "applicable": True,
            "implementation": self.cfg.decoded_energy_implementation,
            "geometry": "effective_concatenated_decoder_gram",
            "precision": "bf16_postcast" if postcast else "fp32_master",
            "gram_residual_max": residual,
            "gram_residual_maximum": limit,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                "stiefel code-norm decoded-energy invariant failed: "
                f"Gram residual {residual:.6g} exceeds {limit:.6g}"
            )
        return record

    def encoder_tensor(self) -> torch.Tensor:
        if self.cfg.site_rank is not None:
            assert self.E_site is not None and self.E_core is not None
            encoder = torch.einsum(
                "sr,rgbd->sgbd",
                self.E_site,
                self._encoder_factor_core_tensor(),
            )
        elif self.E is None:
            return self._tied_encoder_tensor(self.decoder_tensor())
        else:
            encoder = self._encoder_full_tensor()
        if self._has_padded_coordinates:
            return encoder * self.coordinate_mask
        return encoder

    def _encoder_full_tensor(self) -> torch.Tensor:
        """Logical ``[S,G,b,d]`` view of the physical full encoder map."""

        cfg = self.cfg
        assert cfg.site_rank is None and self.E is not None
        return _unpack_full_encoder(
            self.E,
            sites=cfg.n_sites,
            groups=cfg.n_blocks,
            block_dim=cfg.block_dim,
            d_model=cfg.d_model,
        )

    def _tied_encoder_tensor(self, decoder: torch.Tensor) -> torch.Tensor:
        """Build one tied encoder allocation directly in GEMM storage order."""

        cfg = self.cfg
        assert cfg.encoder_mode == "tied" and self.log_gamma is not None
        gamma = self.log_gamma.exp()
        if (
            decoder.is_cuda
            and decoder.dtype in {torch.float32, torch.bfloat16}
            and decoder.numel() >= _CUDA_TIED_ENCODER_PACK_MIN_ELEMENTS
        ):
            packed = _compiled_cuda_scaled_tied_encoder_map()(decoder, gamma)
            return _unpack_full_encoder(
                packed,
                sites=cfg.n_sites,
                groups=cfg.n_blocks,
                block_dim=cfg.block_dim,
                d_model=cfg.d_model,
            )
        return decoder * gamma

    def _factorized_flat_coordinate_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the common rank-major ``[R*d]`` padding mask."""

        assert self.cfg.site_rank is not None
        coordinate = self.coordinate_mask[0, 0, 0].to(dtype)
        return coordinate.expand(self.cfg.site_rank, -1).reshape(-1)

    def _encoder_factor_core_map(self) -> torch.Tensor:
        """Return physical encoder core ``[R*d,G*b]``, structurally masked."""

        assert self.E_core is not None
        if not self._has_padded_coordinates:
            return self.E_core
        return self.E_core * self._factorized_flat_coordinate_mask(
            self.E_core.dtype
        ).unsqueeze(1)

    def _decoder_factor_core_map(self) -> torch.Tensor:
        """Return physical decoder core ``[G*b,R*d]``, structurally masked."""

        assert self.D_core is not None
        if not self._has_padded_coordinates:
            return self.D_core
        return self.D_core * self._factorized_flat_coordinate_mask(
            self.D_core.dtype
        ).unsqueeze(0)

    def _encoder_factor_core_tensor(self, *, masked: bool = False) -> torch.Tensor:
        """Logical ``[R,G,b,d]`` adapter for explicit materialization."""

        cfg = self.cfg
        assert cfg.site_rank is not None and self.E_core is not None
        core = self._encoder_factor_core_map() if masked else self.E_core
        return core.view(
            cfg.site_rank,
            cfg.d_model,
            cfg.n_blocks,
            cfg.block_dim,
        ).permute(0, 2, 3, 1)

    def _decoder_factor_core_tensor(self, *, masked: bool = False) -> torch.Tensor:
        """Logical ``[R,G,b,d]`` adapter for scores and materialization."""

        cfg = self.cfg
        assert cfg.site_rank is not None and self.D_core is not None
        core = self._decoder_factor_core_map() if masked else self.D_core
        return core.view(
            cfg.n_blocks,
            cfg.block_dim,
            cfg.site_rank,
            cfg.d_model,
        ).permute(2, 0, 1, 3)

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

    @staticmethod
    def _tensor_binding_key(tensor: torch.Tensor) -> tuple[object, ...]:
        return (
            id(tensor),
            tensor._version,
            tensor.device,
            tensor.dtype,
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )

    def _encoder_preprocessing_key(self) -> tuple[object, ...]:
        cfg = self.cfg
        return (
            cfg.n_sites,
            cfg.n_blocks,
            cfg.block_dim,
            cfg.d_model,
            tuple(cfg.site_dims),
            self._has_padded_coordinates,
            cfg.apply_decoder_bias_to_input,
            (
                self._tensor_binding_key(self.coordinate_mask)
                if self._has_padded_coordinates
                else None
            ),
            (
                self._tensor_binding_key(self.c)
                if cfg.apply_decoder_bias_to_input
                else None
            ),
        )

    def _encoder_postprocess_key(self) -> tuple[object, ...]:
        cfg = self.cfg
        return (
            cfg.encoder_fusion,
            cfg.source_site,
            cfg.n_sites,
            cfg.n_blocks,
            cfg.block_dim,
            self._tensor_binding_key(self.a) if self.a is not None else None,
            cfg.code_activation,
            cfg.group_threshold_parameterization,
            cfg.group_threshold_scope,
            (
                self._tensor_binding_key(self.log_threshold)
                if self.log_threshold is not None
                else None
            ),
        )

    def _prepare_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if x.ndim != 3 or x.shape[1:] != (cfg.n_sites, cfg.d_model):
            raise ValueError(
                f"expected [B,{cfg.n_sites},{cfg.d_model}], got {tuple(x.shape)}"
            )
        if self._has_padded_coordinates:
            x = x * self.coordinate_mask[:, 0, 0].to(x.dtype)
        if cfg.apply_decoder_bias_to_input:
            x = x - self.c.to(x.dtype).unsqueeze(0)
        return x

    def _finish_encoded_preactivation(
        self,
        z: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply evidence fusion and encoder bias, but no code activation."""

        cfg = self.cfg
        if cfg.encoder_fusion == "mean":
            assert keep is not None
            z = z / keep.sum(dim=1)
        elif cfg.encoder_fusion == "availability_rescaled_sum":
            assert keep is not None
            z = z * (cfg.n_sites / keep.sum(dim=1))
        z = z.view(-1, cfg.n_blocks, cfg.block_dim)
        if self.a is not None:
            z = z + self.a
        return z

    def _activate_code(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the configured code nonlinearity to one exact preactivation."""

        cfg = self.cfg
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
        return z

    def _finish_encoded_sum(
        self,
        z: torch.Tensor,
        keep: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._activate_code(self._finish_encoded_preactivation(z, keep))

    def _encode_with_tensor(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        return_preactivation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode with an already materialized structured encoder.

        A factorized or tied model otherwise rebuilt its full site tensor once
        for encoding and again in later forward stages.  Keeping the tensor
        local to one forward preserves autograd while avoiding those duplicate
        materializations.
        """
        cfg = self.cfg
        x = self._prepare_encoder_input(x)
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
        # Fuse the site sum into one GEMM.  The former strided BMM produced
        # ``[S,B,G*b]`` and then reduced the site axis, a 640 MiB transient at
        # the Phase-2 B=8192/S=4/G*b=8192 geometry.  Flattening in the common
        # ``(site, coordinate)`` order contracts the same mathematical map
        # directly into ``[B,G*b]``.  This deliberately changes bf16 rounding
        # from one rounded result per site plus a sum to one GEMM reduction;
        # Phase 2 binds bf16 precision, not the superseded kernel order.
        W = encoder.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)
        W = W.transpose(1, 2).reshape(
            cfg.n_sites * cfg.d_model,
            cfg.n_latents,
        )
        z = x.reshape(x.shape[0], cfg.n_sites * cfg.d_model) @ W
        finish = (
            self._finish_encoded_preactivation
            if return_preactivation
            else self._finish_encoded_sum
        )
        return finish(z, keep), keep

    def _encode_factorized_direct(
        self,
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        return_preactivation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode through ``[S,R]`` and physical ``[R*d,G*b]`` directly."""

        cfg = self.cfg
        assert self.E_site is not None and self.E_core is not None
        assert cfg.site_rank is not None
        x = self._prepare_encoder_input(x)
        # The materialized encoder masks after optional bias centering.  Repeat
        # that structural mask here before site-to-rank contraction.
        if self._has_padded_coordinates:
            x = x * self.coordinate_mask[:, 0, 0].to(x.dtype)
        if observed is None and cfg.encoder_fusion == "sum":
            keep = None
        else:
            keep = self._site_observation_mask(
                x,
                observed,
                validate=validate_observed,
            )
            x = x * keep
        # [N,d,S] @ [S,R] -> [N,d,R], followed by one flattened rank-core
        # GEMM.  Peak structured weight storage is R*G*b*d rather than S*G*b*d.
        rank_input = torch.matmul(x.transpose(1, 2), self.E_site).transpose(1, 2)
        z = rank_input.reshape(x.shape[0], -1) @ self._encoder_factor_core_map()
        finish = (
            self._finish_encoded_preactivation
            if return_preactivation
            else self._finish_encoded_sum
        )
        return finish(z, keep), keep

    def _frozen_encoder_sites(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
    ) -> _FrozenEncoderSites:
        """Compute the reusable per-site contraction for one no-grad batch."""
        if torch.is_grad_enabled():
            raise RuntimeError("frozen encoder sites require no-grad execution")
        prepared = self._prepare_encoder_input(x)
        cfg = self.cfg
        W = encoder.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)
        values = torch.bmm(prepared.transpose(0, 1), W.transpose(1, 2))
        return _FrozenEncoderSites(
            self._tensor_binding_key(x),
            self._tensor_binding_key(encoder),
            self._encoder_preprocessing_key(),
            self._encoder_postprocess_key(),
            values,
        )

    def _encode_from_frozen_sites(
        self,
        x: torch.Tensor,
        encoder: torch.Tensor,
        frozen_sites: _FrozenEncoderSites,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse one observed view from an exact frozen site contraction."""
        if torch.is_grad_enabled():
            raise RuntimeError("frozen encoder sites require no-grad execution")
        cfg = self.cfg
        if frozen_sites.input_key != self._tensor_binding_key(x):
            raise ValueError("frozen encoder sites are not bound to this input")
        if frozen_sites.encoder_key != self._tensor_binding_key(encoder):
            raise ValueError("frozen encoder sites are not bound to this encoder")
        if frozen_sites.preprocessing_key != self._encoder_preprocessing_key():
            raise ValueError("frozen encoder input preprocessing changed")
        if frozen_sites.postprocess_key != self._encoder_postprocess_key():
            raise ValueError("frozen encoder postprocessing changed")
        expected_shape = (cfg.n_sites, x.shape[0], cfg.n_latents)
        if frozen_sites.values.shape != expected_shape:
            raise ValueError(
                "frozen encoder sites have invalid shape: "
                f"expected {expected_shape}, got {tuple(frozen_sites.values.shape)}"
            )
        # The all-view endpoint must use the same direct flattened contraction
        # as training, calibration, and codec selection.  Frozen per-site
        # contractions exist specifically to amortize the distinct partial
        # views below; their floating reduction order is deliberately not
        # substituted for the operational all-view kernel.
        if observed is None:
            return self._encode_with_tensor(
                x,
                encoder,
                observed=None,
                validate_observed=validate_observed,
            )
        keep = self._site_observation_mask(
            x,
            observed,
            validate=validate_observed,
        )
        per_site = frozen_sites.values * keep.transpose(0, 1)
        z = per_site.sum(dim=0)
        return self._finish_encoded_sum(z, keep), keep

    def encode(
        self,
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, S, d] -> z: [B, G, b]."""
        if self.uses_direct_factorized_execution:
            z, _ = self._encode_factorized_direct(x, observed=observed)
        else:
            z, _ = self._encode_with_tensor(
                x,
                self.encoder_tensor(),
                observed=observed,
            )
        return z

    def encode_preactivation(
        self,
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        _encoder: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the exact fused affine code before its nonlinearity.

        This primitive exposes the affine carrier for exact structured-weight
        diagnostics. An already materialized encoder may be supplied so those
        diagnostics do not rebuild structured weights.
        """

        if self.uses_direct_factorized_execution and _encoder is None:
            z, _ = self._encode_factorized_direct(
                x,
                observed=observed,
                validate_observed=validate_observed,
                return_preactivation=True,
            )
            return z
        encoder = self.encoder_tensor() if _encoder is None else _encoder
        z, _ = self._encode_with_tensor(
            x,
            encoder,
            observed=observed,
            validate_observed=validate_observed,
            return_preactivation=True,
        )
        return z

    @staticmethod
    def _decoder_binding_key(decoder: torch.Tensor) -> tuple[object, ...]:
        return (
            id(decoder),
            decoder._version,
            decoder.device,
            decoder.dtype,
            decoder.data_ptr(),
            tuple(decoder.shape),
            tuple(decoder.stride()),
        )

    def _frozen_score_geometry(self, decoder: torch.Tensor) -> _ScoreGeometry:
        """Precompute decoder-only score terms for a frozen no-grad pass."""

        if torch.is_grad_enabled():
            raise RuntimeError("frozen score geometry requires no-grad execution")
        decoder_weight = None
        decoder_gram = None
        site_decoder_gram = None
        isolated_loss_decoder_map = None
        isolated_loss_all_site_gram = None
        if self.cfg.selection_score == "decoder_weighted":
            D = decoder.float()
            per_site = D.pow(2).sum(dim=(2, 3)).sqrt()
            if self.cfg.decoder_norm_geometry == "sum_l2":
                decoder_weight = per_site.sum(dim=0)
            else:
                decoder_weight = per_site.pow(2).sum(dim=0).sqrt()
        elif (
            self.cfg.selection_score == "decoded_energy"
            and not self.uses_stiefel_code_norm_decoded_energy
        ):
            D = decoder.float()
            decoder_gram = torch.einsum("sgbd,sgcd->gbc", D, D)
        elif self.cfg.selection_score == "isolated_loss_decrease":
            D = decoder.float()
            # Preserve the original per-site contraction exactly rather than
            # relying on a different batched kernel to choose the same sums.
            site_decoder_gram = torch.stack(
                [
                    torch.einsum("gbd,gcd->gbc", D[site], D[site])
                    for site in range(self.cfg.n_sites)
                ]
            )
            if self.uses_mapped_isolated_loss_decrease:
                isolated_loss_decoder_map = D.permute(0, 3, 1, 2).reshape(
                    self.cfg.n_sites * self.cfg.d_model,
                    self.cfg.n_latents,
                )
                # Match the operational all-view contraction exactly.  A sum
                # of cached site Grams is algebraically equal but changes the
                # fp32 reduction tree and can move a threshold tie.
                isolated_loss_all_site_gram = torch.einsum("sgbd,sgcd->gbc", D, D)
        return _ScoreGeometry(
            self._decoder_binding_key(decoder),
            decoder_weight,
            decoder_gram,
            site_decoder_gram,
            isolated_loss_decoder_map,
            isolated_loss_all_site_gram,
        )

    def _exact_isolated_loss_decrease_scores(
        self,
        code: torch.Tensor,
        residual: torch.Tensor,
        keep: torch.Tensor,
        decoder: torch.Tensor,
        score_geometry: _ScoreGeometry | None,
    ) -> torch.Tensor:
        """Reference site-wise projection and three-factor quadratic."""

        projected = torch.zeros_like(code)
        energy_sq = torch.zeros(
            code.shape[:2],
            dtype=torch.float32,
            device=code.device,
        )
        for site in range(self.cfg.n_sites):
            projected.add_(
                torch.einsum(
                    "nd,gbd->ngb",
                    residual[:, site],
                    decoder[site],
                )
            )
            if score_geometry is None:
                site_gram = torch.einsum("gbd,gcd->gbc", decoder[site], decoder[site])
            else:
                assert score_geometry.site_decoder_gram is not None
                site_gram = score_geometry.site_decoder_gram[site]
            site_energy = torch.einsum("ngb,gbc,ngc->ng", code, site_gram, code)
            energy_sq.add_(keep[:, site, 0].float().unsqueeze(1) * site_energy)
        return 2.0 * (projected * code).sum(dim=-1) - energy_sq

    def _mapped_isolated_loss_decrease_scores(
        self,
        code: torch.Tensor,
        residual: torch.Tensor,
        keep: torch.Tensor,
        decoder: torch.Tensor,
        score_geometry: _ScoreGeometry | None,
        *,
        all_sites_observed: bool,
    ) -> torch.Tensor:
        """Mapped free-decoder quadratic without a candidate-output tensor.

        The linear term is one flattened decoder-transpose GEMM.  The
        quadratic first maps each code through its block Gram with BMM and
        then takes the coordinate dot product.  The all-observed path sums the
        site Grams before this map; partial/source views retain one mapped
        quadratic per site and weight it by the exact observation mask.
        """

        cfg = self.cfg
        if score_geometry is None:
            decoder_map = decoder.permute(0, 3, 1, 2).reshape(
                cfg.n_sites * cfg.d_model,
                cfg.n_latents,
            )
            if all_sites_observed:
                # Contract the site axis directly; allocating every G_s only
                # to reduce it would retain the reference path's avoidable
                # site-Gram stack in the dominant full-view training kernel.
                all_site_gram = torch.einsum("sgbd,sgcd->gbc", decoder, decoder)
                site_grams = None
            else:
                site_grams = torch.stack(
                    [
                        torch.einsum("gbd,gcd->gbc", decoder[site], decoder[site])
                        for site in range(cfg.n_sites)
                    ]
                )
                all_site_gram = None
        else:
            assert score_geometry.isolated_loss_decoder_map is not None
            assert score_geometry.site_decoder_gram is not None
            assert score_geometry.isolated_loss_all_site_gram is not None
            decoder_map = score_geometry.isolated_loss_decoder_map
            site_grams = score_geometry.site_decoder_gram
            all_site_gram = score_geometry.isolated_loss_all_site_gram

        projected = (
            residual.reshape(residual.shape[0], cfg.n_sites * cfg.d_model) @ decoder_map
        ).reshape_as(code)
        code_by_group = code.transpose(0, 1)
        if all_sites_observed:
            assert all_site_gram is not None
            mapped = torch.bmm(code_by_group, all_site_gram).transpose(0, 1)
            return ((2.0 * projected - mapped) * code).sum(dim=-1)

        assert site_grams is not None
        score = 2.0 * (projected * code).sum(dim=-1)
        for site in range(cfg.n_sites):
            mapped = torch.bmm(code_by_group, site_grams[site]).transpose(0, 1)
            site_energy = (mapped * code).sum(dim=-1)
            score.sub_(keep[:, site, 0].float().unsqueeze(1) * site_energy)
        return score

    def _factorized_decoder_weight(self) -> torch.Tensor:
        """Per-block decoder norm from the rank carrier in fp32."""

        assert self.D_site is not None and self.D_core is not None
        site = self.D_site.float()
        core = self._decoder_factor_core_tensor(masked=True).float()
        core_gram = torch.einsum("rgbd,tgbd->grt", core, core)
        norm_sq = torch.einsum("sr,grt,st->sg", site, core_gram, site)
        per_site = norm_sq.clamp_min(0).sqrt()
        if self.cfg.decoder_norm_geometry == "sum_l2":
            return per_site.sum(dim=0)
        return per_site.square().sum(dim=0).sqrt()

    def _factorized_decoder_gram(self) -> torch.Tensor:
        """All-site ``[G,b,b]`` Gram without a ``[S,G,b,d]`` decoder."""

        assert self.D_site is not None and self.D_core is not None
        site_metric = self.D_site.float().transpose(0, 1) @ self.D_site.float()
        core = self._decoder_factor_core_tensor(masked=True).float()
        return torch.einsum("rt,rgbd,tgcd->gbc", site_metric, core, core)

    def _factorized_isolated_loss_decrease_scores(
        self,
        code: torch.Tensor,
        residual: torch.Tensor,
        keep: torch.Tensor,
        *,
        all_sites_observed: bool,
    ) -> torch.Tensor:
        """Signed isolated-loss score entirely in the decoder rank carrier."""

        cfg = self.cfg
        assert cfg.site_rank is not None
        assert self.D_site is not None and self.D_core is not None
        site = self.D_site.float()
        core = self._decoder_factor_core_tensor(masked=True).float()

        # Linear term: observed residual sites -> rank coordinates -> blocks.
        rank_residual = torch.matmul(
            residual.transpose(1, 2),
            site,
        ).transpose(1, 2)
        projected = (
            rank_residual.reshape(residual.shape[0], -1)
            @ self._decoder_factor_core_map().float().transpose(0, 1)
        ).reshape_as(code)

        if all_sites_observed:
            gram = torch.einsum(
                "rt,rgbd,tgcd->gbc",
                site.transpose(0, 1) @ site,
                core,
                core,
            )
            energy_sq = torch.einsum("ngb,gbc,ngc->ng", code, gram, code)
        else:
            energy_sq = torch.zeros(
                code.shape[:2],
                dtype=torch.float32,
                device=code.device,
            )
            for site_index in range(cfg.n_sites):
                coefficients = site[site_index]
                site_gram = torch.einsum(
                    "r,t,rgbd,tgcd->gbc",
                    coefficients,
                    coefficients,
                    core,
                    core,
                )
                site_energy = torch.einsum(
                    "ngb,gbc,ngc->ng",
                    code,
                    site_gram,
                    code,
                )
                energy_sq.add_(
                    keep[:, site_index, 0].float().unsqueeze(1) * site_energy
                )
        return 2.0 * (projected * code).sum(dim=-1) - energy_sq

    def scores(
        self,
        z: torch.Tensor,
        *,
        x: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
        _decoder: torch.Tensor | None = None,
        _observation_keep: torch.Tensor | None = None,
        _score_geometry: _ScoreGeometry | None = None,
    ) -> torch.Tensor:
        """Configured sparse-event score.

        Gram-constrained BSCs use the block norm, which is exactly isolated
        decoded energy.  Minder's scalar BatchTopK crosscoder instead uses a
        ReLU activation multiplied by the sum of its site decoder norms.
        """
        decoder = _decoder
        fast_decoded_energy = self.uses_stiefel_code_norm_decoded_energy
        direct_factorized = (
            self.uses_direct_factorized_execution
            and decoder is None
            and _score_geometry is None
        )
        if (
            decoder is None
            and (
                _score_geometry is not None
                or (self.cfg.selection_score != "code_norm" and not fast_decoded_energy)
            )
            and not direct_factorized
        ):
            decoder = self.decoder_tensor()
        if _score_geometry is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("frozen score geometry cannot be used with grad")
            assert decoder is not None
            if _score_geometry.decoder_key != self._decoder_binding_key(decoder):
                raise ValueError("score geometry is not bound to this decoder tensor")
        if self.cfg.selection_score in {"code_norm", "decoder_weighted"} or (
            self.cfg.selection_score == "decoded_energy" and fast_decoded_energy
        ):
            if (
                self.cfg.code_norm_implementation == CODE_NORM_CUDA_IMPLEMENTATION
                and not torch.is_grad_enabled()
                and z.is_cuda
                and z.dtype == torch.bfloat16
                and z.ndim >= 1
                and z.shape[-1] == 4
                and z.is_contiguous()
                and z.numel() // 4 >= CUDA_CODE_NORM_MIN_OUTPUTS
            ):
                from .cuda_code_norm import cuda_code_norm4

                score = cuda_code_norm4(z)
            else:
                score = z.norm(dim=-1)
        if self.cfg.selection_score == "decoder_weighted":
            if _score_geometry is None:
                if direct_factorized:
                    site_norms = self._factorized_decoder_weight()
                else:
                    assert decoder is not None
                    D = decoder.float()
                    per_site = D.pow(2).sum(dim=(2, 3)).sqrt()
                    if self.cfg.decoder_norm_geometry == "sum_l2":
                        site_norms = per_site.sum(dim=0)
                    else:
                        site_norms = per_site.pow(2).sum(dim=0).sqrt()
            else:
                assert _score_geometry.decoder_weight is not None
                site_norms = _score_geometry.decoder_weight
            score = score * site_norms.to(score.dtype).unsqueeze(0)
        elif self.cfg.selection_score == "decoded_energy":
            # Isolated decoded contribution energy. Unlike code_norm this is
            # invariant to reciprocal within-block encoder/decoder gauges, and
            # unlike decoder_weighted it preserves vector-coordinate geometry.
            if fast_decoded_energy:
                # Under the effective concatenated Stiefel constraint this is
                # decoded contribution energy.  The fp32/bf16 deviation from
                # the exact Gram quadratic is explicitly release-bounded.
                pass
            elif _score_geometry is None:
                if direct_factorized:
                    gram = self._factorized_decoder_gram()
                else:
                    assert decoder is not None
                    D = decoder.float()
                    gram = torch.einsum("sgbd,sgcd->gbc", D, D)
            else:
                assert _score_geometry.decoder_gram is not None
                gram = _score_geometry.decoder_gram
            if not fast_decoded_energy:
                code = z.float()
                energy_sq = torch.einsum("ngb,gbc,ngc->ng", code, gram, code)
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
            code = z.float()
            residual = residual.float()
            if direct_factorized:
                score = self._factorized_isolated_loss_decrease_scores(
                    code,
                    residual,
                    keep,
                    all_sites_observed=(
                        observed is None and self.cfg.encoder_fusion != "source"
                    ),
                )
            elif self.uses_mapped_isolated_loss_decrease:
                assert decoder is not None
                score = self._mapped_isolated_loss_decrease_scores(
                    code,
                    residual,
                    keep,
                    decoder.float(),
                    _score_geometry,
                    all_sites_observed=(
                        observed is None and self.cfg.encoder_fusion != "source"
                    ),
                )
            else:
                assert decoder is not None
                score = self._exact_isolated_loss_decrease_scores(
                    code,
                    residual,
                    keep,
                    decoder.float(),
                    _score_geometry,
                )
            score = score.to(z.dtype)
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
                self._require_calibrated_threshold("training")
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
            self._require_calibrated_threshold("inference")
            return p > self.theta
        raise ValueError(f"unknown selection mode {mode!r}")

    def _require_calibrated_threshold(self, context: str) -> None:
        theta = self.theta
        key = (
            id(theta),
            theta._version,
            theta.device,
            theta.dtype,
            theta.data_ptr(),
        )
        if key == self._validated_theta_key:
            return
        if not bool(torch.isfinite(theta)):
            raise RuntimeError(f"{context} threshold not calibrated")
        self._validated_theta_key = key

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
        flat = z_selected.reshape(-1, cfg.n_latents)
        if _decoder is None and self.uses_direct_factorized_execution:
            assert cfg.site_rank is not None
            assert self.D_site is not None and self.D_core is not None
            rank_output = (flat @ self._decoder_factor_core_map()).reshape(
                z_selected.shape[0],
                cfg.site_rank,
                cfg.d_model,
            )
            # [B,d,R] @ [R,S] -> [B,d,S].
            xhat = torch.matmul(
                rank_output.transpose(1, 2),
                self.D_site.transpose(0, 1),
            ).transpose(1, 2)
            if add_bias and cfg.decoder_bias:
                xhat = xhat + self.c.unsqueeze(0)
        else:
            decoder = self.decoder_tensor() if _decoder is None else _decoder
            Wd = decoder.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)
            # [B, G*b] @ [S, G*b, d] broadcasts to [S, B, d].
            xhat = torch.matmul(flat, Wd)
            if add_bias and cfg.decoder_bias:
                xhat = xhat + self.c.unsqueeze(1)
            xhat = xhat.transpose(0, 1)
        if self._has_padded_coordinates:
            xhat = xhat * self.coordinate_mask[:, 0, 0].to(xhat.dtype)
        return xhat

    def _cuda_sparse_topk_decode_eligible(
        self,
        code: torch.Tensor,
        *,
        mode: str,
    ) -> bool:
        """Return the complete shape/device gate for rank-space sparse decode."""

        return self._cuda_sparse_topk_decode_shape_eligible(
            batch=code.shape[0],
            device=code.device,
            dtype=code.dtype,
            mode=mode,
        )

    def _cuda_sparse_topk_decode_shape_eligible(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: str,
    ) -> bool:
        if (
            mode != "topk"
            or self.cfg.selection not in {"batch_topk", "token_topk"}
            or device.type != "cuda"
            or dtype != torch.bfloat16
            or batch < CUDA_SPARSE_DECODE_MIN_BATCH
            or self.cfg.sparse_decode_implementation
            != SPARSE_DECODE_CUDA_IMPLEMENTATION
            or (
                self.cfg.site_rank is not None
                and not self.uses_direct_factorized_execution
            )
        ):
            return False
        groups = self.cfg.n_blocks
        selected = self._hard_topk_selected_count(batch)
        return (
            selected > 0
            and selected * CUDA_SPARSE_DECODE_DENSITY_DENOMINATOR <= batch * groups
        )

    def _hard_topk_selected_count(self, batch: int) -> int:
        groups = self.cfg.n_blocks
        if self.cfg.selection == "batch_topk":
            return min(int(round(self.cfg.k * batch)), batch * groups)
        if self.cfg.selection == "token_topk":
            return batch * min(max(int(round(self.cfg.k)), 0), groups)
        raise RuntimeError("hard TopK event count requires a hard TopK selector")

    def _decode_cuda_sparse_topk(
        self,
        code: torch.Tensor,
        mask: torch.Tensor,
        *,
        decoder: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode selected events without a dense zero-filled code tensor."""

        from .cuda_sparse_decode import cuda_sparse_topk_decode

        cfg = self.cfg
        selected_count = self._hard_topk_selected_count(code.shape[0])
        events_per_row = (
            min(max(int(round(cfg.k)), 0), cfg.n_blocks)
            if cfg.selection == "token_topk"
            else None
        )
        if self.uses_direct_factorized_execution:
            assert cfg.site_rank is not None
            assert self.D_site is not None and self.D_core is not None
            rank_output = cuda_sparse_topk_decode(
                code,
                mask,
                self._decoder_factor_core_map(),
                selected_count=selected_count,
                events_per_row=events_per_row,
            ).view(code.shape[0], cfg.site_rank, cfg.d_model)
            xhat = torch.matmul(
                rank_output.transpose(1, 2),
                self.D_site.transpose(0, 1),
            ).transpose(1, 2)
        else:
            if cfg.site_rank is not None:
                raise RuntimeError(
                    "materialized factorized reference refuses sparse CUDA decode"
                )
            if decoder is None:
                decoder = self.decoder_tensor()
            xhat = cuda_sparse_topk_decode(
                code,
                mask,
                decoder,
                selected_count=selected_count,
                events_per_row=events_per_row,
            )
        if cfg.decoder_bias:
            xhat = xhat + self.c.unsqueeze(0)
        if self._has_padded_coordinates:
            xhat = xhat * self.coordinate_mask[:, 0, 0].to(xhat.dtype)
        return xhat

    def _forward_cuda_sparse_topk_training(
        self,
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        score_grad: bool = False,
    ) -> torch.Tensor:
        """Trainer-only sparse forward returning only the reconstruction."""

        decoder: torch.Tensor | None
        if self.uses_direct_factorized_execution:
            z, keep = self._encode_factorized_direct(
                x,
                observed=observed,
                validate_observed=validate_observed,
            )
            decoder = None
        else:
            if self.cfg.site_rank is not None:
                raise RuntimeError(
                    "materialized factorized reference refuses sparse CUDA decode"
                )
            decoder = self.decoder_tensor()
            encoder = (
                self._tied_encoder_tensor(decoder)
                if self.cfg.encoder_mode == "tied"
                else self.encoder_tensor()
            )
            z, keep = self._encode_with_tensor(
                x,
                encoder,
                observed=observed,
                validate_observed=validate_observed,
            )
        if not self._cuda_sparse_topk_decode_eligible(z, mode="topk"):
            raise RuntimeError("CUDA sparse TopK decode is not eligible")
        with torch.set_grad_enabled(torch.is_grad_enabled() and score_grad):
            scores = self.scores(
                z,
                x=x,
                observed=observed,
                _decoder=decoder,
                _observation_keep=keep,
            )
        mask = self._select_scores(scores, mode="topk", z=z)
        return self._decode_cuda_sparse_topk(
            z,
            mask,
            decoder=decoder,
        )

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
        _score_geometry: _ScoreGeometry | None = None,
        _encoder_sites: _FrozenEncoderSites | None = None,
        _score_grad: bool = True,
    ) -> tuple[BSCOutput, torch.Tensor | None, torch.Tensor | None]:
        """Forward plus any materialized structured weights used by that pass.

        The direct factorized implementation returns ``None`` for both weights;
        regularizers may then materialize explicitly if their objective needs
        the full tensors. Public callers continue to receive the unchanged
        :class:`BSCOutput` contract from :meth:`forward`.
        """
        selection, decoder, encoder = self.select_with_materialized(
            x,
            mode=mode,
            observed=observed,
            validate_observed=validate_observed,
            _decoder=_decoder,
            _encoder=_encoder,
            _score_geometry=_score_geometry,
            _encoder_sites=_encoder_sites,
            _score_grad=_score_grad,
        )
        if self._cuda_sparse_topk_decode_eligible(selection.z, mode=mode):
            xhat = self._decode_cuda_sparse_topk(
                selection.z,
                selection.mask,
                decoder=decoder,
            )
        else:
            xhat = self.decode(selection.z_selected, _decoder=decoder)
        return BSCOutput(xhat, *selection), decoder, encoder

    def select_with_materialized(
        self,
        x: torch.Tensor,
        *,
        mode: str = "topk",
        observed: torch.Tensor | None = None,
        validate_observed: bool = True,
        _decoder: torch.Tensor | None = None,
        _encoder: torch.Tensor | None = None,
        _score_geometry: _ScoreGeometry | None = None,
        _encoder_sites: _FrozenEncoderSites | None = None,
        _score_grad: bool = True,
    ) -> tuple[BSCSelection, torch.Tensor | None, torch.Tensor | None]:
        """Encode and select without paying for an unused dense decode.

        Frozen calibration and codec paths consume only code, score, and
        support tensors.  Keeping that contract separate from
        :meth:`forward_with_materialized` avoids materializing ``[B,S,d]``
        reconstructions while preserving the public forward result exactly.
        """
        direct_factorized = (
            self.uses_direct_factorized_execution
            and _decoder is None
            and _encoder is None
            and _score_geometry is None
            and _encoder_sites is None
        )
        if direct_factorized:
            z, keep = self._encode_factorized_direct(
                x,
                observed=observed,
                validate_observed=validate_observed,
            )
            with torch.set_grad_enabled(torch.is_grad_enabled() and _score_grad):
                scores = self.scores(
                    z,
                    x=x,
                    observed=observed,
                    _observation_keep=keep,
                )
            mask = self._select_scores(scores, mode=mode, z=z)
            z_selected = z * mask.unsqueeze(-1)
            return BSCSelection(z, z_selected, scores, mask), None, None

        decoder = self.decoder_tensor() if _decoder is None else _decoder
        if _encoder is None:
            if self.cfg.encoder_mode == "tied":
                encoder = self._tied_encoder_tensor(decoder)
            else:
                encoder = self.encoder_tensor()
        else:
            encoder = _encoder
        if _encoder_sites is None:
            z, keep = self._encode_with_tensor(
                x,
                encoder,
                observed=observed,
                validate_observed=validate_observed,
            )
        else:
            z, keep = self._encode_from_frozen_sites(
                x,
                encoder,
                _encoder_sites,
                observed=observed,
                validate_observed=validate_observed,
            )
        with torch.set_grad_enabled(torch.is_grad_enabled() and _score_grad):
            scores = self.scores(
                z,
                x=x,
                observed=observed,
                _decoder=decoder,
                _observation_keep=keep,
                _score_geometry=_score_geometry,
            )
        mask = self._select_scores(scores, mode=mode, z=z)
        z_selected = z * mask.unsqueeze(-1)
        return BSCSelection(z, z_selected, scores, mask), decoder, encoder

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
        """One-shot equalize positive homogeneous selection scores by block.

        This convenience method is intentionally narrower than the campaign's
        replayed global calibration.  A signed score or nonlinear group shrink
        destroys the multiplicative identity used here and must use a measured
        solver instead of an analytically inferred post-fit value.
        """
        if self.cfg.encoder_mode == "tied":
            # Fel Grassmannian has one learned gamma, not per-block scales.
            return
        if self.cfg.code_activation == "group_soft_threshold":
            raise ValueError(
                "one-shot encoder calibration is invalid after group soft "
                "thresholding; use a remeasured global solver"
            )
        if self.cfg.selection_score == "isolated_loss_decrease":
            raise ValueError(
                "one-shot encoder calibration requires a nonnegative homogeneous score"
            )
        p = self.scores(self.encode(x), x=x)  # [B, G]
        mean_p = p.mean(dim=0)  # [G]
        if not bool(torch.isfinite(mean_p).all()) or bool((mean_p <= eps).any()):
            raise ValueError(
                "one-shot encoder calibration requires every mean score to be "
                "finite and strictly positive"
            )
        if self.cfg.site_rank is not None and per_block:
            raise ValueError(
                "per-block encoder calibration is not representable by a "
                "site-axis factorization; use the global calibration"
            )
        if per_block:
            scale = mean_p.median() / mean_p  # [G]
            assert self.E is not None
            self._encoder_full_tensor().mul_(scale.view(1, -1, 1, 1))
        else:
            self.scale_encoder_(float(mean_p.median() / mean_p.mean()))
        if self.E is not None:
            self._encoder_full_tensor().mul_(self.coordinate_mask)

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
            self._encoder_full_tensor().mul_(self.coordinate_mask)

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
    def _project_decoder_with_state_(
        self,
        *,
        qr_input_finite: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Project and return a device count plus the exact mutated tensors."""
        mutated: list[torch.Tensor] = []

        def mark(tensor: torch.Tensor) -> None:
            if not any(existing is tensor for existing in mutated):
                mutated.append(tensor)

        bad = torch.zeros((), dtype=torch.int64, device=self.parameter_device)
        if self.cfg.site_rank is not None:
            # BSCConfig permits only the free-decoder factorized arm.  There
            # is no exact factor-space equivalent of the full-tensor Stiefel
            # or norm projections, so silently materializing/projecting and
            # refactorizing would change the optimizer state and is forbidden.
            pass
        else:
            assert self.D is not None
            if self._has_padded_coordinates:
                self.D.mul_(self.coordinate_mask)
                mark(self.D)
            if self.cfg.decoder_constraint == "gram":
                bad = _retract_count_tensor_(
                    self.D,
                    eig_floor=self.cfg.eig_floor,
                    implementation=self.cfg.decoder_retraction_implementation,
                )
                mark(self.D)
            elif self.cfg.decoder_constraint == "qr":
                if (
                    self.cfg.decoder_retraction_implementation
                    == DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
                ):
                    bad = _cholesky_qr_retract_count_tensor_(
                        self.D,
                        input_finite=qr_input_finite,
                    )
                else:
                    assert (
                        self.cfg.decoder_retraction_implementation
                        == DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION
                    )
                    bad = _qr_retract_count_tensor_(
                        self.D,
                        input_finite=qr_input_finite,
                    )
                mark(self.D)
            elif self.cfg.decoder_constraint == "frobenius":
                bad = _project_block_frobenius_count_tensor_(self.D)
                mark(self.D)
            elif self.cfg.decoder_constraint == "unit_frobenius":
                bad = _normalize_block_frobenius_count_tensor_(self.D)
                mark(self.D)
            elif self.cfg.decoder_constraint == "unit_latent":
                bad = _project_latent_rows_count_tensor_(self.D)
                mark(self.D)
            assert torch.is_tensor(bad)
            if self._has_padded_coordinates:
                self.D.mul_(self.coordinate_mask)
                mark(self.D)
            if self.E is not None:
                encoder = self._encoder_full_tensor()
                if self._has_padded_coordinates:
                    encoder.mul_(self.coordinate_mask)
                    mark(self.E)
                if self.cfg.encoder_constraint == "unit_latent":
                    _project_latent_rows_count_tensor_(encoder)
                    mark(self.E)
                    if self._has_padded_coordinates:
                        encoder.mul_(self.coordinate_mask)
                        mark(self.E)
        if not self.cfg.decoder_bias:
            self.c.zero_()
            mark(self.c)
        elif self._has_padded_coordinates:
            self.c.mul_(self.coordinate_mask[:, 0, 0])
            mark(self.c)
        return bad, tuple(mutated)

    @torch.no_grad()
    def project_decoder_(self) -> int:
        """Apply the configured decoder constraint after an optimizer step."""
        bad, _ = self._project_decoder_with_state_()
        if self.cfg.site_rank is not None or self.cfg.decoder_constraint in {
            "free",
            "qr",
        }:
            return 0
        return int(bad.item())

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
        if self.uses_direct_factorized_execution:

            def batch_scores(value: torch.Tensor) -> torch.Tensor:
                z, keep = self._encode_factorized_direct(value)
                return self.scores(
                    z,
                    x=value,
                    _observation_keep=keep,
                )

        else:
            decoder = self.decoder_tensor()
            encoder = (
                self._tied_encoder_tensor(decoder)
                if self.cfg.encoder_mode == "tied"
                else self.encoder_tensor()
            )
            score_geometry = self._frozen_score_geometry(decoder)

            def batch_scores(value: torch.Tensor) -> torch.Tensor:
                z, keep = self._encode_with_tensor(value, encoder)
                return self.scores(
                    z,
                    x=value,
                    _decoder=decoder,
                    _observation_keep=keep,
                    _score_geometry=score_geometry,
                )

        if method == "streaming":
            histogram_type = (
                SignedStreamingScoreQuantile
                if self.cfg.selection_score == "isolated_loss_decrease"
                else StreamingScoreQuantile
            )
            hist = histogram_type(device=self.parameter_device)
            for x in batches:
                value = x.to(self.parameter_device, self.parameter_dtype)
                hist.update(batch_scores(value))
            theta = hist.quantile(q)
        elif method == "exact":
            score_batches = []
            for x in batches:
                value = x.to(self.parameter_device, self.parameter_dtype)
                score_batches.append(batch_scores(value).flatten().float().cpu())
            scores = torch.cat(score_batches)
            n = scores.numel()
            idx = min(max(int(round(q * n)), 1), n)
            theta = float(scores.kthvalue(idx).values)
        else:
            raise ValueError("method must be 'exact' or 'streaming'")
        self.theta.fill_(theta)
        return theta


# A compiled quadratic reduction becomes worthwhile well below the
# smallest Phase-2 batch (2048 * 4 * 768 = 6,291,456 values).  Keep a fixed
# one-megavalue gate so tiny Phase-1 cells, calibration tails, and unit tests do
# not pay TorchInductor compilation churn.  Dynamic batch/site/width symbols
# keep a multi-cell campaign on one graph: a static wrapper hits Dynamo's hard
# recompile limit after eight distinct campaign shapes.  Inductor still emits
# the same fused cast/subtract/square/reduction kernel for the live tensors.
# Compiling the
# reduction changes its parallel summation order, so the release gate bounds
# loss, input-gradient, optimizer-trajectory, and selector-support drift rather
# than claiming bit-exact parity with the eager reduction.
_CUDA_QUADRATIC_FUSION_MIN_ELEMENTS = 1 << 20


def _eager_fp32_squared_error_elements(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (prediction.float() - target.float()).pow(2)


def _eager_fp32_squared_error_reduction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    denominator: int,
) -> torch.Tensor:
    return _eager_fp32_squared_error_elements(prediction, target).sum() / denominator


@cache
def _compiled_cuda_fp32_squared_error_reduction():
    """Create one shape-polymorphic Torch 2.8 Inductor reduction wrapper."""

    return torch.compile(
        _eager_fp32_squared_error_reduction,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
    )


def _fp32_squared_error_reduction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    denominator: int,
) -> torch.Tensor:
    """Fp32 quadratic objective, compiled only for the dominant CUDA path."""

    if (
        prediction.is_cuda
        and target.is_cuda
        and prediction.numel() >= _CUDA_QUADRATIC_FUSION_MIN_ELEMENTS
    ):
        return _compiled_cuda_fp32_squared_error_reduction()(
            prediction,
            target,
            denominator,
        )
    return _eager_fp32_squared_error_reduction(prediction, target, denominator)


def bsc_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    model: BlockCrosscoder,
    observation_mask: torch.Tensor | None = None,
    *,
    validate_observation_mask: bool = True,
) -> torch.Tensor:
    """Compute the pinned reconstruction objective from one prediction."""

    cfg = model.cfg
    coord = model.coordinate_mask[:, 0, 0].to(target.device)
    all_observed = observation_mask is None
    if observation_mask is None:
        observed = None
    else:
        if observation_mask.shape != (target.shape[0], cfg.n_sites):
            raise ValueError(
                f"observation_mask must have shape [{target.shape[0]}, {cfg.n_sites}]"
            )
        observed = observation_mask.to(device=target.device, dtype=torch.bool)
        if validate_observation_mask and not bool(observed.any()):
            raise ValueError("observation_mask excludes the entire batch")

    # The dominant real-model objective is all-observed, rectangular, and
    # quadratic. Fuse its fp32 casts/subtract/square/sum together with the
    # declared normalization division. Missingness, padding, nonquadratic
    # objectives, small tensors, and non-CUDA devices keep the ordinary eager
    # implementation.
    if (
        all_observed
        and not model._has_padded_coordinates
        and cfg.reconstruction_loss in {"mean_squared", "squared_l2"}
    ):
        denominator = (
            target.shape[0]
            if cfg.reconstruction_loss == "squared_l2"
            else target.shape[0] * sum(cfg.site_dims)
        )
        return _fp32_squared_error_reduction(prediction, target, denominator)

    residual = prediction.float() - target.float()
    if model._has_padded_coordinates:
        residual = residual * coord
    if all_observed:
        if cfg.reconstruction_loss == "mean_l2":
            return residual.norm(dim=-1).sum() / (target.shape[0] * cfg.n_sites)
        if cfg.reconstruction_loss == "mean_l1":
            return residual.abs().sum(dim=-1).sum() / (target.shape[0] * cfg.n_sites)
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

    l_rec = bsc_reconstruction_loss(
        out.xhat,
        x,
        model,
        observation_mask,
        validate_observation_mask=validate_observation_mask,
    )
    total = l_rec
    parts: dict[str, torch.Tensor] = {"rec": l_rec}
    if cfg.lambda_regularizer > 0 and cfg.regularizer != "none":
        if cfg.regularizer == "map_nuclear":
            factor_kernel = (
                decoder is None
                and encoder is None
                and model.uses_factorized_nuclear_regularizers
                and model.D_core is not None
                and model.E_core is not None
            )
            if factor_kernel:
                assert model.D_site is not None and model.E_site is not None
                reg = factorized_map_nuclear_penalty(
                    model.D_site,
                    model._decoder_factor_core_tensor(masked=True),
                    model.E_site,
                    model._encoder_factor_core_tensor(masked=True),
                    eps=cfg.sv_eps,
                )
            else:
                D = model.decoder_tensor() if decoder is None else decoder
                E = model.encoder_tensor() if encoder is None else encoder
                reg = map_nuclear_penalty(
                    D,
                    E,
                    eps=cfg.sv_eps,
                    implementation=cfg.map_nuclear_implementation,
                )
            if cfg.map_nuclear_reduction == "sum_blocks":
                reg = reg * cfg.n_blocks * cfg.block_dim
        elif cfg.regularizer == "decoder_nuclear":
            factor_kernel = (
                decoder is None
                and model.uses_factorized_nuclear_regularizers
                and model.D_core is not None
            )
            if factor_kernel:
                assert model.D_site is not None
                reg = factorized_decoder_nuclear_penalty(
                    model.D_site,
                    model._decoder_factor_core_tensor(masked=True),
                    eps=cfg.sv_eps,
                )
            else:
                D = model.decoder_tensor() if decoder is None else decoder
                reg = decoder_nuclear_penalty(D, eps=cfg.sv_eps)
        elif cfg.regularizer == "crosscoder_l1":
            D = model.decoder_tensor() if decoder is None else decoder
            # Anthropic's sitewise decoder-norm-weighted activation L1.
            # For the paper-faithful bridge b=1; the Frobenius extension is
            # well-defined for blocks but is not claimed as their objective.
            per_site = D.float().pow(2).sum(dim=(2, 3)).sqrt()
            if cfg.decoder_norm_geometry == "sum_l2":
                site_cost = per_site.sum(dim=0)
            else:
                site_cost = per_site.pow(2).sum(dim=0).sqrt()
            # The objective is activation times decoder cost.  Selection
            # scores are not the activation carrier: under the paper-faithful
            # ``decoder_weighted`` bridge they already contain ``site_cost``
            # and using them here would silently square the decoder penalty.
            activation = out.z.float().norm(dim=-1)
            reg = (activation * site_cost.unsqueeze(0)).sum(dim=1).mean()
        elif cfg.regularizer == "group_l21":
            # Fel Group-Lasso BSF: mean over examples of the sum of activated
            # block norms.  The learned group soft threshold lives in encode.
            reg = out.z.float().norm(dim=-1).sum(dim=1).mean()
            if cfg.group_lasso_target_k is not None:
                mean_active = out.mask.float().sum(dim=1).mean()
                above_target = mean_active.to(torch.float64) > float(
                    cfg.group_lasso_target_k
                )
                reg = reg * above_target.to(reg.dtype)
        else:  # guarded by BSCConfig
            raise AssertionError(cfg.regularizer)
        parts["regularizer"] = reg
        total = total + cfg.lambda_regularizer * reg
    parts["total"] = total
    return parts
