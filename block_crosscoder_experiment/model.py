"""Block-sparse crosscoder: G blocks of width b, one shared code across sites.

All model mathematics lives in the store's declared per-site coordinates;
the model is agnostic to whether they are raw, scalar-normalized,
token-LayerNorm, or whitened. Inputs are batches x: [B, S, d].

Parameter stacks are [S, G, b, d] so that ``reshape(S, G*b, d)`` is a free
view and encode/decode run as cuBLAS batched matmuls with no per-forward
copies. Parameters are fp32 masters; the bf16 forward copy and the
optimizer-step -> retract -> recast ordering are the trainer's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch import nn

from .gram import (
    init_decoder_stack,
    map_nuclear_penalty,
    project_block_frobenius_,
    retract_,
    site_profile_penalty,
)

__all__ = [
    "BSCConfig",
    "BSCOutput",
    "BlockCrosscoder",
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
            math.log10(lo), math.log10(hi), n_bins + 1,
            dtype=torch.float32, device=device,
        )
        # counts[0] = underflow (<= lo), counts[1..n_bins] = bins,
        # counts[n_bins+1] = overflow (> hi).
        self.counts = torch.zeros(n_bins + 2, dtype=torch.int64)

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        s = scores.detach().flatten().float()
        if not torch.isfinite(s).all():
            raise ValueError("non-finite selection scores in calibration batch")
        idx = torch.bucketize(s, self.edges.to(s.device), right=False)
        self.counts += torch.bincount(idx, minlength=self.n_bins + 2).cpu()

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
    encoder_mode: str = "untied"  # untied | tied
    encoder_bias: bool = False
    code_activation: str = "signed"  # signed | relu | group_soft_threshold
    selection_score: str = "code_norm"  # code_norm | decoder_weighted
    decoder_constraint: str = "gram"  # gram | frobenius | free
    regularizer: str | None = None  # plus map nuclear, crosscoder L1, group L21

    def __post_init__(self) -> None:
        if self.selection not in {"batch_topk", "token_topk", "threshold", "dense"}:
            raise ValueError(
                "selection must be batch_topk, token_topk, threshold, or dense"
            )
        if self.encoder_mode not in {"untied", "tied"}:
            raise ValueError("encoder_mode must be untied or tied")
        if self.code_activation not in {"signed", "relu", "group_soft_threshold"}:
            raise ValueError(
                "code_activation must be signed, relu, or group_soft_threshold"
            )
        if self.selection_score not in {"code_norm", "decoder_weighted"}:
            raise ValueError("selection_score must be code_norm or decoder_weighted")
        if self.decoder_constraint not in {"gram", "frobenius", "free"}:
            raise ValueError("decoder_constraint must be gram, frobenius, or free")
        if self.regularizer is None:
            self.regularizer = (
                "site_profile" if self.lambda_regularizer > 0 else "none"
            )
        if self.regularizer not in {
            "none", "site_profile", "map_nuclear", "crosscoder_l1", "group_l21"
        }:
            raise ValueError("unknown regularizer")
        if self.regularizer == "crosscoder_l1" and self.code_activation != "relu":
            raise ValueError("crosscoder_l1 requires relu codes")
        if self.regularizer == "group_l21" and self.code_activation != "group_soft_threshold":
            raise ValueError("group_l21 requires group_soft_threshold codes")
        if self.selection == "dense" and self.code_activation != "relu":
            if self.code_activation != "group_soft_threshold":
                raise ValueError("dense selection requires relu or group_soft_threshold codes")
        if self.selection_score == "decoder_weighted" and self.code_activation != "relu":
            raise ValueError("decoder_weighted selection is the ReLU crosscoder bridge")

    @property
    def n_latents(self) -> int:
        return self.n_blocks * self.block_dim


class BSCOutput(NamedTuple):
    xhat: torch.Tensor  # [B, S, d] reconstruction in declared coordinates
    z: torch.Tensor  # [B, G, b] pre-selection code
    z_selected: torch.Tensor  # [B, G, b] post-selection code (masked)
    scores: torch.Tensor  # [B, G] selection scores p_g = ||z_g||
    mask: torch.Tensor  # [B, G] bool, selected blocks


def batch_topk_mask(scores: torch.Tensor, k: float) -> torch.Tensor:
    """BatchTopK over blocks: keep the top round(k*B) block-activations
    batch-wide. Fractional k sets the budget below one block per token —
    the under-provisioned regime the capture sweep probes.

    Per-token counts vary by design; only the batch total is pinned.
    scores: [B, G]  ->  bool mask [B, G]
    """
    B, G = scores.shape
    n_keep = min(int(round(k * B)), B * G)
    flat = scores.reshape(-1)
    idx = flat.topk(n_keep, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view(B, G)


def token_topk_mask(scores: torch.Tensor, k: float) -> torch.Tensor:
    """Per-token block TopK used by the published BSF and SASA recipes."""
    B, G = scores.shape
    n_keep = min(max(int(round(k)), 0), G)
    if n_keep == 0:
        return torch.zeros(B, G, dtype=torch.bool, device=scores.device)
    idx = scores.topk(n_keep, dim=1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


class BlockCrosscoder(nn.Module):
    """Gram-constrained block-sparse crosscoder (design v2.2 architecture).

    encode:  z_g = sum_s E_g^s x^s          (summed per-site maps, untied)
    select:  BatchTopK on p_g = ||z_g||     (exact contribution energy under
                                             the Gram constraint)
    decode:  xhat^s = c^s + sum_{g active} D_g^s^T z_g
    """

    E: nn.Parameter | None  # [S, G, b, d], absent for tied Grassmannian
    D: nn.Parameter  # [S, G, b, d]
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
            cfg.n_sites, cfg.n_blocks, cfg.block_dim, cfg.d_model, generator=gen
        )
        if cfg.decoder_constraint == "frobenius":
            project_block_frobenius_(D)
        if device is not None:
            D = D.to(device)
        self.D = nn.Parameter(D)
        # Transpose-tied at init only (Fel App. D convention); encoder scale
        # is norm-calibrated on a data batch via calibrate_encoder_scale_.
        if cfg.encoder_mode == "untied":
            self.E = nn.Parameter(D.detach().clone())
            self.register_parameter("log_gamma", None)
        else:
            self.register_parameter("E", None)
            self.log_gamma = nn.Parameter(torch.zeros((), device=D.device))
        if cfg.encoder_bias:
            self.a = nn.Parameter(
                torch.zeros(cfg.n_blocks, cfg.block_dim, device=D.device)
            )
        else:
            self.register_parameter("a", None)
        if cfg.code_activation == "group_soft_threshold":
            # softplus(-2.252...) ~= 0.1.  One positive learned threshold per
            # block is Fel's Group-Lasso BSF parameterization.
            self.log_threshold = nn.Parameter(
                torch.full((cfg.n_blocks,), -2.2521685, device=D.device)
            )
        else:
            self.register_parameter("log_threshold", None)
        self.c = nn.Parameter(torch.zeros(cfg.n_sites, cfg.d_model, device=D.device))
        # Inference threshold theta: fit on the calibration split, frozen and
        # serialized with the codec (D10). NaN until calibrated.
        self.register_buffer("theta", torch.tensor(float("nan")))

    # -- core ops ---------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, d] -> z: [B, G, b]."""
        cfg = self.cfg
        E = self.D * self.log_gamma.exp() if self.E is None else self.E
        W = E.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)  # view
        # [S, B, d] @ [S, d, G*b] -> [S, B, G*b], summed over sites.
        z = torch.bmm(x.transpose(0, 1), W.transpose(1, 2)).sum(dim=0)
        z = z.view(-1, cfg.n_blocks, cfg.block_dim)
        if self.a is not None:
            z = z + self.a
        if cfg.code_activation == "relu":
            z = torch.relu(z)
        elif cfg.code_activation == "group_soft_threshold":
            norm = z.norm(dim=-1, keepdim=True)
            threshold = torch.nn.functional.softplus(self.log_threshold).view(1, -1, 1)
            z = z * torch.relu(1.0 - threshold / norm.clamp_min(1e-12))
        return z

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        """Configured sparse-event score.

        Gram-constrained BSCs use the block norm, which is exactly isolated
        decoded energy.  Minder's scalar BatchTopK crosscoder instead uses a
        ReLU activation multiplied by the sum of its site decoder norms.
        """
        score = z.norm(dim=-1)
        if self.cfg.selection_score == "decoder_weighted":
            site_norms = self.D.float().pow(2).sum(dim=(2, 3)).sqrt().sum(dim=0)
            score = score * site_norms.to(score.dtype).unsqueeze(0)
        return score

    def select(self, z: torch.Tensor, *, mode: str = "topk") -> torch.Tensor:
        """Bool mask [B, G]. ``topk`` = training BatchTopK; ``threshold`` =
        inference against the frozen calibrated theta."""
        p = self.scores(z)
        if mode == "topk":
            if self.cfg.selection == "batch_topk":
                return batch_topk_mask(p, self.cfg.k)
            if self.cfg.selection == "token_topk":
                return token_topk_mask(p, self.cfg.k)
            if self.cfg.selection == "threshold":
                if torch.isnan(self.theta):
                    raise RuntimeError("training threshold not calibrated")
                return p > self.theta
            return p > 0
        if mode == "threshold":
            if torch.isnan(self.theta):
                raise RuntimeError("inference threshold not calibrated")
            return p > self.theta
        raise ValueError(f"unknown selection mode {mode!r}")

    def decode(self, z_selected: torch.Tensor, *, add_bias: bool = True) -> torch.Tensor:
        """z_selected: [B, G, b] -> xhat: [B, S, d]. AuxK residual
        reconstruction decodes without the bias (add_bias=False)."""
        cfg = self.cfg
        Wd = self.D.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)  # view
        flat = z_selected.reshape(-1, cfg.n_latents)
        # [B, G*b] @ [S, G*b, d] broadcasts to [S, B, d].
        xhat = torch.matmul(flat, Wd)
        if add_bias:
            xhat = xhat + self.c.unsqueeze(1)
        return xhat.transpose(0, 1)

    def forward(self, x: torch.Tensor, *, mode: str = "topk") -> BSCOutput:
        z = self.encode(x)
        mask = self.select(z, mode=mode)
        z_selected = z * mask.unsqueeze(-1)
        xhat = self.decode(z_selected)
        return BSCOutput(xhat, z, z_selected, self.scores(z), mask)

    # -- init calibration --------------------------------------------------

    @torch.no_grad()
    def calibrate_encoder_scale_(
        self, x: torch.Tensor, *, per_block: bool = True, eps: float = 1e-12
    ) -> None:
        """Scale the encoder so initial selection scores are comparable
        across blocks. Fel-inspired (P16, sol S6): Fel App. D prescribes
        transpose-tied init with encoder scale calibration in broad terms;
        the per-block median equalization here is BSC-specific. Preserves
        the global scale the tied Gram-constrained init already gives.
        """
        p = self.scores(self.encode(x))  # [B, G]
        if self.E is None:
            # Fel Grassmannian has one learned gamma, not per-block scales.
            return
        mean_p = p.mean(dim=0).clamp_min(eps)  # [G]
        if per_block:
            scale = mean_p.median() / mean_p  # [G]
            self.E.mul_(scale.view(1, -1, 1, 1))
        else:
            self.E.mul_(mean_p.median() / mean_p.mean())

    @property
    def parameter_device(self) -> torch.device:
        return self.D.device

    @property
    def parameter_dtype(self) -> torch.dtype:
        return self.D.dtype

    @torch.no_grad()
    def project_decoder_(self) -> int:
        """Apply the configured decoder constraint after an optimizer step."""
        if self.cfg.decoder_constraint == "gram":
            return retract_(self.D.data, eig_floor=self.cfg.eig_floor)
        if self.cfg.decoder_constraint == "frobenius":
            return project_block_frobenius_(self.D.data)
        return 0

    @torch.no_grad()
    def fit_threshold_(
        self, batches, target_avg_blocks: float, *, method: str = "exact"
    ) -> float:
        """Fit the frozen inference threshold theta on the calibration
        split so the average active-block count hits the preregistered
        target (D10): mean count = G * P(p > theta), so theta is the
        (1 - target/G) quantile of the pooled score distribution.

        method="exact": kthvalue over host-accumulated scores (not
        torch.quantile, which caps at ~16M elements). At G=4096 the
        pooled score matrix is already ~8.6 GB at a modest pilot slice and
        cannot sit next to the model on a 24 GB card; the scalar pilot arm
        also exhausted 61 GB host RAM. Kept as the validation reference.

        method="streaming": bounded-memory log-histogram quantile
        — the production path, mandatory for the 13M-token calibration
        split, any G >= 8192
        config, and the scalar production arm. Deterministic
        (batch-order independent, int64 counts); resolution ~3e-5
        relative in theta. Validation gate vs exact:
        |Δ avg-blocks| <= 0.1.
        """
        q = 1.0 - target_avg_blocks / self.cfg.n_blocks
        if method == "streaming":
            hist = StreamingScoreQuantile(device=self.parameter_device)
            for x in batches:
                hist.update(self.scores(self.encode(
                    x.to(self.parameter_device, self.parameter_dtype)
                )))
            theta = hist.quantile(q)
        elif method == "exact":
            scores = torch.cat(
                [self.scores(self.encode(
                    x.to(self.parameter_device, self.parameter_dtype)
                ))
                 .flatten().float().cpu()
                 for x in batches]
            )
            n = scores.numel()
            idx = min(max(int(round(q * n)), 1), n)
            theta = float(scores.kthvalue(idx).values)
        else:
            raise ValueError("method must be 'exact' or 'streaming'")
        self.theta.fill_(theta)
        return theta


def bsc_loss(
    out: BSCOutput, x: torch.Tensor, model: BlockCrosscoder
) -> dict[str, torch.Tensor]:
    """Pinned reductions (R12) so lambda and alpha transfer across configs.

    L_rec = mean over tokens, sites, dims of the squared residual.
    L_aux  lives in the trainer (AuxK needs cross-step frequency state).

    Reductions run in fp32 regardless of forward dtype — a bf16 mean over
    millions of elements loses the precision the comparisons need.
    """
    cfg = model.cfg
    l_rec = (out.xhat.float() - x.float()).pow(2).mean()
    total = l_rec
    parts: dict[str, torch.Tensor] = {"rec": l_rec}
    if cfg.lambda_regularizer > 0 and cfg.regularizer != "none":
        if cfg.regularizer == "site_profile":
            reg = site_profile_penalty(model.D, eps=cfg.sv_eps)
        elif cfg.regularizer == "map_nuclear":
            E = model.D * model.log_gamma.exp() if model.E is None else model.E
            reg = map_nuclear_penalty(model.D, E, eps=cfg.sv_eps)
        elif cfg.regularizer == "crosscoder_l1":
            # Anthropic's sitewise decoder-norm-weighted activation L1.
            # For the paper-faithful bridge b=1; the Frobenius extension is
            # well-defined for blocks but is not claimed as their objective.
            site_cost = model.D.float().pow(2).sum(dim=(2, 3)).sqrt().sum(dim=0)
            reg = (out.scores.float() * site_cost.unsqueeze(0)).sum(dim=1).mean()
        elif cfg.regularizer == "group_l21":
            # Fel Group-Lasso BSF: mean over examples of the sum of activated
            # block norms.  The learned group soft threshold lives in encode.
            reg = out.z.float().norm(dim=-1).sum(dim=1).mean()
        else:  # guarded by BSCConfig
            raise AssertionError(cfg.regularizer)
        parts["regularizer"] = reg
        # Legacy key retained for old reports/tests while the audit migrates.
        parts["rank"] = reg
        total = total + cfg.lambda_regularizer * reg
    parts["total"] = total
    return parts
