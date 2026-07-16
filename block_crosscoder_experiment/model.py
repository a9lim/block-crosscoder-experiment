"""Block-sparse crosscoder: G blocks of width b, one shared code across sites.

All model mathematics lives in whitened per-site coordinates (design v2.2,
*Sites, coordinates, whitening*); this module never sees raw activations.
Inputs are whitened batches x: [B, S, d].

Parameter stacks are [S, G, b, d] so that ``reshape(S, G*b, d)`` is a free
view and encode/decode run as cuBLAS batched matmuls with no per-forward
copies. Parameters are fp32 masters; the bf16 forward copy and the
optimizer-step -> retract -> recast ordering are the trainer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
from torch import nn

from .gram import init_decoder_stack, rank_penalty

__all__ = ["BSCConfig", "BSCOutput", "BlockCrosscoder", "batch_topk_mask", "bsc_loss"]


@dataclass
class BSCConfig:
    n_blocks: int  # G
    block_dim: int  # b
    n_sites: int  # S
    d_model: int  # d
    k: float  # average active blocks/token (BatchTopK budget; fractional OK)
    lambda_rank: float = 0.0  # lambda_* on the pinned R_rank reduction
    eig_floor: float = 1e-6  # retraction eigenvalue floor
    sv_eps: float = 1e-8  # eps inside sqrt(eig + eps)
    seed: int = 0

    @property
    def n_latents(self) -> int:
        return self.n_blocks * self.block_dim


class BSCOutput(NamedTuple):
    xhat: torch.Tensor  # [B, S, d] whitened reconstruction
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


class BlockCrosscoder(nn.Module):
    """Gram-constrained block-sparse crosscoder (design v2.2 architecture).

    encode:  z_g = sum_s E_g^s x^s          (summed per-site maps, untied)
    select:  BatchTopK on p_g = ||z_g||     (exact contribution energy under
                                             the Gram constraint)
    decode:  xhat^s = c^s + sum_{g active} D_g^s^T z_g
    """

    E: nn.Parameter  # [S, G, b, d]
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
        if device is not None:
            D = D.to(device)
        self.D = nn.Parameter(D)
        # Transpose-tied at init only (Fel App. D convention); encoder scale
        # is norm-calibrated on a data batch via calibrate_encoder_scale_.
        self.E = nn.Parameter(D.detach().clone())
        self.c = nn.Parameter(torch.zeros(cfg.n_sites, cfg.d_model, device=D.device))
        # Inference threshold theta: fit on the calibration split, frozen and
        # serialized with the codec (D10). NaN until calibrated.
        self.register_buffer("theta", torch.tensor(float("nan")))

    # -- core ops ---------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, d] -> z: [B, G, b]."""
        cfg = self.cfg
        W = self.E.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)  # view
        # [S, B, d] @ [S, d, G*b] -> [S, B, G*b], summed over sites.
        z = torch.bmm(x.transpose(0, 1), W.transpose(1, 2)).sum(dim=0)
        return z.view(-1, cfg.n_blocks, cfg.block_dim)

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        """Selection score p_g = ||z_g||_2 — exact contribution energy."""
        return z.norm(dim=-1)

    def select(self, z: torch.Tensor, *, mode: str = "topk") -> torch.Tensor:
        """Bool mask [B, G]. ``topk`` = training BatchTopK; ``threshold`` =
        inference against the frozen calibrated theta."""
        p = self.scores(z)
        if mode == "topk":
            return batch_topk_mask(p, self.cfg.k)
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
        across blocks (Fel App. D via P16): per-block mean pre-selection
        score is brought to the cross-block median, preserving the global
        scale the tied Gram-constrained init already gives.
        """
        p = self.scores(self.encode(x))  # [B, G]
        mean_p = p.mean(dim=0).clamp_min(eps)  # [G]
        if per_block:
            scale = mean_p.median() / mean_p  # [G]
            self.E.mul_(scale.view(1, -1, 1, 1))
        else:
            self.E.mul_(mean_p.median() / mean_p.mean())

    @torch.no_grad()
    def fit_threshold_(self, batches, target_avg_blocks: float) -> float:
        """Fit the frozen inference threshold theta on the calibration
        split so the average active-block count hits the preregistered
        target (D10): mean count = G * P(p > theta), so theta is the
        (1 - target/G) quantile of the pooled score distribution.
        Uses kthvalue, not torch.quantile (which caps at ~16M elements).
        """
        scores = torch.cat(
            [self.scores(self.encode(x.to(self.E.device, self.E.dtype))).flatten().float()
             for x in batches]
        )
        n = scores.numel()
        q = 1.0 - target_avg_blocks / self.cfg.n_blocks
        idx = min(max(int(round(q * n)), 1), n)
        theta = scores.kthvalue(idx).values
        self.theta.fill_(theta)
        return float(theta)


def bsc_loss(
    out: BSCOutput, x: torch.Tensor, model: BlockCrosscoder
) -> dict[str, torch.Tensor]:
    """Pinned reductions (R12) so lambda and alpha transfer across configs.

    L_rec  = mean over tokens, sites, dims of the squared whitened residual.
    R_rank = mean over blocks of (sum_s ||D_g^s||_* - b)/b.
    L_aux  lives in the trainer (AuxK needs cross-step frequency state).

    Reductions run in fp32 regardless of forward dtype — a bf16 mean over
    millions of elements loses the precision the comparisons need.
    """
    cfg = model.cfg
    l_rec = (out.xhat.float() - x.float()).pow(2).mean()
    total = l_rec
    parts: dict[str, torch.Tensor] = {"rec": l_rec}
    if cfg.lambda_rank > 0:
        r_rank = rank_penalty(model.D, eps=cfg.sv_eps)
        parts["rank"] = r_rank
        total = total + cfg.lambda_rank * r_rank
    parts["total"] = total
    return parts
