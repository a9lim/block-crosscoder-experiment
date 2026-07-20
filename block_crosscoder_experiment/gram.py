"""Gram-constraint primitives for the block-sparse crosscoder.

All functions operate on decoder/encoder stacks of shape ``[S, G, b, d]``
(sites, blocks, block dim, model dim) in whitened per-site coordinates.
The layout is chosen so ``reshape(S, G*b, d)`` is a free view for the
encode/decode batched matmuls.

The load-bearing constraint (design v2.2, *Architecture spec*): per block,
the concatenated decoder Gram is the identity,

    M_g = sum_s D_g^s D_g^s^T = I_b.

It simultaneously (i) kills the z->cz scale gauge, (ii) reduces the
within-block GL(b) gauge to O(b), (iii) makes ||z_g||^2 exactly the block's
contribution energy sum_s ||D_g^s^T z_g||^2, and (iv) blocks the dead-block
decoder-shrinkage spiral. Enforced by retraction after every optimizer step,
on the fp32 master weights.
"""

from __future__ import annotations

import math

import torch

# Safe batch count for cusolver's batched symmetric eigensolvers
# (empirical ceiling sits between 24576 and 32768 on CUDA 12.8).
_EIGH_MAX_BATCH = 16384
# G8192/b4 fits the 4090 for forward/backward but the unconstrained einsum
# planner requests an additional ~2.5 GiB workspace during retraction. Chunk
# over blocks so temporary memory scales with this constant, not the dictionary.
_GRAM_BLOCK_CHUNK = 512
_RETRACT_UNCHUNKED_MAX = 4096
_SPECTRUM_BLOCK_CHUNK = 256
_SPECTRUM_UNCHUNKED_MAX = 4096

__all__ = [
    "block_gram",
    "gram_residual",
    "retract_",
    "site_singular_values",
    "rank_penalty",
    "site_frobenius_shares",
    "init_decoder_stack",
]


def block_gram(D: torch.Tensor) -> torch.Tensor:
    """Concatenated decoder Gram per block: M_g = sum_s D_g^s D_g^s^T.

    D: [S, G, b, d]  ->  M: [G, b, b]
    """
    if D.shape[1] <= _GRAM_BLOCK_CHUNK:
        return torch.einsum("sgbd,sgcd->gbc", D, D)
    return torch.cat(
        [
            torch.einsum(
                "sgbd,sgcd->gbc",
                D[:, start : start + _GRAM_BLOCK_CHUNK],
                D[:, start : start + _GRAM_BLOCK_CHUNK],
            )
            for start in range(0, D.shape[1], _GRAM_BLOCK_CHUNK)
        ],
        dim=0,
    )


def gram_residual(D: torch.Tensor) -> torch.Tensor:
    """Per-block Frobenius residual ||M_g - I_b||_F (training-health metric).

    D: [S, G, b, d]  ->  [G]
    """
    M = block_gram(D)
    b = M.shape[-1]
    eye = torch.eye(b, device=M.device, dtype=M.dtype)
    return (M - eye).norm(dim=(-2, -1))


@torch.no_grad()
def retract_(D: torch.Tensor, *, eig_floor: float = 1e-6) -> int:
    """In-place retraction onto the Gram manifold: D_g^s <- M_g^{-1/2} D_g^s.

    Operates on the fp32 master decoders (design: optimizer step on master ->
    retract master -> regenerate bf16 forward copy -> log post-cast residual).
    Eigenvalues of M_g are floored at ``eig_floor`` before inversion; the
    number of floor hits is returned for logging (persistent hits after init
    indicate a genuinely rank-deficient block the retraction cannot repair).

    D: [S, G, b, d], modified in place. Returns the floor-hit count.
    """
    if D.dtype != torch.float32:
        raise TypeError(f"retraction operates on fp32 master weights, got {D.dtype}")
    floor_hits = 0
    chunk_size = (
        D.shape[1]
        if D.shape[1] <= _RETRACT_UNCHUNKED_MAX else _GRAM_BLOCK_CHUNK
    )
    for start in range(0, D.shape[1], chunk_size):
        chunk = D[:, start : start + chunk_size]
        M = torch.einsum("sgbd,sgcd->gbc", chunk, chunk)
        evals, evecs = torch.linalg.eigh(M)
        floor_hits += int((evals < eig_floor).sum().item())
        evals = evals.clamp_min(eig_floor)
        inv_sqrt = (
            evecs
            @ torch.diag_embed(evals.rsqrt())
            @ evecs.transpose(-1, -2)
        )
        chunk.copy_(torch.einsum("gbc,sgcd->sgbd", inv_sqrt, chunk))
    return floor_hits


def site_singular_values(D: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Per-site decoder singular values via batched b x b Gram eigenvalues.

    No d-dimensional SVD anywhere in the loop (design: fp32, sqrt(eig + eps)).
    eigvalsh (eigenvalues only) keeps the backward well-defined for the
    symmetric functions of the spectrum we use, including near-degenerate
    spectra where eigenvector gradients would blow up.

    D: [S, G, b, d]  ->  [S, G, b]
    """
    if D.shape[1] > _SPECTRUM_UNCHUNKED_MAX:
        chunks = []
        for start in range(0, D.shape[1], _SPECTRUM_BLOCK_CHUNK):
            block = D[:, start : start + _SPECTRUM_BLOCK_CHUNK]
            gram = torch.einsum("sgbd,sgcd->sgbc", block, block).float()
            flat = gram.reshape(-1, gram.shape[-2], gram.shape[-1])
            # CUDA's batched solver reserves roughly 256 KiB per 4x4 matrix;
            # with optimizer state resident that dominates the actual 1 MiB
            # tensor. The exact CPU eigensolve has tiny storage, and autograd
            # carries its gradient back through the device copy.
            evals = torch.linalg.eigvalsh(flat.cpu()).to(D.device)
            evals = evals.reshape(gram.shape[:-1])
            chunks.append(evals)
        return (torch.cat(chunks, dim=1).clamp_min(0.0) + eps).sqrt()

    gram_s = torch.einsum("sgbd,sgcd->sgbc", D, D).float()
    # cusolver's batched syev rejects large batch counts (measured on
    # CUDA 12.8 / 4090: S*G = 24576 passes, 32768 fails with
    # CUSOLVER_STATUS_INVALID_VALUE on finite input), so chunk the flat
    # batch; hit at G=8192, S=6.
    flat = gram_s.reshape(-1, gram_s.shape[-2], gram_s.shape[-1])
    if flat.shape[0] > _EIGH_MAX_BATCH:
        evals = torch.cat(
            [
                torch.linalg.eigvalsh(flat[i : i + _EIGH_MAX_BATCH])
                for i in range(0, flat.shape[0], _EIGH_MAX_BATCH)
            ]
        ).reshape(gram_s.shape[:-1])
    else:
        evals = torch.linalg.eigvalsh(gram_s)
    return (evals.clamp_min(0.0) + eps).sqrt()


def rank_penalty(D: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalized per-site nuclear-norm penalty, pinned reduction (R12).

    R_rank = mean_g (sum_s ||D_g^s||_* - b) / b. Under the Gram constraint
    the per-block sum ranges over [b, b*sqrt(S)], so this lives in
    [0, sqrt(S)-1]: 0 = fully site-concentrated, sqrt(S)-1 = flat.
    """
    sv = site_singular_values(D, eps=eps)  # [S, G, b]
    b = D.shape[2]
    nuc = sv.sum(dim=(0, 2))  # [G]
    return ((nuc - b) / b).mean()


def site_frobenius_shares(D: torch.Tensor) -> torch.Tensor:
    """Per-site Frobenius shares tr(D_g^s D_g^s^T)/b — the depth profile.

    Free under the constraint; shares sum to 1 per block once retracted.

    D: [S, G, b, d]  ->  [S, G]
    """
    b = D.shape[2]
    return torch.einsum("sgbd,sgbd->sg", D, D) / b


def init_decoder_stack(
    n_sites: int,
    n_blocks: int,
    block_dim: int,
    d_model: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Gaussian init followed by one retraction (design: *Sparsity hygiene*).

    Gives approximately equal site shares (1/S) at init. Always fp32.
    """
    D = torch.randn(
        n_sites,
        n_blocks,
        block_dim,
        d_model,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    D /= math.sqrt(d_model)
    retract_(D)
    return D
