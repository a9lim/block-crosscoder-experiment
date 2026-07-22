"""Gram-constraint primitives for the block-sparse crosscoder.

All functions operate on decoder/encoder stacks of shape ``[S, G, b, d]``
(sites, blocks, block dim, model dim) in whitened per-site coordinates.
The layout is chosen so ``reshape(S, G*b, d)`` is a free view for the
encode/decode batched matmuls.

The load-bearing constraint is, per block,
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
from functools import cache

import torch

from .runtime_limits import (
    CHOLESKY_QR_GRAM_CONDITION_MAX,
    CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX,
    CHOLESKY_QR_RECONSTRUCTION_RELATIVE_RESIDUAL_MAX,
)

# Safe batch count for cusolver's batched symmetric eigensolvers
# (empirical ceiling sits between 24576 and 32768 on CUDA 12.8).
_EIGH_MAX_BATCH = 16384
# G8192/b4 fits the 4090 for forward/backward but the unconstrained einsum
# planner requests an additional ~2.5 GiB workspace during retraction. Chunk
# over blocks so temporary memory scales with this constant, not the dictionary.
_GRAM_BLOCK_CHUNK = 512
_RETRACT_UNCHUNKED_MAX = 4096
_SPECTRUM_BLOCK_CHUNK = 256
_SPECTRUM_CUDA_BLOCK_CHUNK = 256
_SPECTRUM_UNCHUNKED_MAX = 4096
_CUDA_FINITE_FUSION_MIN_ELEMENTS = 1 << 20

__all__ = [
    "CholeskyQRRetractionError",
    "block_gram",
    "gram_residual",
    "retract_",
    "qr_retract_",
    "cholesky_qr_retract_",
    "site_singular_values",
    "map_nuclear_penalty",
    "decoder_nuclear_penalty",
    "factorized_map_nuclear_penalty",
    "factorized_decoder_nuclear_penalty",
    "project_block_frobenius_",
    "normalize_block_frobenius_",
    "project_latent_rows_",
    "site_frobenius_shares",
    "init_decoder_stack",
]


class CholeskyQRRetractionError(RuntimeError):
    """Fail-closed refusal from the guarded Cholesky-QR implementation."""


def _eager_all_finite(value: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(value).all()


@cache
def _compiled_cuda_all_finite():
    """Fuse finite classification and reduction without a bool-sized tensor."""

    return torch.compile(
        _eager_all_finite,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
    )


def _all_finite(value: torch.Tensor) -> torch.Tensor:
    if value.is_cuda and value.numel() >= _CUDA_FINITE_FUSION_MIN_ELEMENTS:
        return _compiled_cuda_all_finite()(value)
    return _eager_all_finite(value)


def _validate_qr_retraction_input(D: torch.Tensor) -> None:
    if D.dtype != torch.float32:
        raise TypeError(f"QR retraction operates on fp32 master weights, got {D.dtype}")
    if D.ndim != 4 or any(size <= 0 for size in D.shape):
        raise ValueError("QR retraction requires a nonempty [S,G,b,d] tensor")
    if D.shape[0] * D.shape[3] < D.shape[2]:
        raise ValueError(
            "QR retraction requires at least block_dim concatenated coordinates"
        )


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


@torch.no_grad()
def _block_gram_no_grad(D: torch.Tensor) -> torch.Tensor:
    """Accumulate a block Gram directly, without decoder-sized einsum work."""

    gram = torch.empty(
        D.shape[1],
        D.shape[2],
        D.shape[2],
        dtype=D.dtype,
        device=D.device,
    )
    torch.bmm(D[0], D[0].transpose(-1, -2), out=gram)
    for site in range(1, D.shape[0]):
        torch.baddbmm(
            gram,
            D[site],
            D[site].transpose(-1, -2),
            out=gram,
        )
    return gram


def gram_residual(D: torch.Tensor) -> torch.Tensor:
    """Per-block Frobenius residual ||M_g - I_b||_F (training-health metric).

    D: [S, G, b, d]  ->  [G]
    """
    M = block_gram(D)
    b = M.shape[-1]
    eye = torch.eye(b, device=M.device, dtype=M.dtype)
    return (M - eye).norm(dim=(-2, -1))


@torch.no_grad()
def _retract_count_tensor_(
    D: torch.Tensor,
    *,
    eig_floor: float = 1e-6,
) -> torch.Tensor:
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
    floor_hits = torch.zeros((), dtype=torch.int64, device=D.device)
    chunk_size = (
        D.shape[1] if D.shape[1] <= _RETRACT_UNCHUNKED_MAX else _GRAM_BLOCK_CHUNK
    )
    for start in range(0, D.shape[1], chunk_size):
        chunk = D[:, start : start + chunk_size]
        M = torch.einsum("sgbd,sgcd->gbc", chunk, chunk)
        evals, evecs = torch.linalg.eigh(M)
        floor_hits += (evals < eig_floor).sum()
        evals = evals.clamp_min(eig_floor)
        inv_sqrt = evecs @ torch.diag_embed(evals.rsqrt()) @ evecs.transpose(-1, -2)
        chunk.copy_(torch.einsum("gbc,sgcd->sgbd", inv_sqrt, chunk))
    return floor_hits


@torch.no_grad()
def retract_(D: torch.Tensor, *, eig_floor: float = 1e-6) -> int:
    """Public integer-count wrapper for the Gram-manifold retraction."""
    return int(_retract_count_tensor_(D, eig_floor=eig_floor).item())


@torch.no_grad()
def _qr_retract_count_tensor_(
    D: torch.Tensor,
    *,
    input_finite: bool = False,
) -> torch.Tensor:
    """Canonical positive-diagonal Householder-QR reference retraction.

    Each block's site-concatenated decoder is a ``b x sum(d_s)`` matrix.  QR
    on its transpose produces orthonormal columns, which are transposed back
    to orthonormal decoder rows.  Multiplying each Q column by the sign of its
    R diagonal makes R strictly positive on every full-rank block.  This fixes
    QR's diagonal-sign ambiguity for the ordered input; it does not quotient
    the surviving O(b) gauge.  The complete candidate is validated before the
    caller's tensor is changed.
    """

    _validate_qr_retraction_input(D)
    if not input_finite and not bool(_all_finite(D)):
        raise ValueError("Householder QR retraction requires finite input")
    sites, groups, block_dim, width = D.shape
    concatenated = D.permute(1, 0, 3, 2).reshape(groups, sites * width, block_dim)
    q, r = torch.linalg.qr(concatenated, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    factors_finite = torch.isfinite(q).all() & torch.isfinite(diagonal).all()
    full_rank = (diagonal != 0).all()
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), 1.0)
    q = q * signs.unsqueeze(-2)
    candidate = q.reshape(groups, sites, width, block_dim).permute(1, 0, 3, 2)
    residual = gram_residual(candidate)
    post_ok = (
        torch.isfinite(residual).all()
        & (residual <= CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX).all()
    )
    # The candidate is private until every predicate passes. On the admitted
    # path, one combined host fence replaces three sequential scalar reads;
    # failure-only branches retain the specific refusal diagnostics.
    if not bool(factors_finite & full_rank & post_ok):
        if not bool(factors_finite):
            raise ValueError("Householder QR produced non-finite factors")
        if not bool(full_rank):
            raise ValueError("Householder QR requires full-column-rank blocks")
        raise ValueError("Householder QR candidate violates the Gram bound")
    D.copy_(candidate)
    count = torch.zeros((), dtype=torch.int64, device=D.device)
    return count


@torch.no_grad()
def qr_retract_(D: torch.Tensor) -> int:
    """Public wrapper for canonical positive-diagonal Householder QR."""
    _qr_retract_count_tensor_(D)
    return 0


@torch.no_grad()
def _cholesky_qr_retract_count_tensor_(
    D: torch.Tensor,
    *,
    input_finite: bool = False,
) -> torch.Tensor:
    """Transactional guarded Cholesky-QR1 Stiefel retraction.

    For ordered decoder rows ``B_g`` this computes ``M_g = B_g B_g.T``, the
    positive-diagonal lower Cholesky factor ``L_g``, and ``B'_g=L_g^-1 B_g``.
    This is the same exact QR convention as :func:`qr_retract_` on the admitted
    full-rank carrier.  Cholesky failure, excessive conditioning, or a failed
    numeric postcondition raises without changing ``D``.  There is no jitter,
    eigenvalue floor, or fallback retraction.
    """

    _validate_qr_retraction_input(D)
    if not input_finite and not bool(_all_finite(D)):
        raise CholeskyQRRetractionError(
            "Cholesky-QR requires finite fp32 master weights"
        )

    gram = _block_gram_no_grad(D)
    cholesky, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    diagonal = torch.diagonal(cholesky, dim1=-2, dim2=-1)
    precondition_ok = (
        torch.isfinite(gram).all()
        & torch.isfinite(cholesky).all()
        & (info == 0).all()
        & (diagonal > 0).all()
    )
    block_dim = D.shape[2]
    eye = torch.eye(block_dim, dtype=D.dtype, device=D.device).expand(
        D.shape[1], -1, -1
    )
    inverse_cholesky = torch.linalg.solve_triangular(
        cholesky,
        eye,
        upper=False,
    )
    inverse_gram = inverse_cholesky.transpose(-1, -2) @ inverse_cholesky
    gram_norm_inf = gram.abs().sum(dim=-1).amax(dim=-1)
    inverse_norm_inf = inverse_gram.abs().sum(dim=-1).amax(dim=-1)
    condition = gram_norm_inf * inverse_norm_inf
    reconstruction_residual = (cholesky @ cholesky.transpose(-1, -2) - gram).norm(
        dim=(-2, -1)
    ) / gram.norm(dim=(-2, -1)).clamp_min(torch.finfo(D.dtype).tiny)
    factor_ok = (
        torch.isfinite(inverse_cholesky).all()
        & torch.isfinite(condition).all()
        & torch.isfinite(reconstruction_residual).all()
        & (condition <= CHOLESKY_QR_GRAM_CONDITION_MAX).all()
        & (
            reconstruction_residual <= CHOLESKY_QR_RECONSTRUCTION_RELATIVE_RESIDUAL_MAX
        ).all()
    )
    candidate = torch.empty_like(D)
    # Write each site directly into its contiguous destination.  The former
    # broadcast einsum also materialized an [S, chunk(G), b, d] result before
    # copy_, adding tens of MiB to the live retraction peak on real cells.
    for site in range(D.shape[0]):
        torch.bmm(
            inverse_cholesky,
            D[site],
            out=candidate[site],
        )
    post_gram = _block_gram_no_grad(candidate)
    diagonal = torch.diagonal(post_gram, dim1=-2, dim2=-1)
    diagonal.sub_(1.0)
    post_residual = post_gram.norm(dim=(-2, -1))
    post_ok = (
        torch.isfinite(post_residual).all()
        & (post_residual <= CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX).all()
    )
    # Cholesky, factor, and post-Gram predicates are all transactional: the
    # master is untouched until their conjunction passes. Speculatively
    # completing a rejected candidate is cheap relative to training and lets
    # the admitted path pay one host fence instead of three. Detailed scalar
    # diagnostics execute only after that combined refusal.
    if not bool(precondition_ok & factor_ok & post_ok):
        if not bool(precondition_ok):
            failed = int((info != 0).sum())
            raise CholeskyQRRetractionError(
                "Cholesky-QR requires finite positive-definite decoder Grams "
                f"({failed} Cholesky failures)"
            )
        if not bool(factor_ok):
            max_condition = float(
                condition.nan_to_num(
                    nan=float("inf"), posinf=float("inf"), neginf=float("inf")
                ).max()
            )
            max_reconstruction = float(
                reconstruction_residual.nan_to_num(
                    nan=float("inf"), posinf=float("inf"), neginf=float("inf")
                ).max()
            )
            raise CholeskyQRRetractionError(
                "Cholesky-QR conditioning/reconstruction guard failed: "
                f"condition_inf={max_condition:.6g} "
                f"(max {CHOLESKY_QR_GRAM_CONDITION_MAX:g}), "
                f"relative_residual={max_reconstruction:.6g} "
                "(max "
                f"{CHOLESKY_QR_RECONSTRUCTION_RELATIVE_RESIDUAL_MAX:g})"
            )
        maximum = float(
            post_residual.nan_to_num(
                nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            ).max()
        )
        raise CholeskyQRRetractionError(
            "Cholesky-QR post-Gram guard failed: "
            f"residual={maximum:.6g} "
            f"(max {CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX:g})"
        )

    D.copy_(candidate)
    return torch.zeros((), dtype=torch.int64, device=D.device)


@torch.no_grad()
def cholesky_qr_retract_(D: torch.Tensor) -> int:
    """Public integer-count wrapper for guarded Cholesky-QR1."""

    return int(_cholesky_qr_retract_count_tensor_(D).item())


def site_singular_values(D: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Per-site decoder singular values via batched b x b Gram eigenvalues.

    No d-dimensional SVD anywhere in the loop (design: fp32, sqrt(eig + eps)).
    eigvalsh (eigenvalues only) keeps the backward well-defined for the
    symmetric functions of the spectrum we use, including near-degenerate
    spectra where eigenvector gradients would blow up.

    D: [S, G, b, d]  ->  [S, G, b]
    """
    if D.shape[1] > _SPECTRUM_UNCHUNKED_MAX:
        if D.is_cuda:
            # cuSOLVER reserves substantial workspace per tiny matrix.  A
            # fixed 256-block slice bounds that workspace while keeping both
            # the eigensolve and its backward pass on device; the former CPU
            # fallback paid a full synchronization and PCIe round-trip.
            chunks = []
            for start in range(0, D.shape[1], _SPECTRUM_CUDA_BLOCK_CHUNK):
                block = D[:, start : start + _SPECTRUM_CUDA_BLOCK_CHUNK].float()
                gram = torch.einsum("sgbd,sgcd->sgbc", block, block)
                flat = gram.reshape(-1, gram.shape[-2], gram.shape[-1])
                chunks.append(torch.linalg.eigvalsh(flat).reshape(gram.shape[:-1]))
            return (torch.cat(chunks, dim=1).clamp_min(0.0) + eps).sqrt()
        gram_chunks = []
        for start in range(0, D.shape[1], _SPECTRUM_BLOCK_CHUNK):
            block = D[:, start : start + _SPECTRUM_BLOCK_CHUNK].float()
            gram_chunks.append(torch.einsum("sgbd,sgcd->sgbc", block, block))
        gram = torch.cat(gram_chunks, dim=1)
        flat = gram.reshape(-1, gram.shape[-2], gram.shape[-1])
        # CUDA's batched solver reserves roughly 256 KiB per 4x4 matrix; with
        # optimizer state resident that dominates this small Gram tensor.  Move
        # every independently formed block Gram to CPU together so the exact
        # eigensolve needs one device round-trip rather than one per chunk.
        evals = torch.linalg.eigvalsh(flat.cpu()).to(D.device)
        evals = evals.reshape(gram.shape[:-1])
        return (evals.clamp_min(0.0) + eps).sqrt()

    D32 = D.float()
    gram_s = torch.einsum("sgbd,sgcd->sgbc", D32, D32)
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


def _map_nuclear_from_grams(
    md: torch.Tensor,
    me: torch.Tensor,
    *,
    block_dim: int,
    eps: float,
) -> torch.Tensor:
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    # Cholesky only the decoder Gram: if M_D = L L.T, the squared non-zero
    # singular values of Dbar.T @ Ebar are the eigenvalues of L.T M_E L.
    # Unlike a second Cholesky this remains defined for rank-deficient encoder
    # blocks, and unlike an eigendecomposition-based square root of M_D it
    # avoids undefined eigenvector gradients when the Grassmann constraint
    # deliberately makes all decoder-Gram eigenvalues equal to one.
    ld, info_d = torch.linalg.cholesky_ex(md)
    if bool((info_d != 0).any()):
        raise ValueError(
            "map nuclear regularization requires full-row-rank decoder blocks"
        )
    squared_singular_values = torch.linalg.eigvalsh(
        ld.transpose(-1, -2) @ me @ ld
    ).clamp_min(0.0)
    terms = (squared_singular_values + eps).sqrt()
    return (terms.sum(dim=-1) / block_dim).mean()


def map_nuclear_penalty(
    D: torch.Tensor, E: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """SASA end-to-end map nuclear penalty for concatenated site maps.

    For block ``g`` let ``Dbar`` and ``Ebar`` be the decoder and encoder
    matrices with all site dimensions concatenated.  SASA penalizes
    ``||Dbar.T @ Ebar||_*``.  Its non-zero squared singular values are the
    eigenvalues of ``sqrt(M_D) @ M_E @ sqrt(M_D)``, where ``M_D`` and ``M_E``
    are the two small ``b x b`` row Grams.  This implementation therefore
    remains exact for Gram-constrained, Frobenius-constrained, and completely
    free decoders without ever materializing an ``(S*d) x (S*d)`` map.
    """
    if D.shape != E.shape:
        raise ValueError(
            f"D and E must have identical shape, got {D.shape} and {E.shape}"
        )
    Df, Ef = D.float(), E.float()
    md = torch.einsum("sgbd,sgcd->gbc", Df, Df)
    me = torch.einsum("sgbd,sgcd->gbc", Ef, Ef)
    return _map_nuclear_from_grams(md, me, block_dim=E.shape[2], eps=eps)


def _validate_factorized_regularizer_inputs(
    site: torch.Tensor,
    core: torch.Tensor,
) -> None:
    if site.ndim != 2 or core.ndim != 4:
        raise ValueError("factorized regularizer tensors have invalid rank")
    if site.shape[1] != core.shape[0]:
        raise ValueError("factorized regularizer site/core ranks disagree")


def _factorized_core_pair_gram(core: torch.Tensor) -> torch.Tensor:
    value = core.float()
    return torch.einsum("rgbd,tgcd->rtgbc", value, value)


def factorized_map_nuclear_penalty(
    D_site: torch.Tensor,
    D_core: torch.Tensor,
    E_site: torch.Tensor,
    E_core: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map nuclear penalty directly from a rank-1/2 site factorization.

    The contraction is algebraically identical to materializing both
    ``[site,group,block,width]`` tensors, but casts the factors to fp32 before
    forming their pair Grams.  That avoids the large rounded bf16 full-tensor
    intermediates and is therefore a separately serialized implementation.
    """

    _validate_factorized_regularizer_inputs(D_site, D_core)
    _validate_factorized_regularizer_inputs(E_site, E_core)
    if D_site.shape != E_site.shape or D_core.shape != E_core.shape:
        raise ValueError("factorized map decoder and encoder shapes disagree")
    decoder_site_gram = D_site.float().transpose(0, 1) @ D_site.float()
    encoder_site_gram = E_site.float().transpose(0, 1) @ E_site.float()
    decoder_gram = torch.einsum(
        "rt,rtgbc->gbc",
        decoder_site_gram,
        _factorized_core_pair_gram(D_core),
    )
    encoder_gram = torch.einsum(
        "rt,rtgbc->gbc",
        encoder_site_gram,
        _factorized_core_pair_gram(E_core),
    )
    return _map_nuclear_from_grams(
        decoder_gram,
        encoder_gram,
        block_dim=D_core.shape[2],
        eps=eps,
    )


def _factorized_decoder_gram_eigenvalues(
    site: torch.Tensor,
    core: torch.Tensor,
) -> torch.Tensor:
    pair_gram = _factorized_core_pair_gram(core)
    gram = torch.einsum(
        "sr,st,rtgbc->sgbc",
        site.float(),
        site.float(),
        pair_gram,
    )
    flat = gram.reshape(-1, gram.shape[-2], gram.shape[-1])
    if flat.shape[0] > _EIGH_MAX_BATCH:
        evals = torch.cat(
            [
                torch.linalg.eigvalsh(flat[i : i + _EIGH_MAX_BATCH])
                for i in range(0, flat.shape[0], _EIGH_MAX_BATCH)
            ]
        )
    else:
        evals = torch.linalg.eigvalsh(flat)
    return evals.reshape(gram.shape[:-1])


def factorized_decoder_nuclear_penalty(
    D_site: torch.Tensor,
    D_core: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-site decoder nuclear penalty without full tensor materialization."""

    _validate_factorized_regularizer_inputs(D_site, D_core)
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    if D_core.shape[1] > _SPECTRUM_UNCHUNKED_MAX:
        chunks = [
            _factorized_decoder_gram_eigenvalues(
                D_site,
                D_core[:, start : start + _SPECTRUM_CUDA_BLOCK_CHUNK],
            )
            for start in range(
                0,
                D_core.shape[1],
                _SPECTRUM_CUDA_BLOCK_CHUNK,
            )
        ]
        eigenvalues = torch.cat(chunks, dim=1)
    else:
        eigenvalues = _factorized_decoder_gram_eigenvalues(D_site, D_core)
    return (eigenvalues.clamp_min(0.0) + eps).sqrt().sum(dim=-1).mean()


def decoder_nuclear_penalty(D: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Mean decoder-block nuclear norm used by the inspected SASA release.

    This is intentionally separate from SASA's paper objective
    ``||D_g.T @ E_g||_*``.  For multi-site tensors the mean is over every
    site/block pair; the released-code bridge itself is single-site.
    """

    return site_singular_values(D, eps=eps).sum(dim=-1).mean()


@torch.no_grad()
def _project_block_frobenius_count_tensor_(
    D: torch.Tensor,
    *,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """Project concatenated decoder blocks onto a Frobenius ball.

    At S=1 this is Fel's Vanilla-BSF decoder constraint. For S>1 it is the
    direct concatenated-site extension. Returns the number of clipped blocks.
    """
    norms = D.float().pow(2).sum(dim=(0, 2, 3)).sqrt()  # [G]
    scale = (max_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
    D.mul_(scale.to(D.dtype).view(1, -1, 1, 1))
    return (norms > max_norm).sum()


@torch.no_grad()
def project_block_frobenius_(D: torch.Tensor, *, max_norm: float = 1.0) -> int:
    """Public integer-count wrapper for the concatenated block projection."""
    return int(_project_block_frobenius_count_tensor_(D, max_norm=max_norm).item())


@torch.no_grad()
def _normalize_block_frobenius_count_tensor_(
    D: torch.Tensor,
    *,
    target_norm: float = 1.0,
) -> torch.Tensor:
    """Normalize every site-concatenated decoder block to one Frobenius norm.

    This is the exact scale control used by the pinned BSF release trainer for
    its free-decoder Vanilla and Group-Lasso implementations.  It is distinct
    from the paper's Vanilla unit-ball projection: blocks below the target are
    expanded here and left unchanged by :func:`project_block_frobenius_`.
    """

    if target_norm <= 0:
        raise ValueError("target_norm must be positive")
    norms = D.float().pow(2).sum(dim=(0, 2, 3)).sqrt()
    D.mul_((target_norm / norms.clamp_min(1e-12)).to(D.dtype).view(1, -1, 1, 1))
    return (norms > 0).sum()


@torch.no_grad()
def normalize_block_frobenius_(D: torch.Tensor, *, target_norm: float = 1.0) -> int:
    """Public integer-count wrapper for equality Frobenius normalization."""
    return int(
        _normalize_block_frobenius_count_tensor_(D, target_norm=target_norm).item()
    )


@torch.no_grad()
def _project_latent_rows_count_tensor_(
    W: torch.Tensor,
    *,
    target_norm: float = 1.0,
) -> torch.Tensor:
    """Normalize every scalar latent row over its input/output coordinates.

    ``W`` has shape ``[site, block, coordinate, activation]``.  This matches
    the inspected SASA implementation's decoder-row and encoder-column
    normalization after translating its ``[activation, latent]`` encoder
    orientation into this package's row-major latent orientation.
    """

    norms = W.float().norm(dim=-1, keepdim=True)
    nonzero = norms > 0
    scale = torch.where(
        nonzero,
        torch.full_like(norms, target_norm) / norms.clamp_min(1e-12),
        torch.ones_like(norms),
    )
    W.mul_(scale.to(W.dtype))
    return nonzero.sum()


@torch.no_grad()
def project_latent_rows_(W: torch.Tensor, *, target_norm: float = 1.0) -> int:
    """Public integer-count wrapper for latent-row normalization."""
    return int(_project_latent_rows_count_tensor_(W, target_norm=target_norm).item())


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
    preconditioning: str = "concatenated_gram_retraction",
) -> torch.Tensor:
    """Seeded ``N(0, 1/d)`` init with an explicit optional preconditioner."""
    if preconditioning not in {"concatenated_gram_retraction", "none"}:
        raise ValueError(f"unsupported decoder preconditioning {preconditioning!r}")
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
    if preconditioning == "concatenated_gram_retraction":
        retract_(D)
    return D
