"""Numeric checks that the Gram constraint has its four required properties:
scale-gauge death, O(b)-invariant spectra, exact selection scores, and
free per-site Frobenius shares."""

import pytest
import torch

import block_crosscoder_experiment.gram as gram_module

from block_crosscoder_experiment.gram import (
    block_gram,
    gram_residual,
    init_decoder_stack,
    map_nuclear_penalty,
    project_block_frobenius_,
    retract_,
    site_frobenius_shares,
    site_singular_values,
)

S, G, B_DIM, D_MODEL = 4, 16, 4, 32


def random_stack(device, seed=0, scale=1.0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    D = torch.randn(S, G, B_DIM, D_MODEL, generator=gen) * scale
    return D.to(device)


def random_orthogonal(n, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device)


def test_retraction_satisfies_constraint(device):
    D = random_stack(device, scale=3.0)
    retract_(D)
    assert gram_residual(D).max().item() < 1e-5


def test_retraction_idempotent(device):
    D = random_stack(device)
    retract_(D)
    before = D.clone()
    retract_(D)
    assert (D - before).abs().max().item() < 1e-5


def test_retraction_kills_scale_gauge(device):
    """D and c*D retract to the same point — the z->cz gauge is dead."""
    D1 = random_stack(device)
    D2 = D1 * 7.3
    retract_(D1)
    retract_(D2)
    assert (D1 - D2).abs().max().item() < 1e-4


def test_retraction_requires_fp32(device):
    D = random_stack(device).to(torch.bfloat16)
    with pytest.raises(TypeError):
        retract_(D)


def test_retraction_floor_hits_on_deficient_block(device):
    D = random_stack(device)
    D[:, 0] = 0.0
    D[0, 0, 0, 0] = 1.0  # block 0: rank 1 across all sites
    floor_hits = retract_(D)
    assert floor_hits >= B_DIM - 1
    assert torch.isfinite(D).all()
    # Healthy blocks still land on the constraint.
    assert gram_residual(D)[1:].max().item() < 1e-5


def test_site_shares_sum_to_one_and_start_equal(device):
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    shares = site_frobenius_shares(D)  # [S, G]
    assert torch.allclose(shares.sum(dim=0), torch.ones(G, device=device), atol=1e-5)
    # Gaussian init + one retraction: approximately equal shares (1/S).
    assert (shares.mean(dim=1) - 1 / S).abs().max().item() < 0.05


def site_exclusive_stack(device):
    """Constraint-satisfying stack with each code direction on one site:
    directions 0,1 -> site 0; directions 2,3 -> site 1 (b=4)."""
    gen = torch.Generator(device="cpu").manual_seed(3)
    D = torch.zeros(S, G, B_DIM, D_MODEL)
    for g in range(G):
        q, _ = torch.linalg.qr(torch.randn(D_MODEL, B_DIM, generator=gen))
        rows = q.T  # [b, d] orthonormal rows
        D[0, g, 0] = rows[0]
        D[0, g, 1] = rows[1]
        D[1, g, 2] = rows[2]
        D[1, g, 3] = rows[3]
    return D.to(device)


def test_unequal_shares_preserved(device):
    """The constraint fixes only the total; the depth profile is free."""
    D = site_exclusive_stack(device)
    assert gram_residual(D).max().item() < 1e-5
    before = D.clone()
    retract_(D)
    assert (D - before).abs().max().item() < 1e-5
    shares = site_frobenius_shares(D)
    expected = torch.tensor([0.5, 0.5, 0.0, 0.0], device=device)
    assert torch.allclose(shares[:, 0], expected, atol=1e-5)


def test_map_nuclear_matches_explicit_end_to_end_map(device):
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    E = random_stack(device, seed=21)
    actual = map_nuclear_penalty(D, E, eps=0.0)
    explicit = []
    for g in range(G):
        dbar = D[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        ebar = E[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        explicit.append(torch.linalg.svdvals(dbar.T @ ebar).sum() / B_DIM)
    assert torch.allclose(actual, torch.stack(explicit).mean(), atol=1e-4)


def test_map_nuclear_matches_explicit_for_unconstrained_decoder(device):
    D = random_stack(device, seed=24, scale=0.3)
    E = random_stack(device, seed=25, scale=0.2)
    actual = map_nuclear_penalty(D, E, eps=0.0)
    explicit = []
    for g in range(G):
        dbar = D[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        ebar = E[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        explicit.append(torch.linalg.svdvals(dbar.T @ ebar).sum() / B_DIM)
    assert torch.allclose(actual, torch.stack(explicit).mean(), atol=2e-4)


def test_map_nuclear_exact_zero_smoothing_has_finite_grassmann_gradient(device):
    # The concatenated Gram constraint repeats every decoder-Gram eigenvalue
    # at one.  The exact SASA objective must therefore avoid eigendecomposition
    # eigenvector gradients, which are undefined at this intentional
    # degeneracy.
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device).requires_grad_()
    E = random_stack(device, seed=29).requires_grad_()
    loss = map_nuclear_penalty(D, E, eps=0.0)
    loss.backward()
    assert D.grad is not None and torch.isfinite(D.grad).all()
    assert E.grad is not None and torch.isfinite(E.grad).all()


def test_map_nuclear_accepts_rank_deficient_encoder(device):
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    E = random_stack(device, seed=31)
    E[:, :, 1:] = 0.0
    actual = map_nuclear_penalty(D, E, eps=0.0)
    explicit = []
    for g in range(G):
        dbar = D[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        ebar = E[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        explicit.append(torch.linalg.svdvals(dbar.T @ ebar).sum() / B_DIM)
    assert torch.allclose(actual, torch.stack(explicit).mean(), atol=1e-4)


def test_frobenius_projection(device):
    D = random_stack(device, scale=3.0)
    hits = project_block_frobenius_(D)
    norms = D.float().pow(2).sum(dim=(0, 2, 3)).sqrt()
    assert hits == G
    assert norms.max() <= 1.0 + 1e-5


def test_site_singular_values_casts_before_gram(device):
    D = random_stack(device, seed=22)
    expected = site_singular_values(D)
    actual = site_singular_values(D.to(torch.bfloat16))
    # The only difference is the input parameter cast, not bf16 accumulation.
    reference = site_singular_values(D.to(torch.bfloat16).float())
    assert torch.equal(actual, reference)
    assert torch.allclose(actual, expected, atol=2e-2)


def test_o_b_invariance(device):
    """A per-block O(b) rotation leaves constraint and spectra unchanged."""
    D = random_stack(device, seed=6)
    retract_(D)
    R = random_orthogonal(B_DIM, device, seed=7)
    D_rot = torch.einsum("bc,sgcd->sgbd", R, D)
    assert gram_residual(D_rot).max().item() < 1e-4
    sv, sv_rot = site_singular_values(D), site_singular_values(D_rot)
    assert (sv - sv_rot).abs().max().item() < 1e-4


def test_block_gram_matches_naive(device):
    D = random_stack(device, seed=8)
    M = block_gram(D)
    g = 3
    naive = torch.stack([D[s, g] @ D[s, g].T for s in range(S)]).sum(dim=0)
    assert torch.allclose(M[g], naive, atol=1e-5)


def test_chunked_gram_and_retraction_match(monkeypatch, device):
    D = random_stack(device, seed=9)
    expected = torch.einsum("sgbd,sgcd->gbc", D, D)
    monkeypatch.setattr(gram_module, "_GRAM_BLOCK_CHUNK", 3)
    monkeypatch.setattr(gram_module, "_RETRACT_UNCHUNKED_MAX", 0)
    assert torch.allclose(block_gram(D), expected, atol=1e-5)
    retract_(D)
    assert gram_residual(D).max().item() < 1e-5


def test_chunked_site_spectrum_matches_and_has_grad(monkeypatch, device):
    D = random_stack(device, seed=10)
    expected = site_singular_values(D)
    monkeypatch.setattr(gram_module, "_SPECTRUM_BLOCK_CHUNK", 3)
    monkeypatch.setattr(gram_module, "_SPECTRUM_UNCHUNKED_MAX", 0)
    D.requires_grad_(True)
    actual = site_singular_values(D)
    assert torch.allclose(actual, expected, atol=1e-5)
    actual.sum().backward()
    assert D.grad is not None and torch.isfinite(D.grad).all()
