"""Model-level checks: the selection-score identity, BatchTopK exactness
and gradient masking, init conventions, and a smoke test of the full
step -> retract loop on tiny synthetic data."""

import pytest
import torch

from block_crosscoder_experiment.gram import gram_residual, retract_
from block_crosscoder_experiment.model import (
    BlockCrosscoder,
    BSCConfig,
    batch_topk_mask,
    bsc_loss,
)

CFG = BSCConfig(n_blocks=16, block_dim=4, n_sites=4, d_model=32, k=3, seed=0)


def make_model(device, **overrides):
    cfg = BSCConfig(**{**CFG.__dict__, **overrides})
    return BlockCrosscoder(cfg).to(device)


def whitened_batch(device, n=512, seed=1):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, CFG.n_sites, CFG.d_model, generator=gen).to(device)


def test_selection_score_identity(device):
    """Under the constraint, ||z_g||^2 is exactly the block's contribution
    energy sum_s ||D_g^s^T z_g||^2 — the score is not a proxy."""
    model = make_model(device)
    z = model.encode(whitened_batch(device))  # [B, G, b]
    contrib = torch.einsum("bgk,sgkd->bsgd", z, model.D)
    energy = contrib.pow(2).sum(dim=(1, 3))  # [B, G]
    assert torch.allclose(energy, z.pow(2).sum(dim=-1), rtol=1e-4, atol=1e-5)


def test_encode_matches_naive(device):
    model = make_model(device)
    x = whitened_batch(device, n=8)
    z = model.encode(x)
    naive = torch.einsum("bsd,sgkd->bgk", x, model.E)
    assert torch.allclose(z, naive, rtol=1e-4, atol=1e-5)


def test_decode_matches_naive_and_bias(device):
    model = make_model(device)
    z = torch.randn(8, CFG.n_blocks, CFG.block_dim, device=device)
    xhat = model.decode(z)
    naive = torch.einsum("bgk,sgkd->bsd", z, model.D) + model.c
    assert torch.allclose(xhat, naive, rtol=1e-4, atol=1e-5)
    # Zero code decodes to the bias.
    zero = model.decode(torch.zeros_like(z))
    assert torch.allclose(zero, model.c.expand_as(zero), atol=1e-6)


def test_batchtopk_exact_count_and_variable_per_token(device):
    B, G = 64, CFG.n_blocks
    gen = torch.Generator(device="cpu").manual_seed(2)
    scores = torch.randn(B, G, generator=gen).abs().to(device)
    mask = batch_topk_mask(scores, CFG.k)
    assert int(mask.sum().item()) == CFG.k * B
    per_token = mask.sum(dim=1).float()
    assert per_token.min() != per_token.max()  # counts vary by design
    # Kept scores dominate dropped scores globally.
    assert scores[mask].min() >= scores[~mask].max()


def test_gradient_only_through_selected(device):
    model = make_model(device)
    x = whitened_batch(device, n=64)
    z = model.encode(x)
    z_leaf = z.detach().requires_grad_(True)
    mask = model.select(z_leaf)
    out = model.decode(z_leaf * mask.unsqueeze(-1))
    out.pow(2).sum().backward()
    grad_norms = z_leaf.grad.norm(dim=-1)  # [B, G]
    assert grad_norms[~mask].max().item() == 0.0
    assert grad_norms[mask].min().item() > 0.0


def test_threshold_mode_requires_calibration(device):
    model = make_model(device)
    x = whitened_batch(device, n=8)
    with pytest.raises(RuntimeError):
        model(x, mode="threshold")
    model.theta.fill_(model.scores(model.encode(x)).median())
    out = model(x, mode="threshold")
    assert out.mask.any() and not out.mask.all()


def test_init_tied_and_score_comparability(device):
    model = make_model(device)
    assert torch.equal(model.E, model.D)  # transpose-tied at init
    x = whitened_batch(device, n=2048)
    model.calibrate_encoder_scale_(x)
    p = model.scores(model.encode(x)).mean(dim=0)  # [G]
    spread = (p.max() / p.min()).item()
    assert spread < 1.01  # per-block means equalized on the calib batch


def planted_lowrank_batch(device, n=1024, rank=8, seed=3):
    """Data with structure a sparse model can actually fit: a low-rank
    linear source (rank < k*b) plus small noise — a micro-preview of the
    Phase -1 generator."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(n, rank, generator=gen)
    P = torch.randn(rank, CFG.n_sites * CFG.d_model, generator=gen) / rank**0.5
    x = (u @ P).view(n, CFG.n_sites, CFG.d_model)
    x = x + 0.01 * torch.randn(x.shape, generator=gen)
    return x.to(device)


def test_train_smoke_loss_decreases(device):
    """Full ordering on tiny data: optimizer step -> retract -> next step.
    Loss must fall and the constraint must hold at every step."""
    torch.manual_seed(0)
    model = make_model(device, lambda_rank=1e-3)
    x = planted_lowrank_batch(device)
    opt = torch.optim.Adam(
        [
            {"params": [model.E], "weight_decay": 1e-4},
            {"params": [model.D, model.c], "weight_decay": 0.0},  # decoders: 0
        ],
        lr=3e-3,
    )
    losses = []
    for _ in range(60):
        out = model(x)
        parts = bsc_loss(out, x, model)
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
        retract_(model.D.data, eig_floor=model.cfg.eig_floor)
        losses.append(parts["rec"].item())
        assert gram_residual(model.D).max().item() < 1e-4
    assert losses[-1] < 0.5 * losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))
