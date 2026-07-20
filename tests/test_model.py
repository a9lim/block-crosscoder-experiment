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
    token_topk_mask,
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
    # Fractional budget (the capture-sweep under-provisioned regime):
    # round(k*B) selections batch-wide.
    frac = batch_topk_mask(scores, 0.8)
    assert int(frac.sum().item()) == round(0.8 * B)


def test_token_topk_exact_per_token_count(device):
    scores = torch.rand(64, CFG.n_blocks, device=device)
    mask = token_topk_mask(scores, 3)
    assert torch.equal(mask.sum(dim=1), torch.full((64,), 3, device=device))


def test_encoder_bias_breaks_antipodal_support(device):
    model = make_model(device, encoder_bias=True, selection="token_topk")
    with torch.no_grad():
        model.a.normal_(mean=0.5, std=0.1)
    x = whitened_batch(device, n=64)
    assert not torch.allclose(model.scores(model.encode(x)), model.scores(model.encode(-x)))


def test_tied_grassmannian_uses_single_gamma(device):
    model = make_model(device, encoder_mode="tied")
    assert model.E is None and model.log_gamma.shape == ()
    x = whitened_batch(device, n=8)
    expected = torch.einsum("bsd,sgkd->bgk", x, model.D) * model.log_gamma.exp()
    assert torch.allclose(model.encode(x), expected, atol=1e-5)


def test_relu_dense_crosscoder_bridge(device):
    model = make_model(
        device, block_dim=1, code_activation="relu", selection="dense",
        regularizer="crosscoder_l1", lambda_regularizer=1e-4, encoder_bias=True,
        decoder_constraint="frobenius",
    )
    x = whitened_batch(device, n=32)
    out = model(x)
    assert (out.z >= 0).all()
    assert torch.equal(out.mask, out.scores > 0)
    parts = bsc_loss(out, x, model)
    assert parts["regularizer"] >= 0


def test_decoder_weighted_batchtopk_score_matches_minder(device):
    model = make_model(
        device, block_dim=1, code_activation="relu",
        selection_score="decoder_weighted", decoder_constraint="free",
    )
    x = whitened_batch(device, n=32)
    z = model.encode(x)
    expected = z.squeeze(-1) * model.D.float().norm(dim=-1).squeeze(-1).sum(dim=0)
    assert torch.allclose(model.scores(z), expected.to(z.dtype), atol=1e-5)


def test_group_lasso_bridge_has_positive_learned_threshold(device):
    model = make_model(
        device, selection="dense", code_activation="group_soft_threshold",
        decoder_constraint="free", regularizer="group_l21",
        lambda_regularizer=1e-3,
        encoder_bias=True,
    )
    x = whitened_batch(device, n=32)
    out = model(x)
    assert torch.nn.functional.softplus(model.log_threshold).min() > 0
    assert torch.equal(out.mask, out.scores > 0)
    parts = bsc_loss(out, x, model)
    parts["total"].backward()
    assert model.log_threshold.grad is not None


def test_free_decoder_projection_is_noop(device):
    model = make_model(device, decoder_constraint="free")
    before = model.D.detach().clone()
    assert model.project_decoder_() == 0
    assert torch.equal(model.D, before)


def test_frobenius_decoder_projection(device):
    model = make_model(device, decoder_constraint="frobenius")
    with torch.no_grad():
        model.D.mul_(5)
    hits = model.project_decoder_()
    norms = model.D.float().pow(2).sum(dim=(0, 2, 3)).sqrt()
    assert hits > 0 and norms.max() <= 1 + 1e-5


def test_map_nuclear_regularizer(device):
    model = make_model(
        device, regularizer="map_nuclear", lambda_regularizer=1e-3
    )
    x = whitened_batch(device, n=32)
    parts = bsc_loss(model(x), x, model)
    assert parts["regularizer"] > 0
    parts["total"].backward()
    assert model.E.grad is not None


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


def test_fixed_threshold_training_selector_has_variable_counts(device):
    model = make_model(device, selection="threshold")
    x = whitened_batch(device, n=128)
    model.fit_threshold_([x], target_avg_blocks=CFG.k, method="exact")
    out = model(x)
    assert abs(float(out.mask.float().sum(dim=1).mean()) - CFG.k) < 0.1
    assert out.mask.sum(dim=1).min() != out.mask.sum(dim=1).max()


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


def test_exact_k_planted_model_matches_fel_support():
    from block_crosscoder_experiment.synthetic import (
        BlockSpec,
        ExactKPlantedModel,
    )

    truth = ExactKPlantedModel(
        [BlockSpec(rank=1, frequency=0.25) for _ in range(8)],
        n_sites=1, d_model=16, block_dim=2, active_per_sample=2,
    )
    batch = truth.sample(128, seed=3)
    assert torch.equal(batch.active.sum(dim=1), torch.full((128,), 2))


def test_train_smoke_loss_decreases(device):
    """Full ordering on tiny data: optimizer step -> retract -> next step.
    Loss must fall and the constraint must hold at every step."""
    torch.manual_seed(0)
    model = make_model(device, lambda_regularizer=1e-3)
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


# -- streaming theta quantile ------------------------------------------------


def test_streaming_quantile_matches_exact_kthvalue(device):
    from block_crosscoder_experiment.model import StreamingScoreQuantile

    gen = torch.Generator().manual_seed(5)
    scores = torch.rand(200_000, generator=gen).pow(2) * 30  # skewed, positive
    hist = StreamingScoreQuantile(device=device)
    for chunk in scores.to(device).split(4096):
        hist.update(chunk)
    for q in (0.5, 0.99, 0.9921875):  # last = 1 - 32/4096, the pilot target
        n = scores.numel()
        idx = min(max(int(round(q * n)), 1), n)
        exact = float(scores.kthvalue(idx).values)
        assert abs(hist.quantile(q) - exact) / exact < 1e-3


def test_streaming_quantile_batch_order_independent(device):
    from block_crosscoder_experiment.model import StreamingScoreQuantile

    gen = torch.Generator().manual_seed(6)
    chunks = [torch.rand(1000, generator=gen) * 50 for _ in range(20)]
    a, b = StreamingScoreQuantile(device=device), StreamingScoreQuantile(device=device)
    for c in chunks:
        a.update(c.to(device))
    for c in reversed(chunks):
        b.update(c.to(device))
    assert torch.equal(a.counts, b.counts)
    assert a.quantile(0.99) == b.quantile(0.99)


def test_streaming_quantile_rejects_nonfinite(device):
    from block_crosscoder_experiment.model import StreamingScoreQuantile

    hist = StreamingScoreQuantile(device=device)
    bad = torch.tensor([1.0, float("nan")], device=device)
    with pytest.raises(ValueError, match="non-finite"):
        hist.update(bad)


def test_fit_threshold_streaming_matches_exact(device):
    """The E1 validation gate in miniature: theta and realized avg-blocks
    agree between methods on the same calibration batches."""
    model = BlockCrosscoder(CFG).to(device)
    gen = torch.Generator().manual_seed(7)
    calib = [
        torch.randn(512, CFG.n_sites, CFG.d_model, generator=gen).to(device)
        for _ in range(8)
    ]
    theta_exact = model.fit_threshold_(calib, float(CFG.k), method="exact")
    counts_exact = torch.cat(
        [model(x, mode="threshold").mask.sum(dim=1).float() for x in calib]
    )
    theta_stream = model.fit_threshold_(calib, float(CFG.k), method="streaming")
    counts_stream = torch.cat(
        [model(x, mode="threshold").mask.sum(dim=1).float() for x in calib]
    )
    assert abs(theta_stream - theta_exact) / theta_exact < 1e-3
    assert abs(float(counts_stream.mean()) - float(counts_exact.mean())) <= 0.1
