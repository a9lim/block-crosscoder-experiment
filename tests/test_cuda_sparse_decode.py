"""CUDA release gates for the low-density hard-TopK decoder."""

import copy

import pytest
import torch

pytest.importorskip("triton")

from block_crosscoder_experiment.cuda_sparse_decode import cuda_sparse_topk_decode
from block_crosscoder_experiment.gram import gram_residual
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder
from block_crosscoder_experiment.trainer import TrainConfig, Trainer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = (actual.float() - expected.float()).square().sum().sqrt()
    scale = expected.float().square().sum().sqrt().clamp_min(1e-30)
    return float((delta / scale).detach())


def _fixed_mask(batch: int, groups: int, active: int) -> torch.Tensor:
    rows = torch.arange(batch, device="cuda").unsqueeze(1)
    offsets = torch.arange(active, device="cuda").unsqueeze(0)
    selected = (rows * 17 + offsets * 13).remainder(groups)
    return torch.zeros(batch, groups, dtype=torch.bool, device="cuda").scatter_(
        1,
        selected,
        True,
    )


@pytest.mark.parametrize("output_width", (48, 96, 192))
def test_cuda_sparse_topk_decode_tracks_dense_forward_backward(output_width):
    batch, groups, block_dim, active = 64, 64, 3, 2
    generator = torch.Generator(device="cuda").manual_seed(1901 + output_width)
    initial_code = torch.randn(
        batch,
        groups,
        block_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    initial_weight = torch.randn(
        groups * block_dim,
        output_width,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target = torch.randn(
        batch,
        output_width,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mask = _fixed_mask(batch, groups, active)

    dense_code = initial_code.detach().clone().requires_grad_(True)
    dense_weight = initial_weight.detach().clone().requires_grad_(True)
    dense = (dense_code * mask.unsqueeze(-1)).reshape(batch, -1) @ dense_weight
    (dense.float() - target.float()).square().mean().backward()

    sparse_code = initial_code.detach().clone().requires_grad_(True)
    sparse_weight = initial_weight.detach().clone().requires_grad_(True)
    sparse = cuda_sparse_topk_decode(
        sparse_code,
        mask,
        sparse_weight,
        selected_count=batch * active,
    )
    (sparse.float() - target.float()).square().mean().backward()

    assert _relative_l2(sparse, dense) <= 4e-3
    assert _relative_l2(sparse_code.grad, dense_code.grad) <= 4e-3
    assert _relative_l2(sparse_weight.grad, dense_weight.grad) <= 5e-3
    assert torch.equal(
        sparse_code.grad[~mask], torch.zeros_like(sparse_code.grad[~mask])
    )
    assert torch.equal(
        cuda_sparse_topk_decode(
            initial_code,
            mask,
            initial_weight,
            selected_count=batch * active,
        ),
        cuda_sparse_topk_decode(
            initial_code,
            mask,
            initial_weight,
            selected_count=batch * active,
        ),
    )


@pytest.mark.parametrize(("sites", "site_width"), ((1, 127), (4, 33), (6, 17)))
def test_cuda_sparse_topk_native_decoder_tracks_dense_forward_backward(
    sites,
    site_width,
):
    batch, groups, block_dim, active = 64, 64, 4, 2
    generator = torch.Generator(device="cuda").manual_seed(
        1902 + sites * 100 + site_width
    )
    initial_code = torch.randn(
        batch,
        groups,
        block_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    initial_decoder = torch.randn(
        sites,
        groups,
        block_dim,
        site_width,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target = torch.randn(
        batch,
        sites,
        site_width,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mask = _fixed_mask(batch, groups, active)

    dense_code = initial_code.detach().clone().requires_grad_(True)
    dense_decoder = initial_decoder.detach().clone().requires_grad_(True)
    dense = torch.einsum(
        "bgi,sgid->bsd",
        dense_code * mask.unsqueeze(-1),
        dense_decoder,
    )
    dense_loss = (dense.float() - target.float()).square().mean()
    dense_loss.backward()

    sparse_code = initial_code.detach().clone().requires_grad_(True)
    sparse_decoder = initial_decoder.detach().clone().requires_grad_(True)
    sparse = cuda_sparse_topk_decode(
        sparse_code,
        mask,
        sparse_decoder,
        selected_count=batch * active,
    )
    sparse_loss = (sparse.float() - target.float()).square().mean()
    sparse_loss.backward()

    assert _relative_l2(sparse, dense) <= 4e-3
    assert _relative_l2(sparse_code.grad, dense_code.grad) <= 4e-3
    assert _relative_l2(sparse_decoder.grad, dense_decoder.grad) <= 5e-3
    loss_drift = abs(float((sparse_loss - dense_loss).detach())) / max(
        float(dense_loss.detach()),
        1e-30,
    )
    assert loss_drift <= 5e-4
    assert torch.equal(
        sparse_code.grad[~mask], torch.zeros_like(sparse_code.grad[~mask])
    )
    specialized_code = initial_code.detach().clone().requires_grad_(True)
    specialized_decoder = initial_decoder.detach().clone().requires_grad_(True)
    specialized = cuda_sparse_topk_decode(
        specialized_code,
        mask,
        specialized_decoder,
        selected_count=batch * active,
        events_per_row=active,
    )
    generic_code = initial_code.detach().clone().requires_grad_(True)
    generic_decoder = initial_decoder.detach().clone().requires_grad_(True)
    generic = cuda_sparse_topk_decode(
        generic_code,
        mask,
        generic_decoder,
        selected_count=batch * active,
    )
    probe = torch.randn_like(generic)
    specialized.backward(probe)
    generic.backward(probe)
    assert torch.equal(specialized, generic)
    assert torch.equal(specialized_code.grad, generic_code.grad)
    assert torch.equal(specialized_decoder.grad, generic_decoder.grad)


def test_sparse_factorized_trainer_has_bounded_trajectory_drift(monkeypatch):
    cfg = BSCConfig(
        n_blocks=64,
        block_dim=2,
        n_sites=4,
        d_model=32,
        k=2,
        seed=1911,
        selection="batch_topk",
        decoder_constraint="free",
        site_rank=1,
    )
    training = TrainConfig(
        total_steps=8,
        lr=3e-4,
        warmup_steps=1,
        schedule="cosine",
        forward_dtype="bf16",
        optimizer="adamw",
        fused=True,
        aux_variant="none",
        log_every=1,
    )
    base = BlockCrosscoder(cfg).cuda()
    sparse = Trainer(copy.deepcopy(base), training)
    dense = Trainer(copy.deepcopy(base), training)
    dense.fwd._cuda_sparse_topk_decode_shape_eligible = lambda **_kwargs: False
    generator = torch.Generator(device="cuda").manual_seed(1912)
    batches = [
        torch.randn(
            2048,
            cfg.n_sites,
            cfg.d_model,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(training.total_steps)
    ]
    sparse_records = [sparse.step(batch) for batch in batches]
    dense_records = [dense.step(batch) for batch in batches]

    for sparse_record, dense_record in zip(sparse_records, dense_records, strict=True):
        assert sparse_record is not None and dense_record is not None
        assert (
            abs(sparse_record["rec"] - dense_record["rec"])
            / max(
                abs(dense_record["rec"]),
                1e-30,
            )
            <= 5e-3
        )
    for sparse_parameter, dense_parameter in zip(
        sparse.master.parameters(),
        dense.master.parameters(),
        strict=True,
    ):
        assert _relative_l2(sparse_parameter, dense_parameter) <= 5e-3


@pytest.mark.parametrize("encoder_mode", ("tied", "untied"))
def test_sparse_unfactorized_qr_trainer_has_bounded_trajectory_drift(
    encoder_mode,
):
    cfg = BSCConfig(
        n_blocks=64,
        block_dim=2,
        n_sites=4,
        d_model=32,
        k=2,
        seed=1921,
        selection="token_topk",
        encoder_mode=encoder_mode,
        decoder_constraint="qr",
    )
    training = TrainConfig(
        total_steps=8,
        lr=3e-4,
        warmup_steps=1,
        schedule="cosine",
        forward_dtype="bf16",
        optimizer="adamw",
        fused=True,
        aux_variant="none",
        retract_every=1,
        log_every=1,
    )
    base = BlockCrosscoder(cfg).cuda()
    sparse = Trainer(copy.deepcopy(base), training)
    dense = Trainer(copy.deepcopy(base), training)
    dense.fwd._cuda_sparse_topk_decode_shape_eligible = lambda **_kwargs: False
    generator = torch.Generator(device="cuda").manual_seed(1922)
    intersections = 0
    unions = 0
    for _ in range(training.total_steps):
        batch = torch.randn(
            2048,
            cfg.n_sites,
            cfg.d_model,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        sparse_record = sparse.step(batch)
        dense_record = dense.step(batch)
        assert sparse_record is not None and dense_record is not None
        assert (
            abs(sparse_record["rec"] - dense_record["rec"])
            / max(abs(dense_record["rec"]), 1e-30)
            <= 5e-3
        )
        with torch.no_grad():
            sparse_mask = sparse.fwd.select_with_materialized(batch)[0].mask
            dense_mask = dense.fwd.select_with_materialized(batch)[0].mask
        intersections += int((sparse_mask & dense_mask).sum())
        unions += int((sparse_mask | dense_mask).sum())

    assert intersections / max(unions, 1) >= 0.98
    parameter_delta_sq = torch.zeros((), device="cuda")
    parameter_scale_sq = torch.zeros((), device="cuda")
    for sparse_parameter, dense_parameter in zip(
        sparse.master.parameters(),
        dense.master.parameters(),
        strict=True,
    ):
        parameter_delta_sq += (
            sparse_parameter.float() - dense_parameter.float()
        ).square().sum()
        parameter_scale_sq += dense_parameter.float().square().sum()
    parameter_drift = (
        parameter_delta_sq.sqrt() / parameter_scale_sq.sqrt().clamp_min(1e-30)
    )
    assert float(parameter_drift) <= 0.05
    assert float(gram_residual(sparse.master.decoder_tensor()).max()) <= 1e-4


def test_sparse_unfactorized_model_preserves_bias_padding_and_support():
    cfg = BSCConfig(
        n_blocks=64,
        block_dim=2,
        n_sites=4,
        d_model=32,
        site_dims=(31, 32, 30, 29),
        k=2,
        seed=1931,
        selection="token_topk",
        encoder_mode="untied",
        decoder_constraint="qr",
        decoder_bias=True,
    )
    sparse = BlockCrosscoder(cfg).cuda().to(torch.bfloat16).eval()
    dense = copy.deepcopy(sparse)
    dense._cuda_sparse_topk_decode_shape_eligible = lambda **_kwargs: False
    generator = torch.Generator(device="cuda").manual_seed(1932)
    batch = torch.randn(
        2048,
        cfg.n_sites,
        cfg.d_model,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sparse_output = sparse(batch)
    dense_output = dense(batch)
    assert torch.equal(sparse_output.mask, dense_output.mask)
    assert _relative_l2(sparse_output.xhat, dense_output.xhat) <= 4e-3
    coordinate_mask = sparse.coordinate_mask[:, 0, 0]
    assert torch.equal(
        sparse_output.xhat[:, ~coordinate_mask],
        torch.zeros_like(sparse_output.xhat[:, ~coordinate_mask]),
    )
