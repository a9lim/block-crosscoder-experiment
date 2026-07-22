"""Trainer checks for update ordering, precision copies, Aux, and replay."""

import copy
import math
import weakref
from dataclasses import asdict

import pytest
import torch

import block_crosscoder_experiment.gram as gram_module
import block_crosscoder_experiment.model as model_module
import block_crosscoder_experiment.trainer as trainer_module
from block_crosscoder_experiment.gram import gram_residual
from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.runtime_limits import (
    DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
    DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
    DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
    MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
)
from block_crosscoder_experiment.trainer import (
    DeadTracker,
    TrainConfig,
    Trainer,
    aux_loss,
    build_optimizer,
)

S, G, B_DIM, D_MODEL = 4, 16, 4, 32
CFG = BSCConfig(n_blocks=G, block_dim=B_DIM, n_sites=S, d_model=D_MODEL, k=3, seed=0)


def planted_batches(device, n_batches=100, batch=256, rank=8, seed=3):
    """Fixed list of batches from a planted low-rank source, so runs are
    exactly repeatable across trainers."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(n_batches * batch, rank, generator=gen)
    P = torch.randn(rank, S * D_MODEL, generator=gen) / rank**0.5
    x = (u @ P).view(-1, S, D_MODEL) + 0.01 * torch.randn(
        n_batches * batch, S, D_MODEL, generator=gen
    )
    return list(x.to(device).split(batch))


def train_cfg(**overrides):
    total_steps = int(overrides.get("total_steps", 100))
    base = dict(
        total_steps=100,
        lr=3e-3,
        warmup_steps=min(5, max(0, total_steps - 1)),
        forward_dtype="fp32",
        optimizer="adamw",
        aux_variant="none",
        log_every=5,
    )
    return TrainConfig(**{**base, **overrides})


def _assert_nested_exact(actual, expected) -> None:
    if torch.is_tensor(actual):
        assert torch.is_tensor(expected)
        assert actual.dtype == expected.dtype
        # Optimizer scalar steps are CPU before serialization and restored to
        # the parameter device by PyTorch. Device placement is lifecycle
        # state; compare the serialized numerical state exactly.
        if actual.device != expected.device:
            actual = actual.cpu()
            expected = expected.cpu()
        if actual.is_floating_point() or actual.is_complex():
            actual_nan = torch.isnan(actual)
            expected_nan = torch.isnan(expected)
            assert torch.equal(actual_nan, expected_nan)
            actual = torch.where(actual_nan, torch.zeros_like(actual), actual)
            expected = torch.where(expected_nan, torch.zeros_like(expected), expected)
        assert torch.equal(actual, expected)
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_exact(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_exact(actual_item, expected_item)
        return
    assert actual == expected


def _nested_relative_l2(actual, expected) -> float:
    """Relative L2 over floating leaves while checking all other state exactly."""

    numerator = 0.0
    denominator = 0.0

    def visit(actual_value, expected_value) -> None:
        nonlocal numerator, denominator
        if torch.is_tensor(actual_value):
            assert torch.is_tensor(expected_value)
            assert actual_value.dtype == expected_value.dtype
            if actual_value.device != expected_value.device:
                actual_value = actual_value.cpu()
                expected_value = expected_value.cpu()
            if not (actual_value.is_floating_point() or actual_value.is_complex()):
                assert torch.equal(actual_value, expected_value)
                return
            actual_nan = torch.isnan(actual_value)
            expected_nan = torch.isnan(expected_value)
            assert torch.equal(actual_nan, expected_nan)
            actual_finite = torch.where(
                actual_nan,
                torch.zeros_like(actual_value),
                actual_value,
            ).double()
            expected_finite = torch.where(
                expected_nan,
                torch.zeros_like(expected_value),
                expected_value,
            ).double()
            assert torch.isfinite(actual_finite).all()
            assert torch.isfinite(expected_finite).all()
            numerator += float((actual_finite - expected_finite).square().sum())
            denominator += float(expected_finite.square().sum())
            return
        if isinstance(actual_value, dict):
            assert isinstance(expected_value, dict)
            assert actual_value.keys() == expected_value.keys()
            for key in actual_value:
                visit(actual_value[key], expected_value[key])
            return
        if isinstance(actual_value, (list, tuple)):
            assert type(actual_value) is type(expected_value)
            assert len(actual_value) == len(expected_value)
            for actual_item, expected_item in zip(
                actual_value,
                expected_value,
                strict=True,
            ):
                visit(actual_item, expected_item)
            return
        assert actual_value == expected_value

    visit(actual, expected)
    return math.sqrt(numerator / max(denominator, 1e-30))


def test_fp32_step_ordering_loss_falls(device):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=60))
    batches = planted_batches(device)
    history = trainer.fit(batches)
    assert trainer.step_idx == 60
    assert history[-1]["rec"] < 0.5 * history[0]["rec"]
    assert all(rec["floor_hits"] == 0 for rec in history)
    assert history[-1]["decoder_constraint_residual_master"] < 1e-4


def test_prepacked_full_encoder_matches_materialized_pack_trajectory(monkeypatch):
    config = BSCConfig(
        n_blocks=12,
        block_dim=3,
        n_sites=4,
        d_model=9,
        k=3,
        seed=4701,
        decoder_constraint="free",
        encoder_mode="untied",
        encoder_fusion="sum",
    )
    fast = Trainer(
        BlockCrosscoder(config),
        train_cfg(total_steps=6, warmup_steps=0, log_every=6),
    )
    oracle = Trainer(
        BlockCrosscoder(config),
        train_cfg(total_steps=6, warmup_steps=0, log_every=6),
    )

    def materialized_encoder():
        # Reproduce the superseded logical parameter surface. The following
        # `_encode_with_tensor` call must pack this contiguous [S,G,b,d]
        # tensor before GEMM, while the release path consumes E directly.
        return oracle.master._encoder_full_tensor().contiguous()

    monkeypatch.setattr(oracle.master, "encoder_tensor", materialized_encoder)
    generator = torch.Generator().manual_seed(4702)
    batches = [torch.randn(32, 4, 9, generator=generator) for _ in range(6)]
    for batch in batches:
        fast.step(batch, materialize_record=False)
        oracle.step(batch, materialize_record=False)

    _assert_nested_exact(fast.master.state_dict(), oracle.master.state_dict())
    _assert_nested_exact(fast.opt.state_dict(), oracle.opt.state_dict())
    _assert_nested_exact(fast.sched.state_dict(), oracle.sched.state_dict())
    _assert_nested_exact(fast.history, oracle.history)


def test_step_releases_dead_forward_branches_before_backward(monkeypatch):
    trainer = Trainer(BlockCrosscoder(CFG), train_cfg(total_steps=2))
    original_forward = trainer.fwd.forward_with_materialized
    original_loss = trainer_module.bsc_loss
    references: dict[str, weakref.ReferenceType[torch.Tensor]] = {}
    checked = False

    def observed_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        output = result[0]
        references["z"] = weakref.ref(output.z)
        references["scores"] = weakref.ref(output.scores)
        references["z_selected"] = weakref.ref(output.z_selected)
        return result

    def observed_loss(*args, **kwargs):
        nonlocal checked
        parts = original_loss(*args, **kwargs)

        def assert_lifetimes(gradient):
            nonlocal checked
            assert references["z"]() is None
            assert references["scores"]() is None
            # Decode backward still needs the selected code at this point.
            assert references["z_selected"]() is not None
            checked = True
            return gradient

        parts["total"].register_hook(assert_lifetimes)
        return parts

    monkeypatch.setattr(trainer.fwd, "forward_with_materialized", observed_forward)
    monkeypatch.setattr(trainer_module, "bsc_loss", observed_loss)
    trainer.step(torch.randn(16, S, D_MODEL), materialize_record=False)
    assert checked


def test_trainer_detaches_unconsumed_score_graph_without_changing_trajectory(
    monkeypatch,
):
    base = BlockCrosscoder(CFG)
    training = train_cfg(total_steps=3, log_every=1)
    optimized = Trainer(copy.deepcopy(base), training)
    reference = Trainer(copy.deepcopy(base), training)
    original_forward = reference.fwd.forward_with_materialized

    def force_score_grad(*args, **kwargs):
        kwargs["_score_grad"] = True
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(
        reference.fwd,
        "forward_with_materialized",
        force_score_grad,
    )
    batches = planted_batches("cpu", n_batches=3, batch=32)
    optimized_records = [optimized.step(batch) for batch in batches]
    reference_records = [reference.step(batch) for batch in batches]
    _assert_nested_exact(optimized_records, reference_records)
    _assert_nested_exact(optimized.master.state_dict(), reference.master.state_dict())
    _assert_nested_exact(optimized.opt.state_dict(), reference.opt.state_dict())


def test_trainer_retains_score_graph_for_positive_crosscoder_l1(monkeypatch):
    cfg = BSCConfig(
        n_blocks=8,
        block_dim=1,
        n_sites=2,
        d_model=6,
        k=2,
        selection="batch_topk",
        code_activation="relu",
        selection_score="decoder_weighted",
        decoder_constraint="free",
        regularizer="crosscoder_l1",
        lambda_regularizer=1e-3,
    )
    trainer = Trainer(BlockCrosscoder(cfg), train_cfg(total_steps=2))
    original_scores = trainer.fwd.scores
    observed_requires_grad: list[bool] = []

    def observed_scores(*args, **kwargs):
        result = original_scores(*args, **kwargs)
        observed_requires_grad.append(result.requires_grad)
        return result

    monkeypatch.setattr(trainer.fwd, "scores", observed_scores)
    trainer.step(torch.randn(24, 2, 6), materialize_record=False)
    assert observed_requires_grad == [True]


def test_optimizer_numerics_are_explicitly_frozen():
    model = BlockCrosscoder(CFG)
    cfg = train_cfg(total_steps=1, eps=3e-8, foreach=False, fused=False)
    optimizer, kind = build_optimizer(model, cfg)
    assert kind == "adamw"
    assert all(group["eps"] == pytest.approx(3e-8) for group in optimizer.param_groups)
    assert all(group["foreach"] is False for group in optimizer.param_groups)
    assert all(group["fused"] is False for group in optimizer.param_groups)
    with pytest.raises(ValueError, match="foreach=False"):
        train_cfg(total_steps=1, foreach=True)
    with pytest.raises(ValueError, match="exact boolean"):
        train_cfg(total_steps=1, fused=1)


@pytest.mark.parametrize("optimizer_name", ("adam", "adamw"))
def test_fused_optimizer_refuses_cpu_master_parameters(optimizer_name):
    with pytest.raises(ValueError, match="fp32 CUDA master"):
        build_optimizer(
            BlockCrosscoder(CFG),
            train_cfg(total_steps=1, optimizer=optimizer_name, fused=True),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("optimizer_name", ("adam", "adamw"))
def test_fused_optimizer_binds_every_cuda_parameter_group(optimizer_name):
    optimizer, kind = build_optimizer(
        BlockCrosscoder(CFG).to("cuda"),
        train_cfg(total_steps=1, optimizer=optimizer_name, fused=True),
    )
    assert kind == optimizer_name
    assert all(group["fused"] is True for group in optimizer.param_groups)
    assert all(group["foreach"] is False for group in optimizer.param_groups)


def test_gradient_clipping_updates_master_gradients_and_reports_both_norms(device):
    clip = 1e-4
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=2, gradient_clip_norm=clip),
    )
    record = trainer.step(planted_batches(device, n_batches=1)[0])
    gradients = [
        parameter.grad
        for parameter in trainer.master.parameters()
        if parameter.grad is not None
    ]
    actual = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(gradient.float()) for gradient in gradients]
        )
    )
    assert record["grad_norm_unclipped"] > clip
    assert float(actual) <= clip * 1.001
    assert record["grad_norm"] == pytest.approx(float(actual), rel=2e-5)


def test_projection_cadence_counts_completed_updates(device, monkeypatch):
    """Cadence two projects after updates 2 and 4, not 1, 3, and 5."""
    calls: list[int] = []
    project = trainer_module._project_decoder_

    def counted(model, **kwargs):
        calls.append(1)
        return project(model, **kwargs)

    monkeypatch.setattr(trainer_module, "_project_decoder_", counted)
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=5, retract_every=2, log_every=1),
    )
    trainer.fit(planted_batches(device, n_batches=5))
    assert len(calls) == 2


def test_bf16_forward_copy_stays_in_sync(device):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=30, forward_dtype="bf16"))
    trainer.fit(planted_batches(device))
    # Forward copy is exactly the bf16 cast of the retracted master.
    for m, f in zip(trainer.master.parameters(), trainer.fwd.parameters()):
        assert f.dtype == torch.bfloat16
        assert torch.equal(f, m.to(torch.bfloat16))
    # Master satisfies the constraint tightly; the post-cast residual is
    # bounded by bf16 resolution, not by drift.
    assert float(gram_residual(trainer.master.D).max()) < 1e-4
    assert trainer.history[-1]["decoder_constraint_residual_postcast"] < 5e-2
    # Loss still falls through the cast/copy plumbing.
    assert trainer.history[-1]["rec"] < 0.7 * trainer.history[0]["rec"]


def test_bf16_forward_gradients_are_released_without_skipping_zero_updates(device):
    cfg = BSCConfig(**{**CFG.__dict__, "encoder_bias": True})
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=3, forward_dtype="bf16", optimizer="adamw"),
    )
    batches = planted_batches(device, n_batches=2, seed=210)
    trainer.step(batches[0], materialize_record=False)
    assert all(parameter.grad is None for parameter in trainer.fwd.parameters())
    assert trainer.master.a is not None and trainer.fwd.a is not None
    first_step = trainer.opt.state[trainer.master.a]["step"].detach().clone()

    # Simulate a parameter used by an earlier graph becoming absent from the
    # next one. The historical retained buffer supplied an explicit zero to
    # Adam, so releasing bf16 gradients must preserve that update semantics.
    trainer.fwd.a.requires_grad_(False)
    trainer.step(batches[1], materialize_record=False)
    assert trainer.master.a.grad is not None
    assert torch.count_nonzero(trainer.master.a.grad) == 0
    assert trainer.opt.state[trainer.master.a]["step"] == first_step + 1
    assert all(parameter.grad is None for parameter in trainer.fwd.parameters())


def test_bf16_threshold_cache_survives_steps_and_revalidates_resume(
    device,
    tmp_path,
):
    cfg = BSCConfig(**{**CFG.__dict__, "selection": "threshold"})
    model = BlockCrosscoder(cfg).to(device)
    model.theta.fill_(0.0)
    trainer = Trainer(model, train_cfg(total_steps=3, forward_dtype="bf16"))
    batches = planted_batches(device, n_batches=3, seed=211)

    initial_version = trainer.fwd.theta._version
    trainer.step(batches[0])
    validated_key = trainer.fwd._validated_theta_key
    assert validated_key is not None
    trainer.step(batches[1])
    assert trainer.fwd._validated_theta_key == validated_key
    assert trainer.fwd.theta._version == initial_version
    assert torch.equal(trainer.fwd.theta, trainer.master.theta.to(torch.bfloat16))

    checkpoint = tmp_path / "threshold-cache.pt"
    trainer.save_checkpoint(checkpoint)
    resumed = Trainer.load_checkpoint(checkpoint, device=device)
    assert resumed.fwd._validated_theta_key is None
    resumed.step(batches[2])
    assert resumed.fwd._validated_theta_key is not None


def test_dead_tracker_criteria(device):
    B = 64
    frequency_tracker = DeadTracker(
        n_blocks=4,
        capacity=8,
        device=device,
        max_tokens=6 * B,
        policy="sasa",
        selector="token_topk",
        active_blocks=1,
    )
    horizon_tracker = DeadTracker(
        n_blocks=4,
        capacity=8,
        device=device,
        max_tokens=0,
        policy="long_horizon",
    )
    mask = torch.zeros(B, 4, dtype=torch.bool, device=device)
    mask[:, 0] = True  # block 0 always active
    first = mask.clone()
    first[0, 0] = False
    first[0, 2] = True  # block 2 replaces block 0 once in the first batch

    frequency_tracker.update(first)
    horizon_tracker.update(first)
    for _ in range(3):
        frequency_tracker.update(mask)
        horizon_tracker.update(mask)
    # Warmup gating: the 6-batch-equivalent token window is not yet full.
    assert not frequency_tracker.dead(
        "sasa", threshold=1e-4, window_tokens=6 * B, horizon_tokens=8 * B
    ).any()
    frequency_tracker.update(mask)
    frequency_tracker.update(mask)
    horizon_tracker.update(mask)
    horizon_tracker.update(mask)
    dead = frequency_tracker.dead(
        "sasa", threshold=1e-4, window_tokens=6 * B, horizon_tokens=8 * B
    )
    # Block 1 and 3: never active. Block 2: freq 1/384 > 1e-4. Block 0: alive.
    assert dead.tolist() == [False, True, False, True]
    freq = frequency_tracker.frequency(6 * B)
    assert abs(float(freq[2]) - 1 / (6 * B)) < 1e-6
    # Long-horizon at horizon=8: not full yet, then dead once block 2's
    # single activation scrolls out of the window.
    assert not horizon_tracker.dead(
        "long_horizon",
        threshold=1e-4,
        window_tokens=6 * B,
        horizon_tokens=8 * B,
    ).any()
    for _ in range(4):
        horizon_tracker.update(mask)
    dead_lh = horizon_tracker.dead(
        "long_horizon",
        threshold=1e-4,
        window_tokens=6 * B,
        horizon_tokens=8 * B,
    )
    assert dead_lh.tolist() == [False, True, True, True]


@pytest.mark.parametrize(
    "policy",
    (
        "sasa",
        "long_horizon",
        "decoder_weighted_token_horizon",
        "sasa_release",
    ),
)
def test_dead_tracker_policy_updates_only_exact_required_state(device, policy):
    max_tokens = 19 if policy == "sasa" else 0
    tracker = DeadTracker(
        n_blocks=7,
        block_dim=3,
        capacity=8,
        device=device,
        max_tokens=max_tokens,
        policy=policy,
        selector=("token_topk" if policy == "sasa" else None),
        active_blocks=(2 if policy == "sasa" else None),
    )
    generator = torch.Generator().manual_seed(919)
    for batch_size in (5, 3, 11, 2):
        scores = torch.rand(batch_size, 7, generator=generator)
        if policy == "sasa":
            mask = torch.zeros(batch_size, 7, dtype=torch.bool)
            mask.scatter_(1, scores.topk(2, dim=1).indices, True)
        else:
            mask = scores > 0.7
        activity = torch.rand(batch_size, 7, 3, generator=generator) > 0.75
        mask = mask.to(device)
        activity = activity.to(device)
        tracker.update(mask, activity if policy == "sasa_release" else None)

    policy_keys = {
        "sasa": {
            "selector",
            "active_blocks",
            "representation",
            "chunks",
        },
        "long_horizon": {"tokens_seen", "last_fire"},
        "decoder_weighted_token_horizon": {"tokens_since_fired"},
        "sasa_release": {
            "coordinate_passes_since_fired",
            "forward_passes",
        },
    }
    state = tracker.state_dict()
    assert set(state) == {
        "capacity",
        "max_tokens",
        "block_dim",
        "policy",
        *policy_keys[policy],
    }
    restored = DeadTracker(
        n_blocks=7,
        block_dim=3,
        capacity=8,
        device=device,
        max_tokens=max_tokens,
        policy=policy,
        selector=("token_topk" if policy == "sasa" else None),
        active_blocks=(2 if policy == "sasa" else None),
    )
    restored.load_state_dict(state)
    with pytest.raises(ValueError, match="mapping"):
        restored.load_state_dict([])
    restored_state = restored.state_dict()
    for key, value in state.items():
        other = restored_state[key]
        if torch.is_tensor(value):
            assert torch.equal(value, other)
        elif isinstance(value, list):
            assert len(value) == len(other)
            for chunk, restored_chunk in zip(value, other, strict=True):
                assert chunk.keys() == restored_chunk.keys()
                assert torch.equal(
                    chunk["indices"],
                    restored_chunk["indices"],
                )
                assert chunk["n_tokens"] == restored_chunk["n_tokens"]
                assert chunk["start_token"] == restored_chunk["start_token"]
        else:
            assert value == other

    forged = dict(state)
    if policy == "sasa":
        forged["chunks"] = [
            {
                "indices": torch.zeros(2, 1, dtype=torch.int32, device=device),
                "n_tokens": 2,
                "start_token": 0,
            }
        ]
    elif policy == "long_horizon":
        forged["last_fire"] = torch.zeros(1, dtype=torch.int64, device=device)
    elif policy == "decoder_weighted_token_horizon":
        forged["tokens_since_fired"] = torch.zeros(1, dtype=torch.int64, device=device)
    else:
        forged["coordinate_passes_since_fired"] = torch.zeros(
            1, dtype=torch.int64, device=device
        )
    with pytest.raises(ValueError, match="tracker"):
        restored.load_state_dict(forged)
    for name, value in (
        ("capacity", float(state["capacity"])),
        ("capacity", str(state["capacity"])),
        ("block_dim", float(state["block_dim"])),
        ("max_tokens", float(state["max_tokens"])),
    ):
        forged_metadata = dict(state)
        forged_metadata[name] = value
        with pytest.raises(ValueError, match="configuration"):
            restored.load_state_dict(forged_metadata)


@pytest.mark.parametrize(
    ("selector", "active_blocks"),
    (
        ("token_topk", 2.0),
        ("batch_topk", 1.5),
        ("threshold", 2.0),
        ("dense", 2.0),
    ),
)
def test_sasa_sparse_and_bitpacked_history_match_dense_reference(
    device,
    selector,
    active_blocks,
):
    n_blocks = 7
    window = 17
    tracker = DeadTracker(
        n_blocks=n_blocks,
        capacity=8,
        device=device,
        max_tokens=window,
        policy="sasa",
        selector=selector,
        active_blocks=active_blocks,
    )
    dense = torch.empty(0, n_blocks, dtype=torch.bool, device=device)
    generator = torch.Generator().manual_seed(1931)
    for batch_size in (1, 5, 23, 3, 11):
        scores = torch.rand(batch_size, n_blocks * 2, generator=generator)[:, ::2]
        assert not scores.is_contiguous()
        if selector == "token_topk":
            selected = torch.zeros(batch_size, n_blocks, dtype=torch.bool)
            selected.scatter_(1, scores.topk(2, dim=1).indices, True)
        elif selector == "batch_topk":
            selected = torch.zeros(batch_size, n_blocks, dtype=torch.bool)
            keep = round(batch_size * active_blocks)
            selected.view(-1).scatter_(
                0,
                scores.reshape(-1).topk(keep).indices,
                True,
            )
        else:
            selected = scores > 0.62
        storage = torch.zeros(batch_size, n_blocks * 2, dtype=torch.bool)
        storage[:, ::2] = selected
        mask = storage[:, ::2]
        assert not mask.is_contiguous()
        mask = mask.to(device)
        tracker.update(mask)
        dense = torch.cat((dense, mask), dim=0)[-window:]
        for suffix in {1, min(5, len(dense)), len(dense)}:
            expected = dense[-suffix:].sum(dim=0, dtype=torch.int64).float() / suffix
            assert torch.equal(tracker.frequency(suffix), expected)

        restored = DeadTracker(
            n_blocks=n_blocks,
            capacity=8,
            device=device,
            max_tokens=window,
            policy="sasa",
            selector=selector,
            active_blocks=active_blocks,
        )
        restored.load_state_dict(tracker.state_dict())
        assert torch.equal(
            restored.frequency(len(dense)), tracker.frequency(len(dense))
        )
        tracker = restored

    if tracker.representation != "fixed_batch_indices":
        state = tracker.state_dict()
        forged = dict(state)
        forged["chunks"] = [dict(chunk) for chunk in state["chunks"]]
        forged["chunks"][0]["start_token"] = 1
        with pytest.raises(ValueError, match="stale prefix"):
            tracker.load_state_dict(forged)

    state = tracker.state_dict()
    forged = dict(state)
    forged["chunks"] = [dict(chunk) for chunk in state["chunks"]]
    forged["chunks"][0]["indices"] = forged["chunks"][0]["indices"].to_sparse()
    with pytest.raises(ValueError, match="dense strided"):
        tracker.load_state_dict(forged)


def test_dead_tracker_windows_are_token_denominated(device):
    tracker = DeadTracker(
        n_blocks=2,
        capacity=2,
        device=device,
        max_tokens=10,
        policy="sasa",
        selector="token_topk",
        active_blocks=1,
    )
    first = torch.tensor([[True, False]] * 6, device=device)
    second = torch.tensor([[False, True]] * 2, device=device)
    third = torch.tensor([[False, True]] * 2, device=device)
    tracker.update(first)
    tracker.update(second)
    assert not tracker.dead(
        "sasa", threshold=0.5, window_tokens=10, horizon_tokens=10
    ).any()
    # The third observation fills the exact ten-token window.
    tracker.update(third)
    assert tracker.history_tokens == 10
    assert tracker.frequency(4).tolist() == [0.0, 1.0]
    assert tracker.dead(
        "sasa", threshold=0.5, window_tokens=10, horizon_tokens=10
    ).tolist() == [False, True]
    restored = DeadTracker(
        n_blocks=2,
        capacity=2,
        device=device,
        max_tokens=10,
        policy="sasa",
        selector="token_topk",
        active_blocks=1,
    )
    restored.load_state_dict(tracker.state_dict())
    assert restored.history_tokens == tracker.history_tokens
    assert torch.equal(restored.frequency(10), tracker.frequency(10))
    next_mask = torch.tensor([[True, False]] * 3, device=device)
    tracker.update(next_mask)
    restored.update(next_mask)
    assert torch.equal(restored.frequency(10), tracker.frequency(10))


def test_dead_tracker_slices_the_oldest_batch_at_the_exact_window(device):
    tracker = DeadTracker(
        n_blocks=2,
        capacity=2,
        device=device,
        max_tokens=1_000,
        policy="sasa",
        selector="token_topk",
        active_blocks=1,
    )
    mask = torch.zeros(4_096, 2, dtype=torch.bool, device=device)
    mask[:3_096, 0] = True
    mask[3_096:, 1] = True
    tracker.update(mask)
    assert tracker.history_tokens == 1_000
    assert tracker.frequency(1_000).tolist() == [0.0, 1.0]


def test_sasa_release_deadness_is_scalar_and_pass_denominated(device):
    tracker = DeadTracker(
        n_blocks=2,
        block_dim=3,
        capacity=2,
        device=device,
        max_tokens=0,
        policy="sasa_release",
    )
    mask = torch.tensor([[True, False]], device=device)
    activity = torch.tensor(
        [[[True, False, True], [False, False, False]]], device=device
    )
    tracker.update(mask, activity)
    assert not tracker.dead_coordinates(2).any()
    tracker.update(mask, activity)
    tracker.update(mask, activity)
    # SAELens uses age > window, not >=. Coordinates 0/2 keep firing;
    # coordinate 1 and every coordinate of block 1 have age three.
    assert tracker.dead_coordinates(2).tolist() == [
        [False, True, False],
        [True, True, True],
    ]
    state = tracker.state_dict()
    restored = DeadTracker(
        n_blocks=2,
        block_dim=3,
        capacity=2,
        device=device,
        max_tokens=0,
        policy="sasa_release",
    )
    restored.load_state_dict(state)
    assert torch.equal(
        restored.coordinate_passes_since_fired,
        tracker.coordinate_passes_since_fired,
    )


def test_token_horizon_deadness_updates_at_current_batch_boundary(device):
    tracker = DeadTracker(
        n_blocks=3,
        capacity=2,
        device=device,
        max_tokens=0,
        policy="decoder_weighted_token_horizon",
    )
    assert tracker.tokens_since_fired is not None
    tracker.tokens_since_fired.copy_(
        torch.tensor([7, 7, 1], dtype=torch.int64, device=device)
    )
    current = torch.tensor([[True, False, False], [False, False, False]], device=device)
    # The pinned release increments by B=2, then resets any feature selected
    # in the current batch. Feature 0 therefore remains alive; feature 1
    # reaches the >=9 threshold; feature 2 remains below it.
    assert tracker.token_horizon_dead_after_current(current, 9).tolist() == [
        False,
        True,
        False,
    ]
    tracker.update(current)
    assert tracker.tokens_since_fired.tolist() == [0, 9, 3]
    restored = DeadTracker(
        n_blocks=3,
        capacity=2,
        device=device,
        max_tokens=0,
        policy="decoder_weighted_token_horizon",
    )
    restored.load_state_dict(tracker.state_dict())
    assert restored.tokens_since_fired is not None
    assert torch.equal(restored.tokens_since_fired, tracker.tokens_since_fired)


def test_decoder_weighted_token_horizon_uses_scaled_rank_and_unscaled_values(device):
    cfg = BSCConfig(
        n_blocks=4,
        block_dim=1,
        n_sites=2,
        d_model=3,
        k=1,
        seed=71,
        code_activation="relu",
        selection="batch_topk",
        selection_score="decoder_weighted",
        decoder_norm_geometry="sum_l2",
        decoder_bias=True,
        decoder_constraint="unit_latent",
    )
    model = BlockCrosscoder(cfg).to(device)
    with torch.no_grad():
        # Make decoder-weighted ranking observably different from raw-code
        # ranking without changing the unscaled activations to be decoded.
        model.D[:, 0].mul_(0.25)
        model.D[:, 1].mul_(3.0)
        model.c.fill_(0.17)
    x = torch.randn(5, 2, 3, generator=torch.Generator().manual_seed(73)).to(device)
    out = model(x)
    dead = torch.tensor([True, True, False, False], device=device)
    actual = aux_loss(
        model,
        x,
        out,
        "decoder_weighted_token_horizon",
        dead,
        1,
        reconstruction_loss="squared_l2_over_residual_variance",
    )
    assert actual is not None

    ranked = out.scores.masked_fill(~dead.view(1, -1), float("-inf"))
    chosen = ranked.topk(1, dim=1, sorted=False).indices
    unscaled = out.z.squeeze(-1)
    aux = torch.zeros_like(unscaled)
    aux.scatter_(1, chosen, unscaled.gather(1, chosen))
    residual = (x - out.xhat).detach()
    decoded = model.decode(aux.unsqueeze(-1), add_bias=False)
    numerator = (residual.float() - decoded.float()).pow(2).sum(dim=(1, 2)).mean()
    flattened = residual.float().reshape(len(residual), -1)
    denominator = (flattened - flattened.mean(dim=0)).pow(2).sum(dim=1).mean()
    expected = (numerator / denominator).nan_to_num(0.0)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "reconstruction_loss",
    (
        "mean_l2",
        "mean_squared",
        "squared_l2",
        "squared_l2_over_residual_variance",
    ),
)
@pytest.mark.parametrize("masked_sites", (False, True))
def test_auxiliary_reconstruction_fast_paths_match_definition(
    device,
    monkeypatch,
    reconstruction_loss,
    masked_sites,
):
    cfg = BSCConfig(
        n_blocks=4,
        block_dim=1,
        n_sites=2,
        d_model=3,
        site_dims=(2, 3),
        k=1,
        seed=79,
        decoder_constraint="free",
    )
    model = BlockCrosscoder(cfg).to(device)
    x = torch.randn(5, 2, 3, generator=torch.Generator().manual_seed(83)).to(device)
    coord = model.coordinate_mask[:, 0, 0]
    x = x * coord
    out = model(x)
    observed = None
    if masked_sites:
        observed = torch.tensor(
            [[True, False], [True, True], [False, True], [True, True], [True, False]],
            device=device,
        )
    monkeypatch.setattr(
        model,
        "decode",
        lambda z, *, add_bias=True, _decoder=None: torch.zeros_like(x),
    )

    actual = aux_loss(
        model,
        x,
        out,
        "sasa_release",
        torch.ones(cfg.n_blocks, cfg.block_dim, dtype=torch.bool, device=device),
        s_aux=1,
        observation_mask=observed,
        reconstruction_loss=reconstruction_loss,
    )
    assert actual is not None

    residual = (x - out.xhat).detach().float() * coord
    site_mask = (
        torch.ones(len(x), cfg.n_sites, 1, device=device)
        if observed is None
        else observed.float().unsqueeze(-1)
    )
    masked = residual * site_mask
    if reconstruction_loss == "mean_l2":
        expected = masked.norm(dim=-1).sum() / site_mask.squeeze(-1).sum()
    elif reconstruction_loss == "mean_squared":
        expected = masked.square().sum() / (coord * site_mask).sum()
    elif reconstruction_loss == "squared_l2":
        expected = masked.square().sum() / len(x)
    else:
        target = residual * site_mask
        mean = target.sum(dim=0) / site_mask.sum(dim=0).clamp_min(1.0)
        centered = (target - mean.unsqueeze(0)) * site_mask
        expected = (masked.square().sum() / len(x)) / (
            centered.square().sum() / len(x)
        ).clamp_min(1e-30)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "variant",
    (
        "sasa",
        "long_horizon",
        "sasa_release",
        "decoder_weighted_token_horizon",
    ),
)
def test_auxiliary_warmup_readiness_has_exact_boundary(device, variant):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(
            total_steps=2,
            aux_variant=variant,
            dead_window_tokens=10,
            dead_horizon_tokens=20,
            dead_window_passes=3,
        ),
    )
    if variant == "sasa":
        trainer.tracker._history_tokens = 9
        assert not trainer._auxiliary_can_have_dead_features(4)
        trainer.tracker._history_tokens = 10
    elif variant == "long_horizon":
        trainer.tracker.tokens_seen = 19
        assert not trainer._auxiliary_can_have_dead_features(4)
        trainer.tracker.tokens_seen = 20
    elif variant == "sasa_release":
        trainer.tracker.forward_passes = 3
        assert not trainer._auxiliary_can_have_dead_features(4)
        trainer.tracker.forward_passes = 4
    else:
        trainer.accepted_tokens = 15
        assert not trainer._auxiliary_can_have_dead_features(4)
        trainer.accepted_tokens = 16
    assert trainer._auxiliary_can_have_dead_features(4)


def test_known_empty_deadness_skips_auxiliary_work(device, monkeypatch):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(
            total_steps=1,
            aux_variant="long_horizon",
            dead_horizon_tokens=10_000,
        ),
    )

    def forbidden_aux(*args, **kwargs):
        raise AssertionError("known-empty warmup must not enter AuxK")

    monkeypatch.setattr(trainer_module, "aux_loss", forbidden_aux)
    trainer.step(planted_batches(device, n_batches=1, batch=16)[0])


@pytest.mark.parametrize("accepted_tokens", (None, True, -1, 1.5))
def test_checkpoint_requires_exact_accepted_token_counter(
    device, tmp_path, accepted_tokens
):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=1, aux_variant="long_horizon"),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if accepted_tokens is None:
        del payload["accepted_tokens"]
    else:
        payload["accepted_tokens"] = accepted_tokens
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="accepted-token counter"):
        Trainer.load_checkpoint(checkpoint, device=device)


@pytest.mark.parametrize(
    ("variant", "tracker_key", "tracker_value", "message"),
    (
        ("long_horizon", "tokens_seen", 1, "disagrees"),
        (
            "decoder_weighted_token_horizon",
            "tokens_since_fired",
            torch.ones(G, dtype=torch.int64),
            "exceeds",
        ),
    ),
)
def test_checkpoint_rejects_dead_tracker_token_clock_forgery(
    device,
    tmp_path,
    variant,
    tracker_key,
    tracker_value,
    message,
):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=1, aux_variant=variant),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["tracker"][tracker_key] = tracker_value
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        Trainer.load_checkpoint(checkpoint, device=device)


def test_checkpoint_rejects_sasa_history_clock_forgery(device, tmp_path):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=2, aux_variant="sasa", dead_window_tokens=100),
    )
    trainer.step(planted_batches(device, n_batches=1, batch=16)[0])
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["accepted_tokens"] -= 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="accepted-token counter"):
        Trainer.load_checkpoint(checkpoint, device=device)


def test_checkpoint_rejects_sasa_release_step_clock_forgery(device, tmp_path):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=2, aux_variant="sasa_release"),
    )
    trainer.step(planted_batches(device, n_batches=1, batch=16)[0])
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["step_idx"] += 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="exact step counter"):
        Trainer.load_checkpoint(checkpoint, device=device)


@pytest.mark.parametrize("step_idx", (None, True, -1, 1.5))
def test_checkpoint_requires_exact_step_counter(device, tmp_path, step_idx):
    trainer = Trainer(BlockCrosscoder(CFG).to(device), train_cfg(total_steps=1))
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if step_idx is None:
        del payload["step_idx"]
    else:
        payload["step_idx"] = step_idx
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="exact step counter"):
        Trainer.load_checkpoint(checkpoint, device=device)


def test_auxk_revives_dead_encoders(device):
    """The pilot's synthetic dead-encoder revival test: blocks with zeroed
    encoders are never selected, get flagged dead, and only the aux loss
    can pull them back (decoder shrinkage is impossible by construction;
    starvation is encoder-side)."""
    revived_norms = {}
    for variant in ("sasa", "none"):
        model = BlockCrosscoder(CFG).to(device)
        with torch.no_grad():
            model._encoder_full_tensor()[:, :4] = 0.0  # kill blocks 0-3
        trainer = Trainer(
            model,
            train_cfg(
                total_steps=80,
                aux_variant=variant,
                s_aux=8,
                dead_window_tokens=4 * 256,
                dead_horizon_tokens=8 * 256,
            ),
        )
        trainer.fit(planted_batches(device))
        revived_norms[variant] = float(
            model._encoder_full_tensor().detach()[:, :4].float().norm()
        )
    # Without aux there is no gradient path to a zeroed encoder.
    assert revived_norms["none"] == 0.0
    assert revived_norms["sasa"] > 1e-2


def test_fel_runner_up_aux(device):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=20, aux_variant="fel", s_aux=4))
    history = trainer.fit(planted_batches(device))
    logged = [r for r in history if "aux" in r]
    assert logged and all(torch.isfinite(torch.tensor(r["aux"])) for r in logged)
    assert history[-1]["rec"] < history[0]["rec"]


def test_checkpoint_resume_matches(device, tmp_path):
    batches = planted_batches(device, n_batches=40)
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=25))
    trainer.fit(batches[:15])
    ckpt = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt)

    expected_rng_draw = torch.rand(8, device=device)
    continued_records = [trainer.step(x) for x in batches[15:25]]
    resumed_trainer = Trainer.load_checkpoint(ckpt, device=device)
    actual_rng_draw = torch.rand(8, device=device)
    assert torch.equal(actual_rng_draw, expected_rng_draw)
    assert resumed_trainer.step_idx == 15
    assert (
        resumed_trainer.history
        == torch.load(ckpt, map_location="cpu", weights_only=True)["history"]
    )
    resumed_records = [resumed_trainer.step(x) for x in batches[15:25]]
    assert [row["rec"] for row in continued_records] == pytest.approx(
        [row["rec"] for row in resumed_records], rel=1e-4
    )
    continued_jumps = [
        row["share_jump"] for row in continued_records if "share_jump" in row
    ]
    resumed_jumps = [
        row["share_jump"] for row in resumed_records if "share_jump" in row
    ]
    assert continued_jumps == pytest.approx(resumed_jumps, rel=1e-6, abs=1e-8)
    assert resumed_trainer.history == trainer.history
    for a, b in zip(trainer.master.parameters(), resumed_trainer.master.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


def test_cholesky_qr_checkpoint_resume_is_exact_at_retraction_boundary(
    device,
    tmp_path,
):
    cfg = BSCConfig(
        n_blocks=24,
        block_dim=4,
        n_sites=4,
        d_model=24,
        k=4,
        seed=1671,
        decoder_constraint="qr",
        decoder_retraction_implementation=(
            DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
        ),
    )
    training = train_cfg(total_steps=4, lr=3e-4, retract_every=1, log_every=1)
    generator = torch.Generator().manual_seed(1672)
    batches = [torch.randn(64, 4, 24, generator=generator).to(device) for _ in range(4)]
    uninterrupted = Trainer(BlockCrosscoder(cfg).to(device), training)
    split = Trainer(BlockCrosscoder(cfg).to(device), training)
    expected = [uninterrupted.step(batch) for batch in batches]
    actual = [split.step(batch) for batch in batches[:2]]
    path = tmp_path / "cholesky-qr.pt"
    split.save_checkpoint(path)
    resumed = Trainer.load_checkpoint(path, device=device)
    actual.extend(resumed.step(batch) for batch in batches[2:])
    _assert_nested_exact(actual, expected)
    _assert_nested_exact(resumed.master.state_dict(), uninterrupted.master.state_dict())
    _assert_nested_exact(resumed.opt.state_dict(), uninterrupted.opt.state_dict())


def test_site_bmm_polar_checkpoint_resume_is_exact_at_retraction_boundary(
    device,
    tmp_path,
):
    cfg = BSCConfig(
        n_blocks=24,
        block_dim=4,
        n_sites=4,
        d_model=24,
        k=4,
        seed=1672,
        decoder_constraint="gram",
        decoder_retraction_implementation=(
            DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION
        ),
    )
    training = train_cfg(total_steps=4, lr=3e-4, retract_every=1, log_every=1)
    generator = torch.Generator().manual_seed(1673)
    batches = [torch.randn(64, 4, 24, generator=generator).to(device) for _ in range(4)]
    uninterrupted = Trainer(BlockCrosscoder(cfg).to(device), training)
    split = Trainer(BlockCrosscoder(cfg).to(device), training)
    expected = [uninterrupted.step(batch) for batch in batches]
    actual = [split.step(batch) for batch in batches[:2]]
    path = tmp_path / "site-bmm-polar.pt"
    split.save_checkpoint(path)
    resumed = Trainer.load_checkpoint(path, device=device)
    actual.extend(resumed.step(batch) for batch in batches[2:])
    _assert_nested_exact(actual, expected)
    _assert_nested_exact(resumed.master.state_dict(), uninterrupted.master.state_dict())
    _assert_nested_exact(resumed.opt.state_dict(), uninterrupted.opt.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_site_bmm_polar_small_shape_routes_reference_trajectory_exactly():
    common = dict(
        n_blocks=256,
        block_dim=4,
        n_sites=4,
        d_model=128,
        k=8,
        seed=1674,
        selection="token_topk",
        decoder_constraint="gram",
    )
    actual_cfg = BSCConfig(
        **common,
        decoder_retraction_implementation=(
            DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION
        ),
    )
    reference_cfg = BSCConfig(
        **common,
        decoder_retraction_implementation=(
            DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION
        ),
    )
    training = train_cfg(
        total_steps=20,
        lr=3e-4,
        warmup_steps=1,
        forward_dtype="bf16",
        fused=True,
        retract_every=1,
        log_every=1,
    )
    generator = torch.Generator().manual_seed(1675)
    batches = [
        torch.randn(512, 4, 128, generator=generator).to("cuda") for _ in range(20)
    ]
    actual = Trainer(BlockCrosscoder(actual_cfg).to("cuda"), training)
    reference = Trainer(BlockCrosscoder(reference_cfg).to("cuda"), training)
    actual_records = []
    reference_records = []
    intersections = 0
    unions = 0
    for batch in batches:
        actual_records.append(actual.step(batch))
        reference_records.append(reference.step(batch))
        with torch.no_grad():
            actual_support = actual.fwd(batch.to(torch.bfloat16)).mask
            reference_support = reference.fwd(batch.to(torch.bfloat16)).mask
        intersections += int((actual_support & reference_support).sum())
        unions += int((actual_support | reference_support).sum())

    _assert_nested_exact(actual_records, reference_records)
    assert intersections == unions
    assert all(record["floor_hits"] == 0 for record in actual_records)
    assert all(record["floor_hits"] == 0 for record in reference_records)
    _assert_nested_exact(actual.master.state_dict(), reference.master.state_dict())
    _assert_nested_exact(actual.opt.state_dict(), reference.opt.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_site_bmm_polar_fast_shape_bounds_reference_training_trajectory(
    monkeypatch,
):
    common = dict(
        n_blocks=1024,
        block_dim=4,
        n_sites=4,
        d_model=512,
        k=8,
        seed=1676,
        selection="token_topk",
        encoder_mode="tied",
        decoder_constraint="gram",
    )
    actual_cfg = BSCConfig(
        **common,
        decoder_retraction_implementation=(
            DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION
        ),
    )
    reference_cfg = BSCConfig(
        **common,
        decoder_retraction_implementation=(
            DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION
        ),
    )
    training = train_cfg(
        total_steps=20,
        lr=3e-4,
        warmup_steps=1,
        forward_dtype="bf16",
        fused=True,
        retract_every=1,
        log_every=1,
    )
    generator = torch.Generator().manual_seed(1677)
    batches = [
        torch.randn(256, 4, 512, generator=generator).to("cuda") for _ in range(20)
    ]
    actual = Trainer(BlockCrosscoder(actual_cfg).to("cuda"), training)
    reference = Trainer(BlockCrosscoder(reference_cfg).to("cuda"), training)
    fast_gram = gram_module._block_gram_no_grad
    fast_calls = 0

    def counted_fast_gram(value):
        nonlocal fast_calls
        fast_calls += 1
        return fast_gram(value)

    monkeypatch.setattr(gram_module, "_block_gram_no_grad", counted_fast_gram)
    actual_records = []
    reference_records = []
    intersections = 0
    unions = 0
    for batch in batches:
        actual_records.append(actual.step(batch))
        reference_records.append(reference.step(batch))
        with torch.no_grad():
            actual_support = actual.fwd(batch.to(torch.bfloat16)).mask
            reference_support = reference.fwd(batch.to(torch.bfloat16)).mask
        intersections += int((actual_support & reference_support).sum())
        unions += int((actual_support | reference_support).sum())

    maximum_loss_drift = max(
        abs(actual_record["total"] - reference_record["total"])
        / max(abs(reference_record["total"]), 1e-30)
        for actual_record, reference_record in zip(
            actual_records,
            reference_records,
            strict=True,
        )
    )
    assert fast_calls == training.total_steps
    assert maximum_loss_drift <= 2e-5
    assert intersections / max(unions, 1) >= 0.995
    assert all(record["floor_hits"] == 0 for record in actual_records)
    assert all(record["floor_hits"] == 0 for record in reference_records)
    assert actual.master.D is not None and reference.master.D is not None
    assert float(gram_residual(actual.master.D).max()) <= 1e-4
    assert float(gram_residual(reference.master.D).max()) <= 1e-4
    assert _nested_relative_l2(
        actual.master.state_dict(),
        reference.master.state_dict(),
    ) <= 3e-3
    assert _nested_relative_l2(
        actual.opt.state_dict()["state"],
        reference.opt.state_dict()["state"],
    ) <= 1e-5


@pytest.mark.parametrize("selection", ("token_topk", "batch_topk"))
@pytest.mark.parametrize("selection_score", ("code_norm", "decoded_energy"))
def test_cholesky_qr_bounds_canonical_householder_training_trajectory(
    device,
    selection,
    selection_score,
):
    common = {
        "n_blocks": 32,
        "block_dim": 4,
        "n_sites": 4,
        "d_model": 32,
        "k": 4,
        "seed": 1673,
        "selection": selection,
        "selection_score": selection_score,
        "decoder_constraint": "qr",
    }
    cholesky = BlockCrosscoder(
        BSCConfig(
            **common,
            decoder_retraction_implementation=(
                DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
            ),
        )
    ).to(device)
    householder = BlockCrosscoder(
        BSCConfig(
            **common,
            decoder_retraction_implementation=(
                DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION
            ),
        )
    ).to(device)
    householder.load_state_dict(cholesky.state_dict())
    training = train_cfg(total_steps=5, lr=3e-4, retract_every=1, log_every=1)
    actual = Trainer(cholesky, training)
    reference = Trainer(householder, training)
    batches = planted_batches(device, n_batches=5, batch=96, seed=1674)
    maximum_loss_drift = 0.0
    intersections = 0
    unions = 0
    for batch in batches:
        actual_record = actual.step(batch)
        reference_record = reference.step(batch)
        maximum_loss_drift = max(
            maximum_loss_drift,
            abs(actual_record["total"] - reference_record["total"])
            / max(abs(reference_record["total"]), 1e-30),
        )
        with torch.no_grad():
            actual_mask = actual.fwd(batch).mask
            reference_mask = reference.fwd(batch).mask
        intersections += int((actual_mask & reference_mask).sum())
        unions += int((actual_mask | reference_mask).sum())

    assert maximum_loss_drift <= 1e-4
    assert intersections / max(unions, 1) >= 0.999
    assert (
        _nested_relative_l2(
            actual.master.state_dict(),
            reference.master.state_dict(),
        )
        <= 1e-3
    )
    assert (
        _nested_relative_l2(
            actual.opt.state_dict()["state"],
            reference.opt.state_dict()["state"],
        )
        <= 1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
def test_compiled_quadratic_trainer_bounds_trajectory_and_support_drift(
    tmp_path, monkeypatch
):
    device = torch.device("cuda")
    batch, n_sites, d_model = 512, 4, 512
    assert batch * n_sites * d_model == model_module._CUDA_QUADRATIC_FUSION_MIN_ELEMENTS
    cfg = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=n_sites,
        d_model=d_model,
        k=2,
        seed=1701,
        decoder_constraint="free",
        reconstruction_loss="mean_squared",
    )
    training = train_cfg(
        total_steps=4,
        forward_dtype="bf16",
        warmup_steps=1,
        log_every=1,
        retract_every=100,
    )
    generator = torch.Generator(device="cpu").manual_seed(1702)
    batches = [
        torch.randn(batch, n_sites, d_model, generator=generator).to(device)
        for _ in range(training.total_steps)
    ]

    def post_step_support(trainer: Trainer, batch_tensor: torch.Tensor) -> torch.Tensor:
        dtype = next(trainer.fwd.parameters()).dtype
        with torch.no_grad():
            return trainer.fwd(batch_tensor.to(dtype=dtype)).mask.detach().cpu()

    def run_steps(
        trainer: Trainer,
        step_batches: list[torch.Tensor],
    ) -> tuple[list[dict], list[torch.Tensor]]:
        records = []
        supports = []
        for batch_tensor in step_batches:
            record = trainer.step(batch_tensor)
            assert record is not None
            records.append(record)
            supports.append(post_step_support(trainer, batch_tensor))
        return records, supports

    # The oracle changes only the large-CUDA reduction back to its eager
    # implementation; every trainer and checkpoint operation is otherwise
    # identical.
    with monkeypatch.context() as eager_patch:
        eager_patch.setattr(
            model_module,
            "_fp32_squared_error_reduction",
            model_module._eager_fp32_squared_error_reduction,
        )
        eager = Trainer(BlockCrosscoder(cfg).to(device), training)
        eager_records, eager_supports = run_steps(eager, batches)

    compiled_getter = model_module._compiled_cuda_fp32_squared_error_reduction
    compiled_getter.cache_clear()
    compiled_calls = 0

    def counted_compiled(
        prediction: torch.Tensor,
        target: torch.Tensor,
        denominator: int,
    ) -> torch.Tensor:
        nonlocal compiled_calls
        compiled_calls += 1
        return compiled_getter()(prediction, target, denominator)

    monkeypatch.setattr(
        model_module,
        "_compiled_cuda_fp32_squared_error_reduction",
        lambda: counted_compiled,
    )
    compiled = Trainer(BlockCrosscoder(cfg).to(device), training)
    compiled_records, compiled_supports = run_steps(compiled, batches)

    resumable = Trainer(BlockCrosscoder(cfg).to(device), training)
    resumed_records, resumed_supports = run_steps(resumable, batches[:2])
    checkpoint = tmp_path / "compiled-quadratic.pt"
    resumable.save_checkpoint(checkpoint)
    compiled_getter.cache_clear()
    resumed = Trainer.load_checkpoint(checkpoint, device=device)
    continued_records, continued_supports = run_steps(resumed, batches[2:])
    resumed_records.extend(continued_records)
    resumed_supports.extend(continued_supports)

    assert compiled_calls == 2 * training.total_steps
    _assert_nested_exact(resumed_records, compiled_records)
    _assert_nested_exact(resumed_supports, compiled_supports)
    _assert_nested_exact(resumed.master.state_dict(), compiled.master.state_dict())
    _assert_nested_exact(resumed.fwd.state_dict(), compiled.fwd.state_dict())
    _assert_nested_exact(resumed.opt.state_dict(), compiled.opt.state_dict())
    _assert_nested_exact(resumed.sched.state_dict(), compiled.sched.state_dict())
    _assert_nested_exact(resumed.tracker.state_dict(), compiled.tracker.state_dict())
    _assert_nested_exact(resumed.history, compiled.history)
    _assert_nested_exact(resumed._prev_shares, compiled._prev_shares)
    assert resumed.step_idx == compiled.step_idx
    assert resumed.accepted_tokens == compiled.accepted_tokens
    assert resumed.data_cursor == compiled.data_cursor

    for actual_record, expected_record in zip(
        compiled_records,
        eager_records,
        strict=True,
    ):
        for name in ("rec", "total"):
            relative_error = abs(actual_record[name] - expected_record[name]) / max(
                abs(expected_record[name]),
                1e-30,
            )
            assert relative_error <= 2e-6
    support_disagreement = sum(
        int((actual != expected).sum())
        for actual, expected in zip(
            compiled_supports,
            eager_supports,
            strict=True,
        )
    ) / max(
        sum(
            int(actual.sum() + expected.sum())
            for actual, expected in zip(
                compiled_supports,
                eager_supports,
                strict=True,
            )
        ),
        1,
    )
    assert support_disagreement <= 1e-4
    assert (
        _nested_relative_l2(
            compiled.master.state_dict(),
            eager.master.state_dict(),
        )
        <= 1e-5
    )
    assert (
        _nested_relative_l2(
            compiled.fwd.state_dict(),
            eager.fwd.state_dict(),
        )
        <= 1e-5
    )
    assert (
        _nested_relative_l2(compiled.opt.state_dict(), eager.opt.state_dict()) <= 1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
@pytest.mark.parametrize("selector", ("token_topk", "batch_topk"))
def test_compiled_selector_trainer_state_and_resume_match_eager_exactly(
    tmp_path, monkeypatch, selector
):
    device = torch.device("cuda")
    batch, groups, n_sites, d_model = 512, 2048, 2, 16
    assert batch * groups == model_module._CUDA_SELECTOR_FUSION_MIN_ELEMENTS
    cfg = BSCConfig(
        n_blocks=groups,
        block_dim=1,
        n_sites=n_sites,
        d_model=d_model,
        k=8,
        seed=1811,
        selection=selector,
        decoder_constraint="free",
        reconstruction_loss="mean_squared",
    )
    training = train_cfg(
        total_steps=4,
        forward_dtype="bf16",
        warmup_steps=1,
        log_every=1,
        retract_every=100,
    )
    generator = torch.Generator(device="cpu").manual_seed(1812)
    batches = [
        torch.randn(batch, n_sites, d_model, generator=generator).to(device)
        for _ in range(training.total_steps)
    ]
    interior_name = f"_{selector.removesuffix('_topk')}_topk_interior"
    eager_name = f"_eager_{selector.removesuffix('_topk')}_topk_interior"
    compiled_name = f"_compiled_cuda_{selector.removesuffix('_topk')}_topk_interior"

    with monkeypatch.context() as eager_patch:
        eager_patch.setattr(
            model_module,
            interior_name,
            getattr(model_module, eager_name),
        )
        eager = Trainer(BlockCrosscoder(cfg).to(device), training)
        eager_records = [eager.step(batch_tensor) for batch_tensor in batches]

    compiled_getter = getattr(model_module, compiled_name)
    compiled_getter.cache_clear()
    compiled_kernel = compiled_getter()
    compiled_calls = 0

    def counted_compiled(scores: torch.Tensor, n_keep: int) -> torch.Tensor:
        nonlocal compiled_calls
        compiled_calls += 1
        return compiled_kernel(scores, n_keep)

    monkeypatch.setattr(model_module, compiled_name, lambda: counted_compiled)
    compiled = Trainer(BlockCrosscoder(cfg).to(device), training)
    compiled_records = [compiled.step(batch_tensor) for batch_tensor in batches[:2]]
    checkpoint = tmp_path / f"compiled-{selector}.pt"
    compiled.save_checkpoint(checkpoint)
    resumed = Trainer.load_checkpoint(checkpoint, device=device)
    compiled_records.extend(resumed.step(batch_tensor) for batch_tensor in batches[2:])

    assert compiled_calls == training.total_steps
    _assert_nested_exact(compiled_records, eager_records)
    _assert_nested_exact(resumed.master.state_dict(), eager.master.state_dict())
    _assert_nested_exact(resumed.fwd.state_dict(), eager.fwd.state_dict())
    _assert_nested_exact(resumed.opt.state_dict(), eager.opt.state_dict())
    _assert_nested_exact(resumed.sched.state_dict(), eager.sched.state_dict())
    _assert_nested_exact(resumed.tracker.state_dict(), eager.tracker.state_dict())
    _assert_nested_exact(resumed.history, eager.history)
    _assert_nested_exact(resumed._prev_shares, eager._prev_shares)
    assert resumed.step_idx == eager.step_idx
    assert resumed.accepted_tokens == eager.accepted_tokens
    assert resumed.data_cursor == eager.data_cursor


def test_clean_target_site_mask_zero_probability_is_exact_rng_identity():
    trainer = Trainer(
        BlockCrosscoder(CFG),
        train_cfg(total_steps=1, encoder_site_mask_probability=0.0),
    )
    observed = torch.tensor([[True, False, True, True], [True, True, True, False]])
    torch.manual_seed(4567)
    before = torch.get_rng_state().clone()
    actual = trainer._encoder_observation_mask(observed)
    after = torch.get_rng_state()
    assert actual is observed
    assert torch.equal(actual, observed)
    assert torch.equal(before, after)


def test_clean_target_site_mask_is_deterministic_subset_and_repairs_rows():
    trainer = Trainer(
        BlockCrosscoder(CFG),
        train_cfg(total_steps=1, encoder_site_mask_probability=0.10),
    )
    observed = torch.ones(4096, S, dtype=torch.bool)
    observed[::3] = False
    observed[::3, 0] = True
    torch.manual_seed(9182)
    first = trainer._encoder_observation_mask(observed)
    torch.manual_seed(9182)
    second = trainer._encoder_observation_mask(observed)
    assert torch.equal(first, second)
    assert not bool((first & ~observed).any())
    assert bool(first.any(dim=1).all())
    assert int((observed & ~first).sum()) > 0


def test_structured_clean_target_site_masks_have_exact_cardinality():
    observed = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, True],
            [False, True, True, False],
        ]
    )
    hidden = Trainer(
        BlockCrosscoder(CFG),
        train_cfg(total_steps=1, encoder_site_mask_mode="exactly_one_hidden"),
    )
    torch.manual_seed(771)
    hidden_first = hidden._encoder_observation_mask(observed)
    torch.manual_seed(771)
    hidden_second = hidden._encoder_observation_mask(observed)
    assert torch.equal(hidden_first, hidden_second)
    assert torch.equal(hidden_first.sum(dim=1), observed.sum(dim=1) - 1)
    assert not bool((hidden_first & ~observed).any())

    retained = Trainer(
        BlockCrosscoder(CFG),
        train_cfg(total_steps=1, encoder_site_mask_mode="exactly_one_retained"),
    )
    torch.manual_seed(882)
    retained_mask = retained._encoder_observation_mask(observed)
    assert torch.equal(
        retained_mask.sum(dim=1), torch.ones(len(observed), dtype=torch.long)
    )
    assert not bool((retained_mask & ~observed).any())

    with pytest.raises(ValueError, match="at least two"):
        hidden._encoder_observation_mask(torch.tensor([[True, False, False, False]]))


def test_clean_target_mask_hides_encoder_input_but_not_clean_loss_targets(monkeypatch):
    cfg = BSCConfig(
        n_blocks=1,
        block_dim=1,
        n_sites=2,
        d_model=1,
        k=1,
        decoder_constraint="free",
        decoder_bias=False,
    )
    model = BlockCrosscoder(cfg)
    with torch.no_grad():
        assert model.D is not None and model.E is not None
        model.D.zero_()
        model.E.zero_()
    trainer = Trainer(
        model,
        train_cfg(
            total_steps=1,
            lr=1e-3,
            encoder_site_mask_probability=0.10,
        ),
    )
    encoder_mask = torch.tensor([[True, False], [True, False]])
    monkeypatch.setattr(
        trainer, "_encoder_observation_mask", lambda observed: encoder_mask
    )
    x = torch.tensor([[[1.0], [3.0]], [[1.0], [3.0]]])
    record = trainer.step(x)
    # Both clean sites are targets: (1^2 + 3^2) / 2 coordinates = 5.
    assert record["rec"] == pytest.approx(5.0)
    assert record["encoder_site_keep_fraction"] == pytest.approx(0.5)


def test_clean_target_mask_passes_truth_mask_not_augmented_mask_to_aux(monkeypatch):
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=2,
            block_dim=1,
            n_sites=2,
            d_model=1,
            k=1,
            decoder_constraint="free",
        )
    )
    trainer = Trainer(
        model,
        train_cfg(
            total_steps=1,
            aux_variant="fel",
            s_aux=1,
            encoder_site_mask_probability=0.10,
        ),
    )
    encoder_mask = torch.tensor([[True, False], [True, False]])
    monkeypatch.setattr(
        trainer, "_encoder_observation_mask", lambda observed: encoder_mask
    )
    captured = {}

    def capture_aux(model, x, out, variant, dead, s_aux, **kwargs):
        captured.update(kwargs)
        return x.sum() * 0.0

    monkeypatch.setattr(trainer_module, "aux_loss", capture_aux)
    trainer.step(torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]]))
    assert torch.equal(captured["observation_mask"], torch.ones(2, 2, dtype=torch.bool))
    assert torch.equal(captured["encoder_observed"], encoder_mask)


def test_clean_target_mask_config_and_source_fusion_fail_closed():
    with pytest.raises(ValueError, match="encoder_site_mask_probability"):
        train_cfg(encoder_site_mask_probability=0.03)
    with pytest.raises(ValueError, match="mode itself defines"):
        train_cfg(
            encoder_site_mask_mode="exactly_one_hidden",
            encoder_site_mask_probability=0.02,
        )
    source = BlockCrosscoder(
        BSCConfig(
            n_blocks=4,
            block_dim=1,
            n_sites=2,
            d_model=3,
            k=1,
            decoder_constraint="free",
            encoder_fusion="source",
        )
    )
    with pytest.raises(ValueError, match="source-only"):
        Trainer(source, train_cfg(encoder_site_mask_probability=0.02))
    with pytest.raises(ValueError, match="source-only"):
        Trainer(
            source,
            train_cfg(encoder_site_mask_mode="exactly_one_retained"),
        )

    rescaled = BlockCrosscoder(
        BSCConfig(
            n_blocks=4,
            block_dim=1,
            n_sites=2,
            d_model=3,
            k=1,
            decoder_constraint="free",
            encoder_fusion="availability_rescaled_sum",
        )
    )
    trainer = Trainer(
        rescaled,
        train_cfg(total_steps=1, encoder_site_mask_probability=0.10),
    )
    assert trainer.master.cfg.encoder_fusion == "availability_rescaled_sum"


def test_site_mask_rng_and_factorized_parameters_resume_exactly(device, tmp_path):
    cfg = BSCConfig(
        n_blocks=6,
        block_dim=2,
        n_sites=4,
        d_model=5,
        k=2,
        decoder_constraint="free",
        site_rank=2,
        seed=123,
    )
    batches = [
        torch.randn(64, 4, 5, generator=torch.Generator().manual_seed(seed)).to(device)
        for seed in range(8)
    ]
    torch.manual_seed(9901)
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(
            total_steps=8,
            forward_dtype="bf16",
            encoder_site_mask_probability=0.10,
        ),
    )
    for batch in batches[:3]:
        trainer.step(batch)
    checkpoint = tmp_path / "factorized-site-mask.pt"
    trainer.save_checkpoint(checkpoint)
    continued = [trainer.step(batch) for batch in batches[3:]]

    resumed = Trainer.load_checkpoint(checkpoint, device=device)
    replayed = [resumed.step(batch) for batch in batches[3:]]
    assert [row["rec"] for row in replayed] == pytest.approx(
        [row["rec"] for row in continued], rel=1e-5, abs=1e-7
    )
    assert [row["encoder_site_keep_fraction"] for row in replayed] == [
        row["encoder_site_keep_fraction"] for row in continued
    ]
    for actual, expected in zip(
        resumed.master.parameters(), trainer.master.parameters()
    ):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    for master, forward in zip(resumed.master.parameters(), resumed.fwd.parameters()):
        assert torch.equal(forward, master.to(torch.bfloat16))


def test_factorized_checkpoint_refuses_stale_logical_optimizer_core_shape(
    device, tmp_path
):
    cfg = BSCConfig(
        n_blocks=6,
        block_dim=2,
        n_sites=4,
        d_model=5,
        k=2,
        decoder_constraint="free",
        site_rank=2,
        seed=124,
    )
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=3, forward_dtype="bf16"),
    )
    batch = torch.randn(
        32,
        cfg.n_sites,
        cfg.d_model,
        generator=torch.Generator().manual_seed(125),
    ).to(device)
    trainer.step(batch)
    checkpoint = tmp_path / "factorized-packed.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)

    named = {
        id(parameter): name for name, parameter in trainer.master.named_parameters()
    }
    found = False
    for stored_group, live_group in zip(
        payload["optimizer"]["param_groups"],
        trainer.opt.param_groups,
        strict=True,
    ):
        for state_id, parameter in zip(
            stored_group["params"], live_group["params"], strict=True
        ):
            state = payload["optimizer"]["state"].get(state_id, {})
            for state_name, tensor in state.items():
                if state_name != "step":
                    assert tensor.shape == parameter.shape
                    assert tensor.dtype == parameter.dtype
            if named[id(parameter)] == "D_core":
                packed = state["exp_avg"]
                state["exp_avg"] = (
                    packed.view(
                        cfg.n_blocks,
                        cfg.block_dim,
                        cfg.site_rank,
                        cfg.d_model,
                    )
                    .permute(2, 0, 1, 3)
                    .contiguous()
                )
                found = True
    assert found
    stale = tmp_path / "factorized-stale-optimizer.pt"
    torch.save(payload, stale)
    with pytest.raises(ValueError, match="optimizer exp_avg shape"):
        Trainer.load_checkpoint(stale, device=device)


def test_lr_schedule_linear_fifth():
    """SASA B.3 schedule: warmup, constant, linear decay over the final
    fifth. Cosine remains the default and is untouched."""
    from block_crosscoder_experiment.trainer import _lr_factor

    f = _lr_factor(train_cfg(total_steps=100, warmup_steps=10, schedule="linear_fifth"))
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(50) == 1.0
    assert f(79) == 1.0
    # Twenty final optimizer updates occupy indices 80..99, so inclusive
    # endpoints leave nineteen interpolation intervals.
    assert f(90) == pytest.approx(9 / 19)
    assert f(99) == pytest.approx(0.0)
    assert f(100) == 0.0

    g = _lr_factor(train_cfg(total_steps=100, warmup_steps=10))  # cosine default
    assert (g(54) + g(55)) / 2 == pytest.approx(0.5)
    assert g(99) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="schedule"):
        train_cfg(schedule="nonsense")


def test_checkpoint_free_space_floor(device, tmp_path, monkeypatch):
    """save_checkpoint aborts before writing when the free-space floor would be
    breached, and leaves no partial files behind."""
    from types import SimpleNamespace

    import block_crosscoder_experiment.trainer as trainer_mod

    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=10))
    trainer.step(planted_batches(device, n_batches=1)[0])

    total = 1_000_000_000
    tight = SimpleNamespace(total=total, used=total, free=int(0.15 * total))
    monkeypatch.setattr(trainer_mod.shutil, "disk_usage", lambda _: tight)
    with pytest.raises(RuntimeError, match="free-space floor"):
        trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert not (tmp_path / "ckpt.pt").exists()
    assert not (tmp_path / "ckpt.pt.tmp").exists()


def test_threshold_calibration(device):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=40))
    calib = planted_batches(device, n_batches=20, seed=7)
    trainer.fit(planted_batches(device))
    target = float(CFG.k)
    model.fit_threshold_(calib, target)
    counts = torch.cat(
        [model(x, mode="threshold").mask.sum(dim=1).float() for x in calib]
    )
    assert abs(float(counts.mean()) - target) < 0.25


def test_post_step_nonfinite_refuses_run(device, monkeypatch):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=2))
    original_step = trainer.opt.step

    def poison_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            next(trainer.master.parameters()).view(-1)[0] = float("nan")
        return result

    monkeypatch.setattr(trainer.opt, "step", poison_step)
    with pytest.raises(RuntimeError, match="optimizer produced non-finite"):
        trainer.step(planted_batches(device, n_batches=1, seed=43)[0])


@pytest.mark.parametrize(
    "implementation",
    (
        DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
        DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    ),
)
def test_qr_step_reuses_global_input_and_transactional_output_finite_checks(
    device,
    monkeypatch,
    implementation,
):
    cfg = BSCConfig(
        **{
            **CFG.__dict__,
            "decoder_constraint": "qr",
            "decoder_retraction_implementation": implementation,
            "encoder_mode": "tied",
            "decoder_bias": False,
        }
    )
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=2, log_every=100),
    )
    trainer_finite = trainer_module._all_finite
    gram_finite = gram_module._all_finite
    trainer_calls = 0
    gram_calls = 0

    def counted_trainer(obj):
        nonlocal trainer_calls
        trainer_calls += 1
        return trainer_finite(obj)

    def counted_gram(value):
        nonlocal gram_calls
        gram_calls += 1
        return gram_finite(value)

    monkeypatch.setattr(trainer_module, "_all_finite", counted_trainer)
    monkeypatch.setattr(gram_module, "_all_finite", counted_gram)
    trainer.step(planted_batches(device, n_batches=1, seed=430)[0])

    # One global parameter/state scan establishes QR input finiteness. QR's
    # post-Gram validates its candidate before the transactional copy, so no
    # candidate or post-projection decoder rescan remains.
    assert trainer_calls == 1
    assert gram_calls == 0


@pytest.mark.parametrize(
    "constraint",
    ("frobenius", "unit_frobenius", "unit_latent", "free"),
)
def test_finiteness_preserving_projection_reuses_global_scan(
    device,
    monkeypatch,
    constraint,
):
    cfg = BSCConfig(
        **{
            **CFG.__dict__,
            "decoder_constraint": constraint,
            "decoder_retraction_implementation": None,
            "encoder_mode": "untied",
            "encoder_constraint": (
                "unit_latent" if constraint == "unit_latent" else "none"
            ),
            "decoder_bias": False,
        }
    )
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=2, log_every=100),
    )
    original = trainer_module._all_finite
    finite_calls = 0

    def counted(value):
        nonlocal finite_calls
        finite_calls += 1
        return original(value)

    monkeypatch.setattr(trainer_module, "_all_finite", counted)
    trainer.step(planted_batches(device, n_batches=1, seed=434)[0])
    assert finite_calls == 1


def test_polar_projection_retains_post_projection_finite_scan(device, monkeypatch):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=2, log_every=100),
    )
    original = trainer_module._all_finite
    finite_calls = 0

    def counted(value):
        nonlocal finite_calls
        finite_calls += 1
        return original(value)

    monkeypatch.setattr(trainer_module, "_all_finite", counted)
    trainer.step(planted_batches(device, n_batches=1, seed=435)[0])
    assert finite_calls == 2


@pytest.mark.parametrize(
    "constraint",
    ("frobenius", "unit_frobenius", "unit_latent"),
)
def test_norm_projection_certificate_handles_finite_overflow_extrema(
    device,
    constraint,
):
    cfg = BSCConfig(
        **{
            **CFG.__dict__,
            "decoder_constraint": constraint,
            "decoder_retraction_implementation": None,
            "encoder_mode": "untied",
            "encoder_constraint": (
                "unit_latent" if constraint == "unit_latent" else "none"
            ),
            "decoder_bias": False,
        }
    )
    model = BlockCrosscoder(cfg).to(device)
    with torch.no_grad():
        assert model.D is not None
        model.D.fill_(torch.finfo(model.D.dtype).max)
        if model.E is not None:
            model.E.fill_(torch.finfo(model.E.dtype).max)
    _, mutated, certified = trainer_module._project_decoder_(
        model,
        qr_input_finite=True,
    )
    assert mutated
    assert len(certified) == len(mutated)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in certified)


def test_qr_step_refuses_optimizer_poison_before_trusting_input(
    device,
    monkeypatch,
):
    cfg = BSCConfig(
        **{
            **CFG.__dict__,
            "decoder_constraint": "qr",
            "decoder_retraction_implementation": (
                DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
            ),
            "encoder_mode": "tied",
        }
    )
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=2, log_every=100),
    )
    original_step = trainer.opt.step
    projection_calls = 0

    def poison_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            assert trainer.master.D is not None
            trainer.master.D.view(-1)[0] = float("nan")
        return result

    original_projection = trainer_module._project_decoder_

    def counted_projection(*args, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(trainer.opt, "step", poison_step)
    monkeypatch.setattr(trainer_module, "_project_decoder_", counted_projection)
    with pytest.raises(RuntimeError, match="optimizer produced non-finite"):
        trainer.step(planted_batches(device, n_batches=1, seed=431)[0])
    assert projection_calls == 0


def test_qr_step_retains_separate_encoder_projection_finite_scan(
    device,
    monkeypatch,
):
    cfg = BSCConfig(
        **{
            **CFG.__dict__,
            "decoder_constraint": "qr",
            "decoder_retraction_implementation": (
                DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
            ),
            "encoder_mode": "untied",
            "encoder_constraint": "unit_latent",
        }
    )
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=2, log_every=100),
    )
    project_rows = model_module._project_latent_rows_count_tensor_

    def poison_encoder(tensor):
        result = project_rows(tensor)
        tensor[(0,) * tensor.ndim] = float("nan")
        return result

    monkeypatch.setattr(
        model_module,
        "_project_latent_rows_count_tensor_",
        poison_encoder,
    )
    with pytest.raises(RuntimeError, match="decoder projection produced non-finite"):
        trainer.step(planted_batches(device, n_batches=1, seed=432)[0])


def test_qr_step_does_not_trust_custom_projection_backend(device, monkeypatch):
    cfg = BSCConfig(
        **{
            **CFG.__dict__,
            "decoder_constraint": "qr",
            "decoder_retraction_implementation": (
                DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
            ),
            "encoder_mode": "tied",
        }
    )
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train_cfg(total_steps=2, log_every=100),
    )

    def poison_projection():
        assert trainer.master.D is not None
        with torch.no_grad():
            trainer.master.D[(0,) * trainer.master.D.ndim] = float("nan")
        return (
            torch.zeros((), dtype=torch.int64, device=device),
            (trainer.master.D,),
        )

    monkeypatch.setattr(
        trainer.master,
        "_project_decoder_with_state_",
        poison_projection,
    )
    with pytest.raises(RuntimeError, match="decoder projection produced non-finite"):
        trainer.step(planted_batches(device, n_batches=1, seed=433)[0])


def test_nonlogging_unclipped_fast_step_keeps_only_finite_scans(device, monkeypatch):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(
            total_steps=3,
            log_every=100,
            retract_every=100,
            gradient_clip_norm=None,
        ),
    )
    batches = planted_batches(device, n_batches=2, seed=44)
    trainer.step(batches[0])
    original = trainer_module._all_finite
    finite_calls = 0
    gradient_guard_calls = 0
    original_gradient_guard = trainer_module._finite_gradients_with_l2_guard

    def counted(obj):
        nonlocal finite_calls
        finite_calls += 1
        return original(obj)

    def counted_gradient_guard(*args, **kwargs):
        nonlocal gradient_guard_calls
        gradient_guard_calls += 1
        return original_gradient_guard(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "_all_finite", counted)
    monkeypatch.setattr(
        trainer_module,
        "_finite_gradients_with_l2_guard",
        counted_gradient_guard,
    )
    record = trainer.step(batches[1], materialize_record=False)
    assert record is None
    assert finite_calls == 1
    assert gradient_guard_calls == 1


def test_nonlogging_fast_step_refuses_nonfinite_gradient_before_optimizer(
    device, monkeypatch
):
    trainer = Trainer(
        BlockCrosscoder(CFG).to(device),
        train_cfg(total_steps=3, log_every=100, retract_every=100),
    )
    batches = planted_batches(device, n_batches=2, seed=45)
    trainer.step(batches[0])
    parameter = next(
        parameter
        for parameter in trainer.master.parameters()
        if parameter.requires_grad
    )
    handle = parameter.register_hook(lambda gradient: gradient.fill_(float("nan")))
    optimizer_calls = 0
    original_step = trainer.opt.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_calls
        optimizer_calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer.opt, "step", counted_step)
    with pytest.raises(RuntimeError, match="non-finite loss/gradient"):
        trainer.step(batches[1], materialize_record=False)
    handle.remove()
    assert optimizer_calls == 0
    assert trainer.step_idx == 1
    assert trainer.accepted_tokens == len(batches[0])


@pytest.mark.parametrize("magnitude", (1e30, 1e38))
def test_fast_gradient_guard_matches_historical_finite_l2_refusal(device, magnitude):
    gradients = [torch.full((4096,), magnitude, device=device)]
    historical = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(gradient.float()) for gradient in gradients]
        )
    )
    actual = trainer_module._finite_gradients_with_l2_guard(
        torch.tensor(1.0, device=device),
        torch.tensor(1.0, device=device),
        gradients,
    )
    assert actual is bool(torch.isfinite(historical))


def test_checkpoint_binding_roundtrip_and_mismatch(device, tmp_path):
    cfg = train_cfg(total_steps=2)
    binding = {
        "whitener_hash": "abc",
        "sites": [9, 12],
        "gauge": "whiten",
        "model_cfg": asdict(CFG),
        "train_cfg": asdict(cfg),
    }
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, cfg, run_binding=binding)
    path = tmp_path / "bound.pt"
    trainer.save_checkpoint(path)
    restored = Trainer.load_checkpoint(path, device=device, expected_binding=binding)
    assert restored.run_binding == binding
    with pytest.raises(ValueError, match="binding mismatch"):
        Trainer.load_checkpoint(
            path,
            device=device,
            expected_binding={**binding, "whitener_hash": "different"},
        )


def test_expected_binding_rejects_legacy_checkpoint(device, tmp_path):
    trainer = Trainer(BlockCrosscoder(CFG).to(device), train_cfg(total_steps=2))
    path = tmp_path / "legacy.pt"
    trainer.save_checkpoint(path)
    with pytest.raises(ValueError, match="legacy/unbound"):
        Trainer.load_checkpoint(
            path, device=device, expected_binding={"whitener_hash": "abc"}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("foreach", True),
        ("fused", True),
        ("betas", (0.8, 0.9)),
        ("eps", 7e-7),
        ("weight_decay", 0.125),
    ),
)
def test_checkpoint_refuses_optimizer_group_contract_forgery(
    device,
    tmp_path,
    field,
    value,
):
    trainer = Trainer(BlockCrosscoder(CFG).to(device), train_cfg(total_steps=2))
    path = tmp_path / "optimizer-contract.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["optimizer"]["param_groups"][0][field] = value
    forged = tmp_path / f"forged-{field}.pt"
    torch.save(payload, forged)
    with pytest.raises(ValueError, match="optimizer|foreach|fused|betas|epsilon|decay"):
        Trainer.load_checkpoint(forged, device=device)


def test_checkpoint_refuses_optimizer_kind_forgery(device, tmp_path):
    trainer = Trainer(BlockCrosscoder(CFG).to(device), train_cfg(total_steps=2))
    path = tmp_path / "optimizer-kind.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["optimizer_kind"] = "adam"
    forged = tmp_path / "forged-kind.pt"
    torch.save(payload, forged)
    with pytest.raises(ValueError, match="optimizer kind"):
        Trainer.load_checkpoint(forged, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("optimizer_name", ("adam", "adamw"))
def test_fused_cuda_optimizer_resumes_bit_exactly(
    tmp_path,
    optimizer_name,
):
    cfg = BSCConfig(
        n_blocks=32,
        block_dim=4,
        n_sites=4,
        d_model=32,
        k=4,
        seed=1661,
        decoder_constraint="free",
    )
    training = train_cfg(
        total_steps=4,
        optimizer=optimizer_name,
        fused=True,
        retract_every=100,
        log_every=1,
    )
    base = BlockCrosscoder(cfg).to("cuda")
    uninterrupted = Trainer(copy.deepcopy(base), training)
    split = Trainer(copy.deepcopy(base), training)
    batches = planted_batches("cuda", n_batches=4, batch=64, seed=1662)
    uninterrupted_records = [uninterrupted.step(batch) for batch in batches]
    split_records = [split.step(batch) for batch in batches[:2]]
    path = tmp_path / f"fused-{optimizer_name}.pt"
    split.save_checkpoint(path)
    resumed = Trainer.load_checkpoint(path, device="cuda")
    split_records.extend(resumed.step(batch) for batch in batches[2:])
    _assert_nested_exact(split_records, uninterrupted_records)
    _assert_nested_exact(resumed.master.state_dict(), uninterrupted.master.state_dict())
    _assert_nested_exact(resumed.opt.state_dict(), uninterrupted.opt.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_cuda_optimizer_bounds_scalar_trajectory_and_support():
    cfg = BSCConfig(
        n_blocks=256,
        block_dim=4,
        n_sites=4,
        d_model=128,
        k=8,
        seed=1663,
        selection="token_topk",
        encoder_mode="untied",
        decoder_constraint="qr",
    )
    common = {
        "total_steps": 20,
        "optimizer": "adamw",
        "lr": 3e-4,
        "retract_every": 1,
        "forward_dtype": "bf16",
        "log_every": 1,
    }
    base = BlockCrosscoder(cfg).to("cuda")
    scalar = Trainer(copy.deepcopy(base), train_cfg(**common, fused=False))
    fused = Trainer(copy.deepcopy(base), train_cfg(**common, fused=True))
    generator = torch.Generator().manual_seed(1664)
    batches = [
        torch.randn(512, 4, 128, generator=generator).to("cuda") for _ in range(20)
    ]
    scalar_records = []
    fused_records = []
    intersections = 0
    unions = 0
    for batch in batches:
        scalar_records.append(scalar.step(batch))
        fused_records.append(fused.step(batch))
        with torch.no_grad():
            scalar_mask = scalar.fwd(batch.to(torch.bfloat16)).mask
            fused_mask = fused.fwd(batch.to(torch.bfloat16)).mask
        intersections += int((scalar_mask & fused_mask).sum())
        unions += int((scalar_mask | fused_mask).sum())

    maximum_loss_drift = max(
        abs(actual["total"] - expected["total"]) / max(abs(expected["total"]), 1e-30)
        for actual, expected in zip(fused_records, scalar_records, strict=True)
    )
    assert maximum_loss_drift <= 5e-4
    assert intersections / max(unions, 1) >= 0.99
    assert (
        _nested_relative_l2(
            fused.master.state_dict(),
            scalar.master.state_dict(),
        )
        <= 0.05
    )
    assert (
        _nested_relative_l2(
            fused.opt.state_dict()["state"],
            scalar.opt.state_dict()["state"],
        )
        <= 0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_guarded_map_nuclear_bounds_trajectory_and_resumes_exactly(tmp_path):
    common = dict(
        n_blocks=64,
        block_dim=4,
        n_sites=4,
        d_model=64,
        k=4,
        seed=5270,
        selection="token_topk",
        encoder_mode="untied",
        decoder_constraint="qr",
        regularizer="map_nuclear",
        lambda_regularizer=0.03,
    )
    optimized_cfg = BSCConfig(
        **common,
        map_nuclear_implementation=MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    )
    reference_cfg = BSCConfig(
        **common,
        map_nuclear_implementation=MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    )
    training = train_cfg(
        total_steps=25,
        lr=3e-4,
        warmup_steps=1,
        forward_dtype="bf16",
        fused=True,
        retract_every=1,
        log_every=1,
    )
    generator = torch.Generator().manual_seed(5271)
    batches = list(
        torch.randn(25 * 256, 4, 64, generator=generator).to("cuda").split(256)
    )

    def run(
        trainer: Trainer,
        step_batches: list[torch.Tensor],
    ) -> tuple[list[dict], list[torch.Tensor]]:
        records = []
        supports = []
        for batch in step_batches:
            record = trainer.step(batch)
            assert record is not None
            records.append(record)
            with torch.no_grad():
                supports.append(trainer.fwd(batch.to(torch.bfloat16)).mask.cpu())
        return records, supports

    optimized = Trainer(BlockCrosscoder(optimized_cfg).to("cuda"), training)
    optimized_records, optimized_supports = run(optimized, batches)

    resumable = Trainer(BlockCrosscoder(optimized_cfg).to("cuda"), training)
    resumed_records, resumed_supports = run(resumable, batches[:12])
    checkpoint = tmp_path / "map-nuclear-matmul.pt"
    resumable.save_checkpoint(checkpoint)
    resumed = Trainer.load_checkpoint(checkpoint, device="cuda")
    tail_records, tail_supports = run(resumed, batches[12:])
    resumed_records.extend(tail_records)
    resumed_supports.extend(tail_supports)
    _assert_nested_exact(resumed_records, optimized_records)
    _assert_nested_exact(resumed_supports, optimized_supports)
    _assert_nested_exact(resumed.master.state_dict(), optimized.master.state_dict())
    _assert_nested_exact(resumed.opt.state_dict(), optimized.opt.state_dict())

    reference = Trainer(BlockCrosscoder(reference_cfg).to("cuda"), training)
    reference_records, reference_supports = run(reference, batches)
    maximum_loss_drift = max(
        abs(actual["total"] - expected["total"])
        / max(abs(expected["total"]), 1e-30)
        for actual, expected in zip(
            optimized_records,
            reference_records,
            strict=True,
        )
    )
    maximum_regularizer_drift = max(
        abs(actual["regularizer"] - expected["regularizer"])
        / max(abs(expected["regularizer"]), 1e-30)
        for actual, expected in zip(
            optimized_records,
            reference_records,
            strict=True,
        )
    )
    intersections = sum(
        int((actual & expected).sum())
        for actual, expected in zip(
            optimized_supports,
            reference_supports,
            strict=True,
        )
    )
    unions = sum(
        int((actual | expected).sum())
        for actual, expected in zip(
            optimized_supports,
            reference_supports,
            strict=True,
        )
    )
    assert maximum_loss_drift <= 1e-5
    assert maximum_regularizer_drift <= 2e-5
    assert intersections / max(unions, 1) >= 0.995
    assert _nested_relative_l2(
        optimized.master.state_dict(),
        reference.master.state_dict(),
    ) <= 2e-3
    assert _nested_relative_l2(
        optimized.opt.state_dict()["state"],
        reference.opt.state_dict()["state"],
    ) <= 1e-5


def _fast_decoded_energy_cfg() -> BSCConfig:
    return BSCConfig(
        n_blocks=12,
        block_dim=3,
        n_sites=3,
        d_model=16,
        k=3,
        seed=719,
        selection="token_topk",
        selection_score="decoded_energy",
        decoded_energy_implementation=(DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION),
        decoder_constraint="gram",
    )


def test_stiefel_decoded_energy_trainer_requires_every_step_retraction(device):
    with pytest.raises(ValueError, match="after every optimizer step"):
        Trainer(
            BlockCrosscoder(_fast_decoded_energy_cfg()).to(device),
            train_cfg(total_steps=2, retract_every=2),
        )


def test_stiefel_decoded_energy_diagnostics_resume_and_save_refusal(
    device,
    tmp_path,
):
    cfg = _fast_decoded_energy_cfg()
    train = train_cfg(total_steps=3, log_every=1, forward_dtype="fp32")
    binding = {
        "model_cfg": asdict(cfg),
        "train_cfg": asdict(train),
        "cell_id": "stiefel-fast-test",
    }
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train,
        run_binding=binding,
    )
    batch = torch.randn(
        32,
        cfg.n_sites,
        cfg.d_model,
        generator=torch.Generator().manual_seed(720),
    ).to(device)
    record = trainer.step(batch)
    assert record["decoded_energy_master_gram_residual"] <= 1e-4
    assert (
        record["decoder_constraint_residual_master"]
        == record["decoded_energy_master_gram_residual"]
    )

    checkpoint = tmp_path / "stiefel-fast.pt"
    trainer.save_checkpoint(checkpoint)
    restored = Trainer.load_checkpoint(
        checkpoint,
        device=device,
        expected_binding=binding,
    )
    _assert_nested_exact(restored.master.state_dict(), trainer.master.state_dict())
    _assert_nested_exact(restored.opt.state_dict(), trainer.opt.state_dict())
    assert (
        restored.master.cfg.decoded_energy_implementation
        == DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
    )
    next_batch = torch.randn(
        32,
        cfg.n_sites,
        cfg.d_model,
        generator=torch.Generator().manual_seed(721),
    ).to(device)
    continued = trainer.step(next_batch)
    replayed = restored.step(next_batch)
    _assert_nested_exact(replayed, continued)
    _assert_nested_exact(restored.master.state_dict(), trainer.master.state_dict())
    _assert_nested_exact(restored.opt.state_dict(), trainer.opt.state_dict())

    with torch.no_grad():
        assert trainer.master.D is not None
        trainer.master.D[0, 0, 0, 0].add_(1.0)
    refused = tmp_path / "off-manifold.pt"
    with pytest.raises(RuntimeError, match="Gram residual"):
        trainer.save_checkpoint(refused)
    assert not refused.exists()


def test_stiefel_decoded_energy_checkpoint_identity_is_bound(device, tmp_path):
    cfg = _fast_decoded_energy_cfg()
    train = train_cfg(total_steps=1)
    binding = {
        "model_cfg": asdict(cfg),
        "train_cfg": asdict(train),
    }
    trainer = Trainer(
        BlockCrosscoder(cfg).to(device),
        train,
        run_binding=binding,
    )
    checkpoint = tmp_path / "bound-fast.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    missing = {**payload, "model_cfg": dict(payload["model_cfg"])}
    missing["model_cfg"].pop("decoded_energy_implementation")
    missing_path = tmp_path / "missing-fast-id.pt"
    torch.save(missing, missing_path)
    with pytest.raises(ValueError, match="lacks decoded_energy_implementation"):
        Trainer.load_checkpoint(missing_path, device=device)

    missing_isolated = {**payload, "model_cfg": dict(payload["model_cfg"])}
    missing_isolated["model_cfg"].pop("isolated_loss_decrease_implementation")
    missing_isolated_path = tmp_path / "missing-isolated-loss-id.pt"
    torch.save(missing_isolated, missing_isolated_path)
    with pytest.raises(
        ValueError,
        match="lacks isolated_loss_decrease_implementation",
    ):
        Trainer.load_checkpoint(missing_isolated_path, device=device)

    missing_sparse = {**payload, "model_cfg": dict(payload["model_cfg"])}
    missing_sparse["model_cfg"].pop("sparse_decode_implementation")
    missing_sparse_path = tmp_path / "missing-sparse-decode-id.pt"
    torch.save(missing_sparse, missing_sparse_path)
    with pytest.raises(ValueError, match="lacks sparse_decode_implementation"):
        Trainer.load_checkpoint(missing_sparse_path, device=device)

    missing_map_nuclear = {**payload, "model_cfg": dict(payload["model_cfg"])}
    missing_map_nuclear["model_cfg"].pop("map_nuclear_implementation")
    missing_map_nuclear_path = tmp_path / "missing-map-nuclear-id.pt"
    torch.save(missing_map_nuclear, missing_map_nuclear_path)
    with pytest.raises(ValueError, match="lacks map_nuclear_implementation"):
        Trainer.load_checkpoint(missing_map_nuclear_path, device=device)

    forged = {**payload, "model_cfg": dict(payload["model_cfg"])}
    forged["model_cfg"]["decoded_energy_implementation"] = (
        DECODED_ENERGY_EXACT_IMPLEMENTATION
    )
    forged_path = tmp_path / "forged-fast-id.pt"
    torch.save(forged, forged_path)
    with pytest.raises(ValueError, match="run binding mismatch"):
        Trainer.load_checkpoint(forged_path, device=device)

    forged_map_nuclear = {**payload, "model_cfg": dict(payload["model_cfg"])}
    forged_map_nuclear["model_cfg"]["map_nuclear_implementation"] = (
        MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION
    )
    forged_map_nuclear_path = tmp_path / "forged-map-nuclear-id.pt"
    torch.save(forged_map_nuclear, forged_map_nuclear_path)
    with pytest.raises(ValueError, match="run binding mismatch"):
        Trainer.load_checkpoint(forged_map_nuclear_path, device=device)

    forged_polar = {**payload, "model_cfg": dict(payload["model_cfg"])}
    forged_polar["model_cfg"]["decoder_retraction_implementation"] = (
        DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION
    )
    forged_polar_path = tmp_path / "forged-polar-id.pt"
    torch.save(forged_polar, forged_polar_path)
    with pytest.raises(ValueError, match="run binding mismatch"):
        Trainer.load_checkpoint(forged_polar_path, device=device)


def test_factor_regularizer_checkpoint_refuses_stale_v3_identity(device, tmp_path):
    cfg = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=4,
        d_model=8,
        k=2,
        selection="token_topk",
        decoder_constraint="free",
        site_rank=2,
        regularizer="map_nuclear",
        lambda_regularizer=0.1,
    )
    assert cfg.factorized_execution_implementation == (
        FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION
    )
    train = train_cfg(total_steps=1, warmup_steps=0)
    binding = {"model_cfg": asdict(cfg), "train_cfg": asdict(train)}
    trainer = Trainer(BlockCrosscoder(cfg).to(device), train, run_binding=binding)
    trainer.step(torch.randn(16, 4, 8, device=device))
    checkpoint = tmp_path / "factor-regularizer-v4.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    stale = copy.deepcopy(payload)
    stale["model_cfg"]["factorized_execution_implementation"] = (
        FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
    )
    stale["run_binding"]["model_cfg"]["factorized_execution_implementation"] = (
        FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
    )
    stale_path = tmp_path / "factor-regularizer-stale-v3.pt"
    torch.save(stale, stale_path)
    with pytest.raises(ValueError, match="v4 factor-regularizer"):
        Trainer.load_checkpoint(stale_path, device=device)
