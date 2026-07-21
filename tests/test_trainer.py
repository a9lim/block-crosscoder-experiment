"""Trainer checks for update ordering, precision copies, Aux, and replay."""

from dataclasses import asdict

import pytest
import torch

import block_crosscoder_experiment.trainer as trainer_module
from block_crosscoder_experiment.gram import gram_residual
from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
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


def test_fp32_step_ordering_loss_falls(device):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, train_cfg(total_steps=60))
    batches = planted_batches(device)
    history = trainer.fit(batches)
    assert trainer.step_idx == 60
    assert history[-1]["rec"] < 0.5 * history[0]["rec"]
    assert all(rec["floor_hits"] == 0 for rec in history)
    assert history[-1]["decoder_constraint_residual_master"] < 1e-4


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


def test_projection_cadence_counts_completed_updates(device, monkeypatch):
    """Cadence two projects after updates 2 and 4, not 1, 3, and 5."""
    calls: list[int] = []
    project = trainer_module._project_decoder_

    def counted(model):
        calls.append(1)
        return project(model)

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


def test_dead_tracker_criteria(device):
    B = 64
    tracker = DeadTracker(n_blocks=4, capacity=8, device=device, max_tokens=6 * B)
    mask = torch.zeros(B, 4, dtype=torch.bool, device=device)
    mask[:, 0] = True  # block 0 always active
    first = mask.clone()
    first[0, 2] = True  # block 2 active once, in the first batch only

    tracker.update(first)
    for _ in range(3):
        tracker.update(mask)
    # Warmup gating: the 6-batch-equivalent token window is not yet full.
    assert not tracker.dead(
        "sasa", threshold=1e-4, window_tokens=6 * B, horizon_tokens=8 * B
    ).any()
    tracker.update(mask)
    tracker.update(mask)
    dead = tracker.dead(
        "sasa", threshold=1e-4, window_tokens=6 * B, horizon_tokens=8 * B
    )
    # Block 1 and 3: never active. Block 2: freq 1/384 > 1e-4. Block 0: alive.
    assert dead.tolist() == [False, True, False, True]
    freq = tracker.frequency(6 * B)
    assert abs(float(freq[2]) - 1 / (6 * B)) < 1e-6
    # Long-horizon at horizon=8: not full yet, then dead once block 2's
    # single activation scrolls out of the window.
    assert not tracker.dead(
        "long_horizon",
        threshold=1e-4,
        window_tokens=6 * B,
        horizon_tokens=8 * B,
    ).any()
    for _ in range(4):
        tracker.update(mask)
    dead_lh = tracker.dead(
        "long_horizon",
        threshold=1e-4,
        window_tokens=6 * B,
        horizon_tokens=8 * B,
    )
    assert dead_lh.tolist() == [False, True, True, True]


def test_dead_tracker_windows_are_token_denominated(device):
    tracker = DeadTracker(n_blocks=2, capacity=2, device=device, max_tokens=10)
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


def test_dead_tracker_slices_the_oldest_batch_at_the_exact_window(device):
    tracker = DeadTracker(n_blocks=2, capacity=2, device=device, max_tokens=1_000)
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
        n_blocks=2, block_dim=3, capacity=2, device=device, max_tokens=0
    )
    restored.load_state_dict(state)
    assert torch.equal(
        restored.coordinate_passes_since_fired,
        tracker.coordinate_passes_since_fired,
    )


def test_token_horizon_deadness_updates_at_current_batch_boundary(device):
    tracker = DeadTracker(n_blocks=3, capacity=2, device=device, max_tokens=0)
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
    restored = DeadTracker(n_blocks=3, capacity=2, device=device, max_tokens=0)
    restored.load_state_dict(tracker.state_dict())
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


def test_auxk_revives_dead_encoders(device):
    """The pilot's synthetic dead-encoder revival test: blocks with zeroed
    encoders are never selected, get flagged dead, and only the aux loss
    can pull them back (decoder shrinkage is impossible by construction;
    starvation is encoder-side)."""
    revived_norms = {}
    for variant in ("sasa", "none"):
        model = BlockCrosscoder(CFG).to(device)
        with torch.no_grad():
            model.E[:, :4] = 0.0  # kill blocks 0-3 encoder-side
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
        revived_norms[variant] = float(model.E.detach()[:, :4].float().norm())
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
