"""Trainer checks: the step->retract->recast ordering, master/forward-copy
sync, dead tracking, AuxK revival (the synthetic dead-encoder revival test
from the pilot spec, exercised early), checkpoint/resume, and threshold
calibration. The CUDA-only test verifies the ordering against the actual
8-bit-Adam implementation."""

import pytest
import torch

from block_crosscoder_experiment.gram import gram_residual
from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.trainer import DeadTracker, TrainConfig, Trainer

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
    base = dict(
        total_steps=100,
        lr=3e-3,
        warmup_steps=5,
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
    assert history[-1]["gram_residual_master"] < 1e-4


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
    assert trainer.history[-1]["gram_residual_postcast"] < 5e-2
    # Loss still falls through the cast/copy plumbing.
    assert trainer.history[-1]["rec"] < 0.7 * trainer.history[0]["rec"]


def test_k_anneal_schedule(device):
    """Budget annealing interpolates cfg.k linearly to the model's k over
    k_anneal_steps, then holds; a mid-anneal checkpoint restores the true
    target, not the interpolated value."""
    model = BlockCrosscoder(CFG).to(device)  # k = 3
    trainer = Trainer(
        model,
        train_cfg(total_steps=40, k_anneal_from=1.0, k_anneal_steps=20, log_every=1),
    )
    batches = planted_batches(device)
    trainer.fit(batches[:10])
    # After 10 of 20 anneal steps the last-applied k (step_idx 9) is
    # 1.0 + (3 - 1) * 9/20.
    assert abs(model.cfg.k - (1.0 + 2.0 * 9 / 20)) < 1e-9
    assert abs(trainer.history[0]["k"] - 1.0) < 1e-9
    trainer.fit(batches[10:])
    assert model.cfg.k == 3.0  # held at target after the anneal span
    assert trainer._k_final == 3.0
    # The model owns a config copy: annealing must not leak into the
    # caller's (here module-shared) BSCConfig.
    assert CFG.k == 3


def test_k_anneal_checkpoint_restores_target(device, tmp_path):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(
        model,
        train_cfg(total_steps=40, k_anneal_from=1.0, k_anneal_steps=20),
    )
    trainer.fit(planted_batches(device)[:10])  # stop mid-anneal
    path = tmp_path / "mid_anneal.pt"
    trainer.save_checkpoint(path)
    resumed = Trainer.load_checkpoint(path, device=device)
    assert resumed._k_final == 3.0
    resumed.fit(planted_batches(device)[10:])
    assert resumed.master.cfg.k == 3.0


def test_dead_tracker_criteria(device):
    B = 64
    tracker = DeadTracker(n_blocks=4, capacity=8, device=device)
    mask = torch.zeros(B, 4, dtype=torch.bool, device=device)
    mask[:, 0] = True  # block 0 always active
    first = mask.clone()
    first[0, 2] = True  # block 2 active once, in the first batch only

    tracker.update(first)
    for _ in range(3):
        tracker.update(mask)
    # Warmup gating: window not yet full at window=6.
    assert not tracker.dead("sasa", threshold=1e-4, window=6, horizon=8).any()
    tracker.update(mask)
    tracker.update(mask)
    dead = tracker.dead("sasa", threshold=1e-4, window=6, horizon=8)
    # Block 1 and 3: never active. Block 2: freq 1/384 > 1e-4. Block 0: alive.
    assert dead.tolist() == [False, True, False, True]
    freq = tracker.frequency(6)
    assert abs(float(freq[2]) - 1 / (6 * B)) < 1e-6
    # Long-horizon at horizon=8: not full yet, then dead once block 2's
    # single activation scrolls out of the window.
    assert not tracker.dead("long_horizon", threshold=1e-4, window=6, horizon=8).any()
    for _ in range(4):
        tracker.update(mask)
    dead_lh = tracker.dead("long_horizon", threshold=1e-4, window=6, horizon=8)
    assert dead_lh.tolist() == [False, True, True, True]


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
                dead_window_batches=4,
                dead_horizon_batches=8,
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

    continued = [trainer.step(x)["rec"] for x in batches[15:25]]
    resumed_trainer = Trainer.load_checkpoint(ckpt, device=device)
    assert resumed_trainer.step_idx == 15
    resumed = [resumed_trainer.step(x)["rec"] for x in batches[15:25]]
    assert continued == pytest.approx(resumed, rel=1e-4)
    for a, b in zip(trainer.master.parameters(), resumed_trainer.master.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


def test_lr_schedule_linear_fifth():
    """SASA B.3 schedule: warmup, constant, linear decay over the final
    fifth. Cosine remains the default and is untouched."""
    from block_crosscoder_experiment.trainer import _lr_factor

    f = _lr_factor(train_cfg(total_steps=100, warmup_steps=10, schedule="linear_fifth"))
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(50) == 1.0
    assert f(79) == 1.0
    assert f(90) == pytest.approx(0.5)
    assert f(100) == 0.0

    g = _lr_factor(train_cfg(total_steps=100, warmup_steps=10))  # cosine default
    assert g(55) == pytest.approx(0.5)  # midpoint of the decay span
    with pytest.raises(ValueError, match="schedule"):
        train_cfg(schedule="nonsense")


def test_checkpoint_free_space_floor(device, tmp_path, monkeypatch):
    """save_checkpoint aborts before writing when the D14 floor would be
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_adamw8bit_retraction_ordering():
    """The ordering check pulled forward from 0.9: step -> retract -> recast
    with the actual bitsandbytes 8-bit Adam. External in-place mutation of
    the masters between steps must not destabilize the 8-bit state."""
    pytest.importorskip("bitsandbytes")
    device = torch.device("cuda")
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(
        model,
        train_cfg(total_steps=80, optimizer="auto", forward_dtype="bf16"),
    )
    assert trainer.optimizer_kind == "adamw8bit"
    history = trainer.fit(planted_batches(device))
    assert history[-1]["rec"] < 0.5 * history[0]["rec"]
    assert float(gram_residual(trainer.master.D).max()) < 1e-4
    assert all(torch.isfinite(p).all() for p in trainer.master.parameters())


# -- loss-spike guard --------------------------------------------------------


def _warm_guarded_trainer(device, *, window=10, max_consecutive=3, **overrides):
    """Trainer with the guard armed: enough clean accepted steps to fill
    the trailing-median window."""
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(
        model,
        train_cfg(
            total_steps=200,
            guard=True,
            guard_window=window,
            guard_max_consecutive=max_consecutive,
            **overrides,
        ),
    )
    for x in planted_batches(device, n_batches=window + 5, seed=11):
        trainer.step(x)
    assert trainer.skipped_steps == 0
    return trainer


def test_guard_skips_poisoned_batch_and_recovers(device):
    trainer = _warm_guarded_trainer(device)
    clean = planted_batches(device, n_batches=2, seed=13)
    before = [p.detach().clone() for p in trainer.master.parameters()]
    rec = trainer.step(clean[0] * 1e4)  # grad AND loss anomalous -> skip
    assert rec["skipped"] is True and rec["skip_reason"] == "spike"
    for p, b in zip(trainer.master.parameters(), before):
        assert torch.equal(p.detach(), b), "skipped step must not move params"
    assert trainer.skipped_steps == 1
    assert trainer.guard_events[-1]["reason"] == "spike"
    assert "batch_hash" in trainer.guard_events[-1]
    rec2 = trainer.step(clean[1])  # clean batch accepted, counter resets
    assert "skipped" not in rec2
    assert trainer._guard_consecutive == 0


def test_guard_nonfinite_always_skips(device):
    trainer = _warm_guarded_trainer(device)
    x = planted_batches(device, n_batches=1, seed=17)[0].clone()
    x[0, 0, 0] = float("nan")
    before = [p.detach().clone() for p in trainer.master.parameters()]
    rec = trainer.step(x)
    assert rec["skipped"] is True and rec["skip_reason"] == "nonfinite"
    for p, b in zip(trainer.master.parameters(), before):
        assert torch.equal(p.detach(), b)


def test_guard_consecutive_cap_raises(device):
    trainer = _warm_guarded_trainer(device, max_consecutive=2)
    poison = planted_batches(device, n_batches=1, seed=19)[0] * 1e4
    trainer.step(poison)
    trainer.step(poison)
    with pytest.raises(RuntimeError, match="not stable"):
        trainer.step(poison)


def test_guard_state_survives_checkpoint(device, tmp_path):
    trainer = _warm_guarded_trainer(device)
    trainer.step(planted_batches(device, n_batches=1, seed=23)[0] * 1e4)
    ckpt = tmp_path / "guarded.pt"
    trainer.save_checkpoint(ckpt)
    restored = Trainer.load_checkpoint(ckpt, device=device)
    assert restored.skipped_steps == trainer.skipped_steps
    assert restored._guard_grad_hist == trainer._guard_grad_hist
    assert restored._guard_rec_hist == trainer._guard_rec_hist
    assert restored.guard_events == trainer.guard_events
    assert restored.cfg.guard and restored.cfg.guard_window == 10


def test_guard_off_by_default_unchanged(device):
    """Two trainers, guard off vs on, identical clean data: identical
    parameters — the guard must be a no-op on clean runs."""
    runs = []
    for guard in (False, True):
        model = BlockCrosscoder(CFG).to(device)
        trainer = Trainer(model, train_cfg(total_steps=30, guard=guard))
        trainer.fit(planted_batches(device, n_batches=30, seed=29))
        assert trainer.skipped_steps == 0
        runs.append([p.detach().clone() for p in trainer.master.parameters()])
    for a, b in zip(*runs):
        assert torch.equal(a, b)


# -- E3 AuxK caps ------------------------------------------------------------


def _aux_cfg(**overrides):
    """SASA aux with a tiny window and an impossible threshold: every block
    counts as dead once the window fills, so the cap arithmetic is exercised
    deterministically."""
    return train_cfg(
        aux_variant="sasa",
        dead_threshold=1.0,
        dead_window_batches=3,
        s_aux=8,
        **overrides,
    )


def test_aux_frac_cap_changes_selection(device):
    recs = {}
    for frac in (None, 0.25):
        model = BlockCrosscoder(CFG).to(device)
        trainer = Trainer(model, _aux_cfg(aux_frac_cap=frac))
        for x in planted_batches(device, n_batches=6, seed=31):
            rec = trainer.step(x)
        recs[frac] = rec
    # all G=16 blocks dead: uncapped keep=8, frac-capped keep=ceil(.25*16)=4
    assert "aux" in recs[None] and "aux" in recs[0.25]
    assert recs[None]["aux"] != recs[0.25]["aux"]


def test_aux_ratio_cap_bounds_alpha(device):
    model = BlockCrosscoder(CFG).to(device)
    trainer = Trainer(model, _aux_cfg(aux_ratio_cap=1e-6, log_every=1))
    for x in planted_batches(device, n_batches=6, seed=37):
        rec = trainer.step(x)
    assert rec["alpha_aux_eff"] < 1.0
    assert rec["grad_norm_aux"] > 0
