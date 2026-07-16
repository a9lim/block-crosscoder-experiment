"""Trainer checks: the step->retract->recast ordering, master/forward-copy
sync, dead tracking, AuxK revival (the synthetic dead-encoder revival test
from the pilot spec, exercised early), checkpoint/resume, and threshold
calibration. The CUDA-only test verifies the ordering against the actual
8-bit-Adam implementation (pulled forward from Phase 0.9)."""

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
