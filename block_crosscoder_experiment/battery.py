"""The Phase -1 scenario battery (design v2.2, Phase -1).

Seven scenarios, each mandated by the spec:

  core                  shared blocks with cross-site frame rotation,
                        ranks 1..b, hollow + thickened shells (P18)
  lambda_veto           flat-profile shared blocks under the quantitative
                        lambda-veto; emits the admissible lambda set (or
                        the documented lambda=0 fallback)
  decoys                site-specific decoys, scored as expected
                        site-exclusive recoveries
  bundle_null           the weakened null (D11): perfectly co-active
                        scalars — bundling is legitimate; hallucinating
                        *hollow shell* geometry on them is the failure
  frequency_ladder      recovery vs planted frequency (R24) — the curve
                        Phase 1's rare-block claims are read against;
                        report-only
  rotation_equivariance paired seeds from O(b)-rotated inits (R8): Adam is
                        not equivariant to the residual gauge; material
                        divergence moves decoders off coordinatewise Adam
  auxk_comparison       SASA vs long-horizon vs Fel runner-up on a
                        dead-prone config (P8); report-only, feeds the
                        0.9 calibration

Scale constants come from the 2026-07-16 harness calibration: d well above
total planted latent dims (superposition crosstalk, not the optimizer, was
the dominant failure mode at d=32), selection budget k*B matched to
E[active blocks], learner capacity ~1.5x planted count (spare capacity
tiles manifolds into arcs, tight capacity mixes features). Per-seed basin
variance (capture vs tiling) is real; the battery reports distributions
over seeds, never single runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .metrics import RecoveryReport, evaluate_recovery
from .model import BlockCrosscoder, BSCConfig
from .synthetic import BlockSpec, PlantedModel
from .trainer import TrainConfig, Trainer

__all__ = ["BatteryConfig", "run_scenario", "run_battery", "SCENARIOS"]


@dataclass
class BatteryConfig:
    """Predeclared thresholds and scales — recorded verbatim in the output,
    per the spec's 'set when the harness lands; recorded in its config'."""

    steps: int = 3000
    batch_size: int = 1024
    n_eval: int = 32768
    seeds: tuple[int, ...] = (0, 1)
    n_sites: int = 4
    d_model: int = 128
    block_dim: int = 4
    lr: float = 3e-3
    lambda_grid: tuple[float, ...] = (0.0, 3e-4, 1e-3, 3e-3)
    # Gates
    overlap_pass: float = 0.9
    code_r2_pass: float = 0.8
    recovered_fraction_pass: float = 0.75  # per-seed fraction of planted blocks
    share_tol_floor: float = 0.02  # lambda-veto tolerance floor
    veto_multiplier: float = 2.0  # tolerance = mult x (lambda=0 share error)
    veto_retention: float = 0.85  # overlap retention vs lambda=0
    norm_cv_shell_max: float = 0.1  # below: hollow-shell signature
    norm_cv_full_min: float = 0.2  # above: full-support signature
    rotation_spectrum_tol: float = 0.1  # relative L2 on sorted spectra
    rotation_span_pass: float = 0.9  # cross-learner span overlap


def _recovered(rec, bc: BatteryConfig) -> bool:
    return (
        rec.matched is not None
        and rec.overlap > bc.overlap_pass
        and rec.code_r2 > bc.code_r2_pass
    )


def rotate_blocks_(model: BlockCrosscoder, seed: int) -> None:
    """Apply an independent random O(b) to every block's decoder AND
    encoder (preserving the tie) — the R8 rotated-init transform."""
    gen = torch.Generator().manual_seed(seed)
    b = model.cfg.block_dim
    with torch.no_grad():
        for g in range(model.cfg.n_blocks):
            q, r = torch.linalg.qr(torch.randn(b, b, generator=gen))
            R = (q * torch.sign(torch.diagonal(r))).to(model.D.device)
            model.D[:, g] = torch.einsum("bc,scd->sbd", R, model.D[:, g])
            model.E[:, g] = torch.einsum("bc,scd->sbd", R, model.E[:, g])


def run_one_full(
    specs: list[BlockSpec],
    bc: BatteryConfig,
    *,
    n_blocks: int,
    k: int,
    lam: float = 0.0,
    learner_seed: int = 0,
    data_seed: int = 0,
    truth_seed: int = 5,
    device: str | torch.device = "cpu",
    aux_variant: str = "sasa",
    rotate_init_seed: int | None = None,
    min_active: int = 50,
    n_eval: int | None = None,
) -> tuple[RecoveryReport, Trainer, PlantedModel]:
    truth = PlantedModel(
        specs,
        n_sites=bc.n_sites,
        d_model=bc.d_model,
        block_dim=bc.block_dim,
        noise_std=0.02,
        seed=truth_seed,
    )
    cfg = BSCConfig(
        n_blocks=n_blocks,
        block_dim=bc.block_dim,
        n_sites=bc.n_sites,
        d_model=bc.d_model,
        k=k,
        lambda_rank=lam,
        seed=learner_seed,
    )
    learner = BlockCrosscoder(cfg).to(device)
    if rotate_init_seed is not None:
        rotate_blocks_(learner, rotate_init_seed)
    learner.calibrate_encoder_scale_(
        truth.sample(4096, seed=11).x.to(device)
    )
    trainer = Trainer(
        learner,
        TrainConfig(
            total_steps=bc.steps,
            lr=bc.lr,
            warmup_steps=max(20, bc.steps // 100),
            forward_dtype="fp32",
            optimizer="auto",
            aux_variant=aux_variant,
            s_aux=4,
            dead_window_batches=10,
            log_every=max(100, bc.steps // 10),
        ),
    )
    trainer.fit(truth.batches(bc.batch_size, bc.steps, seed=1000 + data_seed))
    report = evaluate_recovery(
        truth,
        learner,
        n_eval=n_eval or bc.n_eval,
        seed=99,
        min_active=min_active,
    )
    return report, trainer, truth


def run_one(specs: list[BlockSpec], bc: BatteryConfig, **kwargs) -> RecoveryReport:
    report, _, _ = run_one_full(specs, bc, **kwargs)
    return report


# -- scenario zoos ------------------------------------------------------------


def core_zoo() -> tuple[list[BlockSpec], int, int]:
    """Six shared flat-profile blocks, ranks 1..4, hollow + thickened
    shells, energy-balanced (E||z||^2 = 4), E[active] = 1.0 -> k=1."""
    f = 1.0 / 6.0
    specs = [
        BlockSpec(rank=1, frequency=f, scale=2.0),
        BlockSpec(rank=2, frequency=f, spectrum=(2.4, 1.6)),
        BlockSpec(rank=2, frequency=f, geometry="shell", scale=2.0),
        BlockSpec(rank=2, frequency=f, geometry="shell", radial_spread=0.3, scale=2.0),
        BlockSpec(rank=3, frequency=f, spectrum=(2.0, 1.2, 0.8)),
        BlockSpec(rank=4, frequency=f, spectrum=(1.6, 1.2, 0.8, 0.6)),
    ]
    return specs, 10, 1  # specs, G_learner, k


def decoy_zoo(n_sites: int) -> tuple[list[BlockSpec], int, int]:
    f = 1.0 / 6.0
    one_hot = lambda s: tuple(1.0 if i == s else 0.0 for i in range(n_sites))
    specs = [
        BlockSpec(rank=1, frequency=f, scale=2.0),
        BlockSpec(rank=2, frequency=f, spectrum=(2.4, 1.6)),
        BlockSpec(rank=3, frequency=f, spectrum=(2.0, 1.2, 0.8)),
        BlockSpec(rank=2, frequency=f, spectrum=(2.4, 1.6), depth_profile=one_hot(0)),
        BlockSpec(rank=2, frequency=f, spectrum=(2.4, 1.6), depth_profile=one_hot(1)),
        BlockSpec(rank=2, frequency=f, spectrum=(2.4, 1.6), depth_profile=one_hot(3)),
    ]
    return specs, 10, 1


def bundle_zoo() -> tuple[list[BlockSpec], int, int]:
    f = 0.25
    specs = [
        # The weakened null: four perfectly co-active rank-1 scalars.
        *[
            BlockSpec(rank=1, frequency=f, scale=2.0, gate_group=0, gate_coupling=1.0)
            for _ in range(4)
        ],
        # Positive contrast: a genuine hollow ring and an honest scalar.
        BlockSpec(rank=2, frequency=f, geometry="shell", scale=2.0),
        BlockSpec(rank=1, frequency=f, scale=2.0),
    ]
    return specs, 8, 1


def frequency_zoo() -> tuple[list[BlockSpec], int, int]:
    ladder = (0.1, 0.03, 0.01, 0.003, 0.001)
    specs = [
        BlockSpec(rank=2, frequency=fr, spectrum=(2.4, 1.6)) for fr in ladder
    ] + [
        BlockSpec(rank=1, frequency=0.3, scale=2.0),
        BlockSpec(rank=1, frequency=0.3, scale=2.0),
    ]
    return specs, 10, 1


def auxk_zoo() -> tuple[list[BlockSpec], int, int]:
    specs = [
        BlockSpec(rank=2, frequency=0.2, spectrum=(2.4, 1.6)),
        BlockSpec(rank=1, frequency=0.2, scale=2.0),
        BlockSpec(rank=3, frequency=0.2, spectrum=(2.0, 1.2, 0.8)),
        BlockSpec(rank=2, frequency=0.005, spectrum=(2.4, 1.6)),
        BlockSpec(rank=1, frequency=0.005, scale=2.0),
        BlockSpec(rank=2, frequency=0.005, geometry="shell", scale=2.0),
    ]
    return specs, 16, 1  # oversized learner: dead-prone by design


# -- scenarios ----------------------------------------------------------------


def scenario_core(bc: BatteryConfig, device) -> dict:
    specs, G, k = core_zoo()
    runs = []
    for seed in bc.seeds:
        rep = run_one(
            specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed, device=device
        )
        frac = sum(_recovered(r, bc) for r in rep.blocks) / len(rep.blocks)
        runs.append({"seed": seed, "recovered_fraction": frac, "report": rep.to_dict()})
    gate = all(r["recovered_fraction"] >= bc.recovered_fraction_pass for r in runs)
    return {"runs": runs, "gate_pass": gate}


def scenario_lambda_veto(bc: BatteryConfig, device) -> dict:
    specs, G, k = core_zoo()  # all flat-profile by construction
    grid: dict[float, list[RecoveryReport]] = {}
    for lam in bc.lambda_grid:
        grid[lam] = [
            run_one(
                specs, bc, n_blocks=G, k=k, lam=lam,
                learner_seed=seed, data_seed=seed, device=device,
            )
            for seed in bc.seeds
        ]

    def mean_share_error(reps):
        vals = [r.share_error for rep in reps for r in rep.blocks if r.matched is not None]
        return sum(vals) / max(1, len(vals))

    def mean_overlap(reps):
        vals = [r.overlap for rep in reps for r in rep.blocks if r.matched is not None]
        return sum(vals) / max(1, len(vals))

    base_share = mean_share_error(grid[0.0])
    base_overlap = mean_overlap(grid[0.0])
    tolerance = max(bc.share_tol_floor, bc.veto_multiplier * base_share)
    rows, admissible = [], []
    for lam in bc.lambda_grid:
        share, ov = mean_share_error(grid[lam]), mean_overlap(grid[lam])
        ok = share <= tolerance and ov >= bc.veto_retention * base_overlap
        rows.append(
            {"lambda": lam, "share_error": share, "overlap": ov, "admissible": ok}
        )
        if ok and lam > 0:
            admissible.append(lam)
    return {
        "tolerance": tolerance,
        "base_share_error": base_share,
        "base_overlap": base_overlap,
        "rows": rows,
        "admissible_nonzero": admissible,
        "lambda_zero_fallback": not admissible,
        # The gate is a nonempty admissible set OR the documented fallback;
        # either way the veto machinery itself must have run on a sane base.
        "gate_pass": base_overlap >= bc.overlap_pass * 0.9,
        "runs": {str(lam): [r.to_dict() for r in reps] for lam, reps in grid.items()},
    }


def scenario_decoys(bc: BatteryConfig, device) -> dict:
    specs, G, k = decoy_zoo(bc.n_sites)
    runs, gates = [], []
    for seed in bc.seeds:
        rep = run_one(
            specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed, device=device
        )
        decoys = [r for r, s in zip(rep.blocks, specs) if s.depth_profile is not None]
        shared = [r for r, s in zip(rep.blocks, specs) if s.depth_profile is None]
        # Decoys must come back site-exclusive (share error is measured
        # against the planted one-hot profile) — an expected recovery, not
        # a nothing-recovered null.
        decoy_ok = all(
            r.matched is not None and r.share_error < 0.15 and r.overlap > bc.overlap_pass
            for r in decoys
        )
        shared_ok = (
            sum(_recovered(r, bc) for r in shared) / len(shared)
            >= bc.recovered_fraction_pass
        )
        gates.append(decoy_ok and shared_ok)
        runs.append(
            {"seed": seed, "decoy_ok": decoy_ok, "shared_ok": shared_ok,
             "report": rep.to_dict()}
        )
    return {"runs": runs, "gate_pass": all(gates)}


def scenario_bundle_null(bc: BatteryConfig, device) -> dict:
    specs, G, k = bundle_zoo()
    runs, gates = [], []
    for seed in bc.seeds:
        rep = run_one(
            specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed, device=device
        )
        bundle = [r for r, s in zip(rep.blocks, specs) if s.gate_group == 0]
        ring = rep.blocks[4]
        # Bundling is legitimate (D11); the failure would be a hollow-shell
        # signature on bundle-matched blocks. NaN cv (nothing matched) is
        # not a hallucination.
        no_hallucination = all(
            not (r.norm_cv_learned == r.norm_cv_learned  # not NaN
                 and r.norm_cv_learned < bc.norm_cv_shell_max)
            for r in bundle
        )
        ring_detected = (
            ring.matched is not None
            and ring.norm_cv_learned < bc.norm_cv_shell_max
        )
        gates.append(no_hallucination and ring_detected)
        runs.append(
            {"seed": seed, "no_hallucination": no_hallucination,
             "ring_detected": ring_detected, "report": rep.to_dict()}
        )
    return {"runs": runs, "gate_pass": all(gates)}


def scenario_frequency_ladder(bc: BatteryConfig, device) -> dict:
    specs, G, k = frequency_zoo()
    runs = []
    for seed in bc.seeds:
        rep = run_one(
            specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed,
            device=device, n_eval=4 * bc.n_eval, min_active=20,
        )
        curve = [
            {
                "frequency": s.frequency,
                "matched": r.matched is not None,
                "overlap": r.overlap,
                "code_r2": r.code_r2,
                "n_active_planted": r.n_active_planted,
            }
            for r, s in zip(rep.blocks, specs)
            if s.rank == 2
        ]
        runs.append({"seed": seed, "curve": curve, "report": rep.to_dict()})
    return {"runs": runs, "gate_pass": None}  # report-only: the R24 calibration


def scenario_rotation_equivariance(bc: BatteryConfig, device) -> dict:
    from .metrics import block_site_spans, subspace_overlap

    specs, G, k = core_zoo()
    runs, gates = [], []
    for seed in bc.seeds:
        rep_a, tr_a, truth = run_one_full(
            specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed,
            device=device,
        )
        rep_b, tr_b, _ = run_one_full(
            specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed,
            device=device, rotate_init_seed=500 + seed,
        )
        batch = truth.sample(bc.n_eval, seed=99)
        xa = batch.x.to(device)
        with torch.no_grad():
            out_a = tr_a.master(xa)
            out_b = tr_b.master(xa)
        pairs = []
        for ra, rb in zip(rep_a.blocks, rep_b.blocks):
            if ra.matched is None or rb.matched is None:
                continue
            za = out_a.z_selected[out_a.mask[:, ra.matched], ra.matched].cpu()
            zb = out_b.z_selected[out_b.mask[:, rb.matched], rb.matched].cpu()
            if za.shape[0] < 50 or zb.shape[0] < 50:
                continue
            spans_a, spec_a = block_site_spans(
                tr_a.master.D.detach().float().cpu()[:, ra.matched], za
            )
            spans_b, spec_b = block_site_spans(
                tr_b.master.D.detach().float().cpu()[:, rb.matched], zb
            )
            r = ra.rank
            span_agree = sum(
                subspace_overlap(spans_a[s, :, :r], spans_b[s, :, :r])[0]
                for s in range(bc.n_sites)
            ) / bc.n_sites
            sa, sb = spec_a.sort(descending=True).values, spec_b.sort(descending=True).values
            rel = float((sa - sb).norm() / sa.norm().clamp_min(1e-12))
            pairs.append(
                {"planted": ra.planted, "span_agreement": span_agree,
                 "spectrum_rel_diff": rel}
            )
        ok = bool(pairs) and all(
            p["span_agreement"] > bc.rotation_span_pass
            and p["spectrum_rel_diff"] < bc.rotation_spectrum_tol
            for p in pairs
        )
        gates.append(ok)
        runs.append({"seed": seed, "pairs": pairs, "gate": ok})
    # Material divergence => move decoders off coordinatewise Adam (R8);
    # the battery flags it rather than silently passing.
    return {"runs": runs, "gate_pass": all(gates)}


def scenario_auxk_comparison(bc: BatteryConfig, device) -> dict:
    specs, G, k = auxk_zoo()
    out = {}
    for variant in ("sasa", "long_horizon", "fel"):
        rows = []
        for seed in bc.seeds:
            rep = run_one(
                specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed,
                device=device, aux_variant=variant,
                n_eval=4 * bc.n_eval, min_active=20,
            )
            rare = [
                {"planted": r.planted, "matched": r.matched is not None,
                 "overlap": r.overlap}
                for r, s in zip(rep.blocks, specs)
                if s.frequency < 0.01
            ]
            rows.append(
                {"seed": seed, "n_dead": rep.n_learned_dead, "rare": rare,
                 "report": rep.to_dict()}
            )
        out[variant] = rows
    return {"variants": out, "gate_pass": None}  # report-only: feeds 0.9


SCENARIOS = {
    "core": scenario_core,
    "lambda_veto": scenario_lambda_veto,
    "decoys": scenario_decoys,
    "bundle_null": scenario_bundle_null,
    "frequency_ladder": scenario_frequency_ladder,
    "rotation_equivariance": scenario_rotation_equivariance,
    "auxk_comparison": scenario_auxk_comparison,
}


def run_scenario(name: str, bc: BatteryConfig, device) -> dict:
    result = SCENARIOS[name](bc, device)
    result["scenario"] = name
    return result


def run_battery(
    bc: BatteryConfig,
    *,
    device: str | torch.device = "cpu",
    scenarios: list[str] | None = None,
    out_path: str | Path | None = None,
) -> dict:
    names = scenarios or list(SCENARIOS)
    results = {}
    for name in names:
        print(f"[battery] {name} ...", flush=True)
        results[name] = run_scenario(name, bc, device)
        gp = results[name]["gate_pass"]
        print(f"[battery] {name}: gate={'REPORT-ONLY' if gp is None else ('PASS' if gp else 'FAIL')}")
    hard = [r["gate_pass"] for r in results.values() if r["gate_pass"] is not None]
    payload = {
        "battery_config": asdict(bc),
        "results": results,
        "all_hard_gates_pass": all(hard) if hard else None,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1))
        print(f"[battery] wrote {out_path}")
    return payload
