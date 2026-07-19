"""E3 validation, second axis (runbook-phase099 tranche 1): revival retention.

The cascade-suppression axis (r4/r5/r6 at 4b lr 6e-4) shows whether a cap
prevents the AuxK spike amplifier; this script measures what the same cap
costs on the mechanism AuxK exists for — reviving dead/rare blocks — on
Phase -1 synthetic ground truth (auxk_zoo: 3 common + 3 rare f=0.005
blocks, G=16 oversized/dead-prone, SASA C.1, the Phase -1 operating
point: 10k steps x batch 1024 unless overridden).

Arms: control (uncapped SASA), frac cap 0.5, ratio cap 1.0, alpha 0.5
(the third candidate: a static global attenuation, what the caps are
trying to beat).

Gate, per cap arm x seed: every rare planted block kept by the control
arm (matched, overlap > overlap_pass) is also kept by the capped arm,
and n_dead exceeds control's by at most `--dead-slack` (default 2 of
G=16). A cap that suppresses cascades by killing revival fails here.

Reference (Phase -1 battery run 6): uncapped SASA keeps 12/12 rare
features with 1-4 dead of 16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from block_crosscoder_experiment.battery import BatteryConfig, auxk_zoo, budget_k, run_one_full

ARMS: dict[str, dict] = {
    "control": {},
    "fcap0.5": {"aux_frac_cap": 0.5},
    "rcap1.0": {"aux_ratio_cap": 1.0},
    "alpha0.5": {"alpha_aux": 0.5},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    # None = BatteryConfig operating point (never shadow its defaults — the
    # 3k-step CLI-shadowing incident, run_phase_minus1.py).
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dead-slack", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("data/e3_revival_report.json"))
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    overrides = {
        k: v
        for k, v in (("steps", args.steps), ("batch_size", args.batch_size))
        if v is not None
    }
    bc = BatteryConfig(seeds=tuple(args.seeds), **overrides)
    specs, G = auxk_zoo()
    k = budget_k(specs, bc)
    rare_idx = [i for i, s in enumerate(specs) if s.frequency < 0.01]

    results: dict[str, list[dict]] = {}
    for arm, kwargs in ARMS.items():
        rows = []
        for seed in bc.seeds:
            rep, trainer, _ = run_one_full(
                specs, bc, n_blocks=G, k=k, learner_seed=seed, data_seed=seed,
                device=device, aux_variant="sasa", n_eval=4 * bc.n_eval,
                min_active=20, **kwargs,
            )
            rare = [
                {
                    "planted": r.planted,
                    "matched": r.matched is not None,
                    "overlap": r.overlap,
                    "code_r2": r.code_r2,
                    "kept": r.matched is not None and r.overlap > bc.overlap_pass,
                }
                for i, r in enumerate(rep.blocks)
                if i in rare_idx
            ]
            # Cap engagement, from the log-step samples (log_every intervals).
            alphas = [h["alpha_aux_eff"] for h in trainer.history if "alpha_aux_eff" in h]
            s_effs = [h["s_aux_eff"] for h in trainer.history if "s_aux_eff" in h]
            rows.append(
                {
                    "seed": seed,
                    "n_dead": rep.n_learned_dead,
                    "rare": rare,
                    "rare_kept": sum(r["kept"] for r in rare),
                    "engagement": {
                        "alpha_eff_min": min(alphas) if alphas else None,
                        "alpha_eff_capped_frac": (
                            sum(a < 1.0 for a in alphas) / len(alphas) if alphas else None
                        ),
                        "s_aux_eff_min": min(s_effs) if s_effs else None,
                    },
                    "report": rep.to_dict(),
                }
            )
            print(
                f"[e3] {arm} seed {seed}: rare kept "
                f"{rows[-1]['rare_kept']}/{len(rare_idx)}, dead {rep.n_learned_dead}/{G}",
                flush=True,
            )
        results[arm] = rows

    gates = {}
    for arm in ARMS:
        if arm == "control":
            continue
        ok = True
        for ctl, capped in zip(results["control"], results[arm]):
            kept_ctl = {r["planted"] for r in ctl["rare"] if r["kept"]}
            kept_arm = {r["planted"] for r in capped["rare"] if r["kept"]}
            ok &= kept_ctl <= kept_arm
            ok &= capped["n_dead"] <= ctl["n_dead"] + args.dead_slack
        gates[arm] = bool(ok)

    payload = {
        "battery_config": bc.__dict__ | {"seeds": list(bc.seeds)},
        "dead_slack": args.dead_slack,
        "results": results,
        "gate_retention": gates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[e3] gates: {gates}", flush=True)


if __name__ == "__main__":
    main()
