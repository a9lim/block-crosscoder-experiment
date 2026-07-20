"""Run the Phase-0.5 paper-faithful synthetic bridge experiments.

The Fel bridge uses one site, exact-k factor superposition, signed block
codes, the paper's selector for each method, no AuxK, and separate
Grassmannian, Vanilla, Group-Lasso, and current-hybrid learners. It is
intentionally kept distinct from the
multi-site Bernoulli/noisy Phase-minus-1 BSC stress battery.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from block_crosscoder_experiment.metrics import evaluate_recovery
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder
from block_crosscoder_experiment.synthetic import (
    BlockSpec,
    ExactKPlantedModel,
)
from block_crosscoder_experiment.trainer import TrainConfig, Trainer


def fel_specs(n_factors: int, k: int, block_dim: int) -> list[BlockSpec]:
    specs = []
    for g in range(n_factors):
        rank = 1 + g % min(4, block_dim)
        geometry = "shell" if rank >= 2 and g % 2 else "gaussian"
        # Equal expected factor energy across intrinsic ranks.
        specs.append(BlockSpec(
            rank=rank, frequency=k / n_factors, geometry=geometry,
            scale=1 / math.sqrt(rank),
        ))
    return specs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-factors", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--block-dim", type=int, default=4)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--blocks", type=int, default=256)
    p.add_argument("--examples", type=int, default=300_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--group-lambda", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    truth = ExactKPlantedModel(
        fel_specs(args.n_factors, args.k, args.block_dim), n_sites=1,
        d_model=args.d_model, block_dim=args.block_dim,
        noise_std=0.0, seed=args.seed, active_per_sample=args.k,
    )
    steps = math.ceil(args.examples / args.batch_size)
    recipes = {
        "fel_grassmannian": dict(
            encoder_mode="tied", encoder_bias=False, decoder_constraint="gram",
            selection="token_topk", code_activation="signed", regularizer="none",
        ),
        "fel_vanilla": dict(
            encoder_mode="untied", encoder_bias=True,
            decoder_constraint="frobenius", selection="token_topk",
            code_activation="signed", regularizer="none",
        ),
        "fel_group_lasso": dict(
            encoder_mode="untied", encoder_bias=True,
            decoder_constraint="free", selection="dense",
            code_activation="group_soft_threshold", regularizer="group_l21",
            lambda_regularizer=args.group_lambda,
        ),
        "bsc_hybrid": dict(
            encoder_mode="untied", encoder_bias=False, decoder_constraint="gram",
            selection="token_topk", code_activation="signed", regularizer="none",
        ),
    }
    reports = {}
    for i, (name, variant) in enumerate(recipes.items()):
        cfg = BSCConfig(
            n_blocks=args.blocks, block_dim=args.block_dim, n_sites=1,
            d_model=args.d_model, k=float(args.k), seed=args.seed + i,
            **variant,
        )
        model = BlockCrosscoder(cfg, device=args.device)
        first = truth.sample(args.batch_size, seed=10_000).x.to(args.device)
        model.calibrate_encoder_scale_(first)
        trainer = Trainer(model, TrainConfig(
            total_steps=steps, lr=args.lr, warmup_steps=min(100, steps // 10),
            forward_dtype="fp32", optimizer="adamw", aux_variant="none",
            log_every=max(1, steps // 10),
        ))
        trainer.fit(truth.batches(args.batch_size, steps, seed=20_000))
        report = evaluate_recovery(
            truth, trainer.master, n_eval=100_000, seed=30_000,
            min_active=100, min_joint=30,
        )
        recovered = [
            b for b in report.blocks
            if b.matched is not None and b.overlap >= 0.9 and b.code_r2 >= 0.8
        ]
        reports[name] = {
            "model_cfg": asdict(cfg),
            "recovered_fraction": len(recovered) / len(report.blocks),
            "recovery": report.to_dict(),
        }

    payload = {
        "protocol": {
            "paper": "Fel et al. 2026, arXiv:2606.25234",
            "n_sites": 1,
            "n_factors": args.n_factors,
            "active_per_sample": args.k,
            "noise_std": 0.0,
            "selection": "per-token block TopK",
            "aux": "none",
            "examples": args.examples,
            "batch_size": args.batch_size,
            "steps": steps,
            "lr": args.lr,
            "note": "support/training-recipe reproduction with a local rank/shell zoo",
        },
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
