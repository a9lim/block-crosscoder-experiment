"""Eval-split activation statistics for trained 0.9/0.9.5 checkpoints.

Streams the 1M-token eval split (stored order, threshold-mode selection
against each run's calibrated theta) and accumulates, per block:

  - fire_count[G]        tokens where the block clears theta
  - score_sum[G]         sum of selection scores ||z_g|| over fired tokens
  - zz[G,b,b]            second moment of the selected code (fired tokens)
                         -> code anisotropy; the Phase-(-1) packing flag is
                         a strongly split spectrum with ~50/50 share
  - site_energy[G,S]     contribution energy ||D_g^s^T z_g||^2 over fired
                         tokens (the manifold-export "share" statistic),
                         computed at the end as tr(D_g^s D_g^s^T · zz_g) —
                         never materializing the [n,S,G,d] contribution
  - coact[G,G]           block co-activation counts (fp32 accumulator)
  - l0_hist              per-token active-block-count histogram

Runs on jobe (CUDA):

  python scripts/analysis/eval_activation_stats.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.store import StoreReader, Whitener

STORE = Path("/data/stores/bcc-phase09/gemma3_1b_6site_fineweb")
RUNS = {
    "winner": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_lr0.0012",
    "lam0_at_winner": "/data/runs/bcc-phase095/bsc_lam0_seed0_lr0.0012",
    "renorm_lr3e-4": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_renorm",
    "G4096_k32": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_G4096_k32",
    "scalar_winner": "/data/runs/bcc-phase095/scalar_lam0_seed0_lr0.0012",
}
MAX_L0_BIN = 256


def load_model(root: Path, device: str) -> tuple[BlockCrosscoder, dict]:
    ckpt = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    report = json.loads((root / "report.json").read_text())
    cfg = BSCConfig(
        n_blocks=ckpt["model_cfg"]["n_blocks"],
        block_dim=ckpt["model_cfg"]["block_dim"],
        n_sites=ckpt["model_cfg"]["n_sites"],
        d_model=ckpt["model_cfg"]["d_model"],
        k=ckpt["model_cfg"]["k"],
    )
    model = BlockCrosscoder(cfg, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), report


@torch.no_grad()
def run_stats(name: str, root: Path, device: str, out_dir: Path,
              batch: int, renorm_scalars: torch.Tensor) -> None:
    model, report = load_model(root, device)
    cfg = model.cfg
    G, b, S = cfg.n_blocks, cfg.block_dim, cfg.n_sites
    reader = StoreReader(STORE, "eval")

    fire = torch.zeros(G, dtype=torch.float64, device=device)
    score_sum = torch.zeros(G, dtype=torch.float64, device=device)
    zz = torch.zeros(G, b, b, dtype=torch.float64, device=device)
    coact = torch.zeros(G, G, dtype=torch.float32, device=device)
    l0_hist = torch.zeros(MAX_L0_BIN + 1, dtype=torch.float64, device=device)
    n_tokens = 0

    scale = None
    if report.get("site_renorm"):
        scale = renorm_scalars.to(device).view(1, S, 1)

    for x in reader.sequential_batches(batch):
        x = x.to(device=device, dtype=torch.float32)
        if scale is not None:
            x = x * scale
        z = model.encode(x)
        p = model.scores(z)
        mask = p > model.theta
        zsel = z * mask.unsqueeze(-1)

        fm = mask.float()
        fire += fm.sum(0).double()
        score_sum += (p * fm).sum(0).double()
        zz += torch.einsum("ngb,ngc->gbc", zsel, zsel).double()
        coact += fm.T @ fm
        l0 = mask.sum(1).clamp(max=MAX_L0_BIN)
        l0_hist += torch.bincount(l0, minlength=MAX_L0_BIN + 1).double()
        n_tokens += x.shape[0]

    # Contribution energy from the accumulated second moment:
    # sum_n ||D_g^s^T z_g||^2 = tr(D_g^s D_g^s^T · sum_n z z^T).
    M = torch.einsum("sgbd,sgcd->sgbc", model.D.double(), model.D.double())
    site_energy = torch.einsum("gbc,sgbc->gs", zz, M)

    np.savez_compressed(
        out_dir / f"evalstats_{name}.npz",
        fire_count=fire.cpu().numpy(),
        score_sum=score_sum.cpu().numpy(),
        zz=zz.cpu().numpy().astype(np.float32),
        site_energy=site_energy.cpu().numpy().astype(np.float32),
        coact=coact.cpu().numpy(),
        l0_hist=l0_hist.cpu().numpy(),
        n_tokens=np.int64(n_tokens),
        theta=np.float32(float(model.theta)),
        meta=json.dumps({"run": str(root), "model_cfg": ckpt_cfg_json(model)}),
    )
    freq = (fire / n_tokens).cpu()
    print(
        f"{name}: {n_tokens:,} tokens, mean L0 "
        f"{float((fire.sum() / n_tokens)):.2f}, dead(<1e-6) "
        f"{int((freq < 1e-6).sum())}/{G}",
        flush=True,
    )


def ckpt_cfg_json(model: BlockCrosscoder) -> dict:
    c = model.cfg
    return {
        "n_blocks": c.n_blocks, "block_dim": c.block_dim,
        "n_sites": c.n_sites, "d_model": c.d_model, "k": c.k,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=Path("/data/runs/bcc-analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    whitener = Whitener.load(STORE / "whitener.pt")
    renorm = whitener.site_rms_scalars()
    for name, root in RUNS.items():
        root = Path(root)
        if not (root / "latest.pt").exists():
            print(f"{name}: missing {root}, skipped")
            continue
        run_stats(name, root, args.device, args.out, args.batch, renorm)


if __name__ == "__main__":
    main()
