"""Winner-scoped eval-split activation and contribution statistics.

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

By default this follows ``data/winner.json`` and evaluates the winner plus its
matched primary-gauge counterpart against the winner's store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.store import StoreReader, Whitener

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
              batch: int, renorm_scalars: torch.Tensor, store: Path) -> None:
    model, report = load_model(root, device)
    cfg = model.cfg
    G, b, S = cfg.n_blocks, cfg.block_dim, cfg.n_sites
    reader = StoreReader(store, "eval")

    fire = torch.zeros(G, dtype=torch.float64, device=device)
    score_sum = torch.zeros(G, dtype=torch.float64, device=device)
    zz = torch.zeros(G, b, b, dtype=torch.float64, device=device)
    coact = torch.zeros(G, G, dtype=torch.float32, device=device)
    l0_hist = torch.zeros(MAX_L0_BIN + 1, dtype=torch.float64, device=device)
    n_tokens = 0

    scale = None
    if report.get("site_renorm_at_load", report.get("site_renorm")):
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
    from .artifacts import analysis_dir, load_winner

    winner = load_winner()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--store", type=Path, default=None,
        help="whitened store root; defaults to the winner store",
    )
    ap.add_argument(
        "--runs", nargs="*", default=None, metavar="NAME=PATH",
        help="override winner + primary (name=path pairs)",
    )
    args = ap.parse_args()
    args.store = args.store or Path(winner["store"])
    args.out = args.out or analysis_dir(winner)
    runs = {
        "winner": str(Path(winner["ckpt"]).parent),
        "primary": winner["counterpart_primary"],
    }
    if args.runs:
        runs = dict(pair.split("=", 1) for pair in args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    whitener = Whitener.load(args.store / "whitener.pt")
    renorm = whitener.site_rms_scalars()
    for name, root in runs.items():
        root = Path(root)
        if not (root / "latest.pt").exists():
            print(f"{name}: missing {root}, skipped")
            continue
        run_stats(name, root, args.device, args.out, args.batch, renorm, args.store)


if __name__ == "__main__":
    main()
