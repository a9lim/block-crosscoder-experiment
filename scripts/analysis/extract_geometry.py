"""Weight-space geometry extraction from trained 0.9/0.9.5 checkpoints.

Interim artifact analysis (2026-07-17, pre-Phase-1): the rehearsal and
calibration runs were judged on FVU/mortality only; this pulls the learned
geometry out of the decoder/encoder stacks so it can be visualized.

Per run (BSC [S,G,b,d] and scalar b=1 alike):
  - share[g,s]        per-site weight-energy share ||D_g^s||_F^2 / b
                      (Gram constraint makes rows of share sum to 1)
  - svals[s,g,:]      per-site decoder singular values (spectrum-budget use)
  - pair_cos[g,p,:]   principal cosines between row-spans of D_g^s, D_g^s'
                      for all site pairs p, plus a shuffled-block null
  - stacked_svals[g]  singular values of the cross-site stack [S*b, d] —
                      the block's total spectral footprint across depth
                      (b if one shared subspace rides all sites, S*b if
                      every site uses a fresh subspace)
  - encdec_cos[s,g,:] principal cosines span(E_g^s) vs span(D_g^s)
  - enc_share[g,s]    encoder-side energy share (no Gram constraint)
  - gram_residual[g]  ||sum_s D_g^s D_g^s^T - I_b||_F  (sanity)
  - c_norm[s], theta, model_cfg, eval FVU (copied from report.json)

Outputs one .npz per run under --out. Runs on jobe:

  python scripts/analysis/extract_geometry.py --device cuda
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

RUNS = {
    # 0.9.5 calibration arms (4M tokens each)
    "winner": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_lr0.0012",
    "winner_seed1": "/data/runs/bcc-phase095/bsc_lam0.001_seed1_lr0.0012",
    "lam0_at_winner": "/data/runs/bcc-phase095/bsc_lam0_seed0_lr0.0012",
    "base_lr3e-4": "/data/runs/bcc-phase095/bsc_lam0.001_seed0",
    "renorm_lr3e-4": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_renorm",
    "G4096_k32": "/data/runs/bcc-phase095/bsc_lam0.001_seed0_G4096_k32",
    "scalar_winner": "/data/runs/bcc-phase095/scalar_lam0_seed0_lr0.0012",
    "scalar_base": "/data/runs/bcc-phase095/scalar_lam0_seed0",
    # 0.9 rehearsal lambda ladder (identical trainer, lr default)
    "p09_lam3e-4": "/data/runs/bcc-phase09/bsc_lam0.0003_seed0",
}


def orthobasis(D: torch.Tensor) -> torch.Tensor:
    """[N, b, d] rows -> [N, d, b] orthonormal column bases (batched QR)."""
    q, _ = torch.linalg.qr(D.transpose(1, 2))
    return q


def pair_principal_cos(Q: torch.Tensor, sites: int) -> tuple[np.ndarray, list]:
    """Q: [S, G, d, b] bases -> [G, P, b] principal cosines per site pair."""
    pairs = list(itertools.combinations(range(sites), 2))
    out = []
    for a, b_ in pairs:
        cross = Q[a].transpose(1, 2) @ Q[b_]  # [G, b, b]
        out.append(torch.linalg.svdvals(cross).clamp(0, 1))
    return torch.stack(out, dim=1).cpu().numpy(), pairs


def extract(name: str, root: Path, device: str, out_dir: Path) -> None:
    ckpt = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    report = json.loads((root / "report.json").read_text())
    cfg = ckpt["model_cfg"]
    S, G, b, d = cfg["n_sites"], cfg["n_blocks"], cfg["block_dim"], cfg["d_model"]
    D = ckpt["model"]["D"].to(device=device, dtype=torch.float32)  # [S,G,b,d]
    E = ckpt["model"]["E"].to(device=device, dtype=torch.float32)
    c = ckpt["model"]["c"].float()
    theta = float(ckpt["model"]["theta"])

    # Energy shares and Gram sanity.
    energy = D.pow(2).sum(dim=(2, 3)).T  # [G, S]
    share = (energy / b).cpu().numpy()
    gram = torch.einsum("sgbd,sgcd->gbc", D, D)
    eye = torch.eye(b, device=device).expand(G, b, b)
    gram_residual = (gram - eye).flatten(1).norm(dim=1).cpu().numpy()

    enc_energy = E.pow(2).sum(dim=(2, 3)).T  # [G, S]
    enc_share = (enc_energy / enc_energy.sum(1, keepdim=True)).cpu().numpy()

    # Per-site spectra.
    svals = torch.linalg.svdvals(D.reshape(S * G, b, d)).reshape(S, G, b)

    # Cross-site frame rotation + shuffled-block null.
    Q = orthobasis(D.reshape(S * G, b, d)).reshape(S, G, d, b)
    pair_cos, pairs = pair_principal_cos(Q, S)
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(G, generator=gen).to(device)
    Qn = Q.clone()
    for s in range(1, S):  # decorrelate blocks across sites, keep marginals
        perm = perm[torch.randperm(G, generator=gen).to(device)]
        Qn[s] = Q[s, perm]
    null_cos, _ = pair_principal_cos(Qn, S)

    # Cross-site stacked spectrum.
    stacked = D.permute(1, 0, 2, 3).reshape(G, S * b, d)
    stacked_svals = torch.linalg.svdvals(stacked).cpu().numpy()

    # Encoder-decoder span alignment.
    Qe = orthobasis(E.reshape(S * G, b, d)).reshape(S, G, d, b)
    encdec = torch.linalg.svdvals(
        Q.reshape(S * G, d, b).transpose(1, 2) @ Qe.reshape(S * G, d, b)
    ).reshape(S, G, b).clamp(0, 1)

    np.savez_compressed(
        out_dir / f"geometry_{name}.npz",
        share=share.astype(np.float32),
        enc_share=enc_share.astype(np.float32),
        svals=svals.cpu().numpy().astype(np.float32),
        pair_cos=pair_cos.astype(np.float16),
        null_pair_cos=null_cos.astype(np.float16),
        pairs=np.array(pairs, dtype=np.int32),
        stacked_svals=stacked_svals.astype(np.float32),
        encdec_cos=encdec.cpu().numpy().astype(np.float16),
        gram_residual=gram_residual.astype(np.float32),
        c_norm=c.norm(dim=1).numpy().astype(np.float32),
        theta=np.float32(theta),
        meta=json.dumps(
            {
                "run": str(root),
                "model_cfg": cfg,
                "arm": report.get("arm"),
                "lam": report.get("lam"),
                "lr": report.get("lr"),
                "schedule": report.get("schedule"),
                "site_renorm": report.get("site_renorm"),
                "fvu_pooled": report["eval"]["topk"]["fvu_pooled"],
                "fvu_per_site": report["eval"]["topk"]["fvu_per_site"],
                "sites": [7, 10, 13, 17, 20, 22],
            }
        ),
    )
    print(
        f"{name}: G={G} b={b} gram_res max {gram_residual.max():.2e} "
        f"share depth-argmax hist {np.bincount(share.argmax(1), minlength=S)}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/data/runs/bcc-analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, root in RUNS.items():
        root = Path(root)
        if not (root / "latest.pt").exists():
            print(f"{name}: missing {root}, skipped")
            continue
        extract(name, root, args.device, args.out)


if __name__ == "__main__":
    main()
