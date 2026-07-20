"""Extract winner-scoped capacity and *used* block geometry.

Decoder spectra remain in the artifact because they are the correct capacity
diagnostic.  Every empirical span statement, however, is anchored to active
code moments from ``evalstats_<arm>.npz``:

``second contribution``
    ``D.T E[zz.T | active] D`` (conditional mean included).
``centered contribution``
    ``D.T Cov[z | active] D`` (within-feature position).

Rank-deficient factors are represented by their numerical rank.  Padded
principal-cosine entries are NaN; no QR completion is allowed to turn a zero
singular direction into an apparently observed frame direction.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

CHUNK = 32768  # cusolver batched kernels cap at 65535
RANK_RTOL = 1e-5
RANK_ATOL = 1e-10


def svd_chunked(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return singular values and right bases for ``[..., rows, d]``."""
    lead = A.shape[:-2]
    flat = A.reshape(-1, A.shape[-2], A.shape[-1])
    values, bases = [], []
    for i in range(0, flat.shape[0], CHUNK):
        _, sv, vh = torch.linalg.svd(flat[i : i + CHUNK], full_matrices=False)
        values.append(sv)
        bases.append(vh.transpose(1, 2))
    return (
        torch.cat(values).reshape(*lead, -1),
        torch.cat(bases).reshape(*lead, A.shape[-1], -1),
    )


def svdvals_chunked(A: torch.Tensor) -> torch.Tensor:
    """Singular values without retaining the often enormous right bases."""
    lead = A.shape[:-2]
    flat = A.reshape(-1, A.shape[-2], A.shape[-1])
    values = [
        torch.linalg.svdvals(flat[i : i + CHUNK])
        for i in range(0, flat.shape[0], CHUNK)
    ]
    return torch.cat(values).reshape(*lead, -1)


def numerical_rank(svals: torch.Tensor) -> torch.Tensor:
    """Relative numerical rank, preserving exact zero-rank factors."""
    cutoff = torch.maximum(
        svals[..., :1] * RANK_RTOL,
        torch.full_like(svals[..., :1], RANK_ATOL),
    )
    return (svals > cutoff).sum(-1)


def psd_sqrt(moment: torch.Tensor) -> torch.Tensor:
    """Stable batched PSD square root; small negative roundoff is clipped."""
    values, vectors = torch.linalg.eigh(moment.double())
    scale = values[..., -1:].clamp_min(0)
    # Legacy evalstats stored moment sums in fp32, so allow the corresponding
    # cancellation-sized negative tail while still rejecting a genuinely
    # indefinite empirical moment.  New artifacts preserve fp64 sums.
    floor = scale * 1e-5 + 1e-12
    values = torch.where(values >= -floor, values.clamp_min(0), values)
    if bool((values < 0).any()):
        worst = float(values.min())
        raise ValueError(f"code moment is not PSD (minimum eigenvalue {worst:.3e})")
    return ((vectors * values.sqrt().unsqueeze(-2)) @ vectors.mT).float()


def moment_factors(moment: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """``K^1/2 D_s`` factors, shaped ``[S,G,b,d]``."""
    root = psd_sqrt(moment)
    return torch.einsum("gij,sgjd->sgid", root, D)


def aligned_principal_cos(
    Qa: torch.Tensor,
    rank_a: torch.Tensor,
    Qb: torch.Tensor,
    rank_b: torch.Tensor,
) -> torch.Tensor:
    """Batched variable-rank principal cosines, NaN-padded to width ``b``."""
    width = Qa.shape[-1]
    cols = torch.arange(width, device=Qa.device)
    ma = cols < rank_a.unsqueeze(-1)
    mb = cols < rank_b.unsqueeze(-1)
    cross = (Qa * ma.unsqueeze(-2)).mT @ (Qb * mb.unsqueeze(-2))
    sv = torch.linalg.svdvals(cross).clamp(0, 1)
    common = torch.minimum(rank_a, rank_b)
    return torch.where(cols < common.unsqueeze(-1), sv, torch.nan)


def pair_principal_cos(
    Q: torch.Tensor,
    ranks: torch.Tensor,
    *,
    permutations: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Variable-rank site-pair cosines for bases ``Q[S,G,d,b]``."""
    sites, blocks = Q.shape[:2]
    pairs = list(itertools.combinations(range(sites), 2))
    identity = torch.arange(blocks, device=Q.device)
    out = []
    for a, b_ in pairs:
        ia = identity if permutations is None else permutations[a]
        ib = identity if permutations is None else permutations[b_]
        out.append(
            aligned_principal_cos(Q[a, ia], ranks[a, ia], Q[b_, ib], ranks[b_, ib])
        )
    return torch.stack(out, dim=1).cpu().numpy(), pairs


def conditional_moments(stats: np.lib.npyio.NpzFile) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conditional second moment and centered covariance ``[G,b,b]``."""
    if "z_sum" not in stats.files:
        raise ValueError(
            "evalstats lacks z_sum; rerun `bsc activation-stats` before geometry"
        )
    count = torch.from_numpy(stats["fire_count"].astype(np.float64))
    total = torch.from_numpy(stats["zz"].astype(np.float64))
    summed = torch.from_numpy(stats["z_sum"].astype(np.float64))
    denom = count.clamp_min(1).view(-1, 1, 1)
    second = total / denom
    mean = summed / count.clamp_min(1).unsqueeze(-1)
    centered = second - torch.einsum("gi,gj->gij", mean, mean)
    second[count == 0] = 0
    centered[count <= 1] = 0
    return second, centered


def _site_share(site_energy: np.ndarray) -> np.ndarray:
    denom = site_energy.sum(1, keepdims=True)
    return np.divide(
        site_energy,
        denom,
        out=np.zeros_like(site_energy, dtype=np.float64),
        where=denom > 0,
    )


def spectral_summaries(evals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Participation ratio and 95%-energy rank; zero energy has rank zero."""
    values = evals.clamp_min(0)
    total = values.sum(-1)
    pr = total.square() / values.square().sum(-1).clamp_min(1e-24)
    cumulative = values.cumsum(-1) / total.clamp_min(1e-24).unsqueeze(-1)
    rank95 = (cumulative < 0.95).sum(-1) + 1
    rank95 = torch.where(total > 0, rank95, torch.zeros_like(rank95))
    return pr, rank95


def extract(
    name: str,
    root: Path,
    device: str,
    out_dir: Path,
    sites: list[int],
) -> None:
    stats_path = out_dir / f"evalstats_{name}.npz"
    if not stats_path.exists():
        raise SystemExit(
            f"missing {stats_path}; run `bsc activation-stats` before geometry"
        )
    stats = np.load(stats_path)
    second_cpu, centered_cpu = conditional_moments(stats)

    ckpt = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    report = json.loads((root / "report.json").read_text())
    cfg = ckpt["model_cfg"]
    S, G, b, d = cfg["n_sites"], cfg["n_blocks"], cfg["block_dim"], cfg["d_model"]
    D = ckpt["model"]["D"].to(device=device, dtype=torch.float32)
    E = ckpt["model"]["E"].to(device=device, dtype=torch.float32)
    c = ckpt["model"]["c"].float()
    theta = float(ckpt["model"]["theta"])

    # Decoder capacity and Gram sanity.
    energy = D.pow(2).sum(dim=(2, 3)).T
    decoder_share = (energy / b).cpu().numpy()
    gram = torch.einsum("sgbd,sgcd->gbc", D, D)
    eye = torch.eye(b, device=device).expand(G, b, b)
    gram_residual = (gram - eye).flatten(1).norm(dim=1).cpu().numpy()
    enc_energy = E.pow(2).sum(dim=(2, 3)).T
    enc_share = (enc_energy / enc_energy.sum(1, keepdim=True)).cpu().numpy()

    decoder_svals, Qd = svd_chunked(D)
    decoder_rank = numerical_rank(decoder_svals)
    decoder_pair_cos, pairs = pair_principal_cos(Qd, decoder_rank)
    stacked = D.permute(1, 0, 2, 3).reshape(G, S * b, d)
    decoder_stacked_svals = svdvals_chunked(stacked)

    # Contribution spectra.  The centered factors define empirical used spans.
    second = second_cpu.to(device=device, dtype=torch.float32)
    centered = centered_cpu.to(device=device, dtype=torch.float32)
    factors = moment_factors(second, D)
    contrib_svals = svdvals_chunked(factors)
    del factors, second
    centered_factors = moment_factors(centered, D)
    centered_svals, used_Q = svd_chunked(centered_factors)
    used_rank = numerical_rank(centered_svals)
    used_pair_cos, _ = pair_principal_cos(used_Q, used_rank)

    used_stack = centered_factors.permute(1, 0, 2, 3).reshape(G, S * b, d)
    used_stacked_svals = svdvals_chunked(used_stack)

    decoder_evals = decoder_svals.square()
    contribution_evals = contrib_svals.square()
    centered_contribution_evals = centered_svals.square()
    decoder_pr, decoder_rank95 = spectral_summaries(decoder_evals)
    contribution_pr, contribution_rank95 = spectral_summaries(contribution_evals)
    centered_pr, centered_rank95 = spectral_summaries(centered_contribution_evals)
    centered_stacked_pr, centered_stacked_rank95 = spectral_summaries(
        used_stacked_svals.square()
    )

    # Rank-aware shuffled-block null: site marginals retained, identities broken.
    gen = torch.Generator().manual_seed(0)
    perms = torch.stack(
        [torch.arange(G)]
        + [torch.randperm(G, generator=gen) for _ in range(1, S)]
    ).to(device)
    used_null_cos, _ = pair_principal_cos(used_Q, used_rank, permutations=perms)
    decoder_null_cos, _ = pair_principal_cos(Qd, decoder_rank, permutations=perms)

    # Encoder/decoder capacity alignment, also rank-aware.
    encoder_svals, Qe = svd_chunked(E)
    encoder_rank = numerical_rank(encoder_svals)
    encdec = aligned_principal_cos(
        Qd.reshape(S * G, d, b),
        decoder_rank.reshape(S * G),
        Qe.reshape(S * G, d, b),
        encoder_rank.reshape(S * G),
    ).reshape(S, G, b)

    site_energy = stats["site_energy"].astype(np.float64)
    site_share = _site_share(site_energy)
    active_count = stats["fire_count"].astype(np.int64)
    np.savez_compressed(
        out_dir / f"geometry_{name}.npz",
        # Explicit names plus compatibility aliases used by older consumers.
        decoder_share=decoder_share.astype(np.float32),
        share=decoder_share.astype(np.float32),
        site_share=site_share.astype(np.float32),
        site_energy=site_energy.astype(np.float32),
        active_count=active_count,
        enc_share=enc_share.astype(np.float32),
        decoder_svals=decoder_svals.cpu().numpy().astype(np.float32),
        decoder_evals=decoder_evals.cpu().numpy().astype(np.float32),
        decoder_pr=decoder_pr.cpu().numpy().astype(np.float32),
        decoder_rank95=decoder_rank95.cpu().numpy().astype(np.int8),
        svals=decoder_svals.cpu().numpy().astype(np.float32),
        decoder_rank=decoder_rank.cpu().numpy().astype(np.int8),
        decoder_pair_cos=decoder_pair_cos.astype(np.float16),
        decoder_null_pair_cos=decoder_null_cos.astype(np.float16),
        pair_cos=used_pair_cos.astype(np.float16),
        null_pair_cos=used_null_cos.astype(np.float16),
        used_rank=used_rank.cpu().numpy().astype(np.int8),
        pairs=np.array(pairs, dtype=np.int32),
        decoder_stacked_svals=decoder_stacked_svals.cpu().numpy().astype(np.float32),
        stacked_svals=decoder_stacked_svals.cpu().numpy().astype(np.float32),
        contribution_evals=contribution_evals.cpu().numpy().astype(np.float32),
        contribution_pr=contribution_pr.cpu().numpy().astype(np.float32),
        contribution_rank95=contribution_rank95.cpu().numpy().astype(np.int8),
        centered_contribution_evals=centered_contribution_evals.cpu().numpy().astype(np.float32),
        centered_contribution_pr=centered_pr.cpu().numpy().astype(np.float32),
        centered_contribution_rank95=centered_rank95.cpu().numpy().astype(np.int8),
        centered_stacked_svals=used_stacked_svals.cpu().numpy().astype(np.float32),
        centered_stacked_pr=centered_stacked_pr.cpu().numpy().astype(np.float32),
        centered_stacked_rank95=centered_stacked_rank95.cpu().numpy().astype(np.int8),
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
                "fvu_pooled": report.get("eval", {}).get("topk", {}).get("fvu_pooled"),
                "fvu_per_site": report.get("eval", {}).get("topk", {}).get("fvu_per_site"),
                "sites": sites,
                "used_span": "centered conditional contribution covariance",
                "rank_rtol": RANK_RTOL,
                "rank_atol": RANK_ATOL,
            }
        ),
    )
    print(
        f"{name}: G={G} b={b} gram_res max {gram_residual.max():.2e}; "
        f"headline eligible {(active_count >= 10_000).sum()}/{G}",
        flush=True,
    )


def main() -> None:
    from .artifacts import analysis_dir, load_winner

    winner = load_winner()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--runs", nargs="*", default=None, metavar="NAME=PATH",
        help="override winner + primary (name=path pairs)",
    )
    ap.add_argument(
        "--sites", type=int, nargs="*", default=None,
        help="site layer list recorded in metadata; defaults to winner sites",
    )
    args = ap.parse_args()
    runs = {
        "winner": str(Path(winner["ckpt"]).parent),
        "primary": winner["counterpart_primary"],
    }
    if args.runs:
        runs = dict(pair.split("=", 1) for pair in args.runs)
    args.out = args.out or analysis_dir(winner)
    args.sites = args.sites or winner["sites"]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, root in runs.items():
        root = Path(root)
        if not (root / "latest.pt").exists():
            print(f"{name}: missing {root}, skipped")
            continue
        extract(name, root, args.device, args.out, args.sites)


if __name__ == "__main__":
    main()
