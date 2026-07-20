"""Held-out shared-code validity and effective-span endpoints for trained BSCs.

The calibration split fits per-(block, input-site) affine code maps, pairwise
orthogonal Procrustes maps, and rank-truncation bases.  The untouched eval
split supplies every reported score.  Cross-site code moments are conditioned
on the *full-code* threshold support; operational support IoU is reported
separately so coordinate agreement is not silently conflated with detection.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from block_crosscoder_experiment.store import StoreReader, Whitener

from .eval_activation_stats import load_model


@dataclass
class CodeMoments:
    count: torch.Tensor  # [G]
    full_sum: torch.Tensor  # [G,b]
    full_second: torch.Tensor  # [G,b,b]
    site_sum: torch.Tensor  # [S,G,b]
    site_second: torch.Tensor  # [S,G,b,b]
    site_full_cross: torch.Tensor  # [S,G,b,b], site.T @ full
    pair_cross: torch.Tensor  # [P,G,b,b], first-site.T @ second-site
    pairs: tuple[tuple[int, int], ...]
    n_tokens: int


def _zeros(S: int, G: int, b: int, device: str) -> CodeMoments:
    pairs = tuple(itertools.combinations(range(S), 2))
    kw = {"device": device, "dtype": torch.float64}
    return CodeMoments(
        count=torch.zeros(G, **kw),
        full_sum=torch.zeros(G, b, **kw),
        full_second=torch.zeros(G, b, b, **kw),
        site_sum=torch.zeros(S, G, b, **kw),
        site_second=torch.zeros(S, G, b, b, **kw),
        site_full_cross=torch.zeros(S, G, b, b, **kw),
        pair_cross=torch.zeros(len(pairs), G, b, b, **kw),
        pairs=pairs,
        n_tokens=0,
    )


def _site_code(model, x: torch.Tensor, site: int) -> torch.Tensor:
    return torch.einsum("nd,gbd->ngb", x[:, site], model.E[site].float())


def _scaled_batches(
    reader: StoreReader,
    batch_size: int,
    device: str,
    scale: torch.Tensor | None,
    limit: int | None,
):
    seen = 0
    for x in reader.sequential_batches(batch_size):
        if limit is not None:
            if seen >= limit:
                break
            x = x[: limit - seen]
        x = x.to(device=device, dtype=torch.float32)
        if scale is not None:
            x = x * scale
        seen += x.shape[0]
        yield x


@torch.no_grad()
def _update_code_moments(
    out: CodeMoments,
    full: torch.Tensor,
    mask: torch.Tensor,
    active_site_values: list[torch.Tensor],
) -> None:
    """Update sufficient statistics from one already-encoded batch."""
    G = out.count.shape[0]
    token_idx, block_idx = mask.nonzero(as_tuple=True)
    values_full = full[token_idx, block_idx].double()
    out.count += torch.bincount(block_idx, minlength=G).double()
    out.full_sum.index_add_(0, block_idx, values_full)
    out.full_second.index_add_(
        0, block_idx, torch.einsum("ni,nj->nij", values_full, values_full)
    )

    values = []
    for site, active_values in enumerate(active_site_values):
        site_values = active_values.double()
        values.append(site_values)
        out.site_sum[site].index_add_(0, block_idx, site_values)
        out.site_second[site].index_add_(
            0,
            block_idx,
            torch.einsum("ni,nj->nij", site_values, site_values),
        )
        out.site_full_cross[site].index_add_(
            0,
            block_idx,
            torch.einsum("ni,nj->nij", site_values, values_full),
        )
    for pair_idx, (a, b_) in enumerate(out.pairs):
        out.pair_cross[pair_idx].index_add_(
            0,
            block_idx,
            torch.einsum("ni,nj->nij", values[a], values[b_]),
        )
    out.n_tokens += full.shape[0]


@torch.no_grad()
def accumulate_code_moments(
    model,
    batches,
) -> CodeMoments:
    """Accumulate active-event moments without materializing ``[B,S,G,b]``."""
    S, G, b = model.cfg.n_sites, model.cfg.n_blocks, model.cfg.block_dim
    out = _zeros(S, G, b, str(model.E.device))
    for x in batches:
        full = model.encode(x)
        mask = model.scores(full) > model.theta
        token_idx, block_idx = mask.nonzero(as_tuple=True)
        # Retain only active events between sites: O(B*k*S*b), not
        # O(B*G*S*b).
        active_values = [
            _site_code(model, x, site)[token_idx, block_idx] for site in range(S)
        ]
        _update_code_moments(out, full, mask, active_values)
    return out


def _centered(
    count: torch.Tensor,
    x_sum: torch.Tensor,
    y_sum: torch.Tensor,
    xx: torch.Tensor,
    xy: torch.Tensor,
    yy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = count.clamp_min(1)
    cxx = xx - torch.einsum("...i,...j->...ij", x_sum, x_sum) / n[..., None, None]
    cxy = xy - torch.einsum("...i,...j->...ij", x_sum, y_sum) / n[..., None, None]
    cyy = yy - torch.einsum("...i,...j->...ij", y_sum, y_sum) / n[..., None, None]
    return cxx, cxy, cyy


def fit_affine_maps(
    count: torch.Tensor,
    x_sum: torch.Tensor,
    y_sum: torch.Tensor,
    xx: torch.Tensor,
    xy: torch.Tensor,
    yy: torch.Tensor,
    *,
    ridge_rel: float = 1e-5,
    min_count: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit ``y = x @ A + t`` independently over leading dimensions."""
    cxx, cxy, _ = _centered(count, x_sum, y_sum, xx, xy, yy)
    width = x_sum.shape[-1]
    trace = cxx.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0)
    ridge = ridge_rel * trace.clamp_min(1e-12) / width
    eye = torch.eye(width, dtype=cxx.dtype, device=cxx.device)
    A = torch.linalg.solve(cxx + ridge[..., None, None] * eye, cxy)
    n = count.clamp_min(1).unsqueeze(-1)
    mx, my = x_sum / n, y_sum / n
    t = my - torch.einsum("...i,...ij->...j", mx, A)
    valid = count >= min_count
    A = torch.where(valid[..., None, None], A, eye)
    t = torch.where(valid[..., None], t, torch.zeros_like(t))
    return A, t, valid


def affine_r2(
    count: torch.Tensor,
    x_sum: torch.Tensor,
    y_sum: torch.Tensor,
    xx: torch.Tensor,
    xy: torch.Tensor,
    yy: torch.Tensor,
    A: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Held-out multivariate R2 for fixed affine row-vector maps."""
    cross = (A * xy).sum(dim=(-2, -1))
    quadratic = ((xx @ A) * A).sum(dim=(-2, -1))
    sxA = torch.einsum("...i,...ij->...j", x_sum, A)
    sse = (
        yy.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 2 * cross
        - 2 * (t * y_sum).sum(-1)
        + quadratic
        + 2 * (t * sxA).sum(-1)
        + count * t.square().sum(-1)
    )
    baseline = yy.diagonal(dim1=-2, dim2=-1).sum(-1) - (
        y_sum.square().sum(-1) / count.clamp_min(1)
    )
    score = 1 - sse / baseline.clamp_min(1e-24)
    return torch.where((count > 1) & (baseline > 1e-24), score, torch.nan)


def fit_procrustes_maps(
    count: torch.Tensor,
    x_sum: torch.Tensor,
    y_sum: torch.Tensor,
    xx: torch.Tensor,
    xy: torch.Tensor,
    yy: torch.Tensor,
    *,
    min_count: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibration-fit centered orthogonal maps, with held-out scoring later."""
    _, cxy, _ = _centered(count, x_sum, y_sum, xx, xy, yy)
    U, _, Vh = torch.linalg.svd(cxy)
    R = U @ Vh
    n = count.clamp_min(1).unsqueeze(-1)
    mx, my = x_sum / n, y_sum / n
    t = my - torch.einsum("...i,...ij->...j", mx, R)
    valid = count >= min_count
    eye = torch.eye(x_sum.shape[-1], dtype=x_sum.dtype, device=x_sum.device)
    R = torch.where(valid[..., None, None], R, eye)
    t = torch.where(valid[..., None], t, torch.zeros_like(t))
    return R, t, valid


def canonical_correlations(
    count: torch.Tensor,
    x_sum: torch.Tensor,
    y_sum: torch.Tensor,
    xx: torch.Tensor,
    xy: torch.Tensor,
    yy: torch.Tensor,
    *,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """Rank-aware canonical correlations from held-out centered moments."""
    cxx, cxy, cyy = _centered(count, x_sum, y_sum, xx, xy, yy)

    def invsqrt(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e, V = torch.linalg.eigh(cov)
        keep = e > e[..., -1:].clamp_min(0) * rtol
        inv = torch.where(keep, e.clamp_min(1e-24).rsqrt(), 0)
        return (V * inv.unsqueeze(-2)) @ V.mT, keep.sum(-1)

    ix, rx = invsqrt(cxx)
    iy, ry = invsqrt(cyy)
    corr = torch.linalg.svdvals(ix @ cxy @ iy).clamp(0, 1)
    cols = torch.arange(corr.shape[-1], device=corr.device)
    rank = torch.minimum(rx, ry)
    return torch.where(cols < rank.unsqueeze(-1), corr, torch.nan)


def _pair_views(m: CodeMoments):
    a = torch.tensor([p[0] for p in m.pairs], device=m.count.device)
    b = torch.tensor([p[1] for p in m.pairs], device=m.count.device)
    count = m.count.unsqueeze(0).expand(len(m.pairs), -1)
    return (
        count,
        m.site_sum[a],
        m.site_sum[b],
        m.site_second[a],
        m.pair_cross,
        m.site_second[b],
    )


def _site_full_views(m: CodeMoments):
    count = m.count.unsqueeze(0).expand(m.site_sum.shape[0], -1)
    return (
        count,
        m.site_sum,
        m.full_sum.unsqueeze(0).expand_as(m.site_sum),
        m.site_second,
        m.site_full_cross,
        m.full_second.unsqueeze(0).expand_as(m.site_second),
    )


def _fvu(error: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    return error / energy.clamp_min(1e-24)


def _pooled_fvu(error: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
    return error.sum(-1) / energy.sum().clamp_min(1e-24)


def sparse_decode(
    model,
    code: torch.Tensor,
    mask: torch.Tensor,
    *,
    event_chunk: int = 1024,
) -> torch.Tensor:
    """Exact decoder evaluation with work proportional to active events.

    The dense training decoder is ideal when gradients and regular batches are
    required.  This endpoint evaluates many threshold-sparse alternatives; a
    block-event gather and token scatter avoids multiplying each candidate by
    all ``G`` decoder blocks.  Chunking bounds the temporary ``[events,b,d]``
    gather on the 24 GB production GPU.
    """
    batch, sites, d_model = code.shape[0], model.cfg.n_sites, model.cfg.d_model
    out = model.c.float().unsqueeze(0).expand(batch, sites, d_model).clone()
    token_idx, block_idx = mask.nonzero(as_tuple=True)
    active_code = code[token_idx, block_idx].float()
    for start in range(0, token_idx.numel(), event_chunk):
        stop = start + event_chunk
        tokens = token_idx[start:stop]
        blocks = block_idx[start:stop]
        values = active_code[start:stop]
        for site in range(sites):
            contribution = torch.einsum(
                "mb,mbd->md", values, model.D[site, blocks].float()
            )
            out[:, site].index_add_(0, tokens, contribution)
    return out


@torch.no_grad()
def reconstruction_endpoints(
    model,
    batches,
    maps: torch.Tensor,
    offsets: torch.Tensor,
    cal: CodeMoments,
) -> tuple[dict[str, torch.Tensor | int], CodeMoments]:
    """Eval reconstruction/moment endpoints in one pass over untouched eval."""
    S, G, b = model.cfg.n_sites, model.cfg.n_blocks, model.cfg.block_dim
    device = model.E.device
    eval_moments = _zeros(S, G, b, str(device))
    energy = torch.zeros(S, dtype=torch.float64, device=device)
    full_error = torch.zeros(S, dtype=torch.float64, device=device)
    raw = torch.zeros(S, S, dtype=torch.float64, device=device)
    mapped = torch.zeros_like(raw)
    raw_oracle = torch.zeros_like(raw)
    mapped_oracle = torch.zeros_like(raw)
    loo = torch.zeros_like(raw)
    loo_oracle = torch.zeros_like(raw)
    trunc_second = torch.zeros(b, S, dtype=torch.float64, device=device)
    trunc_centered = torch.zeros_like(trunc_second)
    raw_inter = torch.zeros(S, dtype=torch.float64, device=device)
    raw_union = torch.zeros_like(raw_inter)
    map_inter = torch.zeros_like(raw_inter)
    map_union = torch.zeros_like(raw_inter)
    n_tokens = 0

    count = cal.count.clamp_min(1)
    mean = cal.full_sum / count.unsqueeze(-1)
    second = cal.full_second / count[:, None, None]
    cov = second - torch.einsum("gi,gj->gij", mean, mean)
    _, U2 = torch.linalg.eigh(second)
    _, Uc = torch.linalg.eigh(cov)
    U2, Uc = U2.flip(-1), Uc.flip(-1)

    def add_error(
        target: torch.Tensor,
        code: torch.Tensor,
        mask: torch.Tensor,
        acc: torch.Tensor,
        row: int | None = None,
    ):
        err = (
            (target - sparse_decode(model, code, mask))
            .double()
            .square()
            .sum(dim=(0, 2))
        )
        if row is None:
            acc += err
        else:
            acc[row] += err

    for x in batches:
        full = model.encode(x)
        full_mask = model.scores(full) > model.theta
        selected = full * full_mask.unsqueeze(-1)
        energy += x.double().square().sum(dim=(0, 2))
        add_error(x, selected, full_mask, full_error)
        token_idx, block_idx = full_mask.nonzero(as_tuple=True)
        active_site_values = []
        for site in range(S):
            zs = _site_code(model, x, site)
            active_site_values.append(zs[token_idx, block_idx])
            raw_mask = model.scores(zs) > model.theta
            zm = (
                torch.einsum("ngb,gbc->ngc", zs, maps[site].float())
                + offsets[site].float()
            )
            mapped_mask = model.scores(zm) > model.theta
            raw_inter[site] += (raw_mask & full_mask).sum()
            raw_union[site] += (raw_mask | full_mask).sum()
            map_inter[site] += (mapped_mask & full_mask).sum()
            map_union[site] += (mapped_mask | full_mask).sum()
            add_error(x, zs, raw_mask, raw, site)
            add_error(x, zm, mapped_mask, mapped, site)
            add_error(x, zs, full_mask, raw_oracle, site)
            add_error(x, zm, full_mask, mapped_oracle, site)
            zloo = full - zs
            loo_mask = model.scores(zloo) > model.theta
            add_error(x, zloo, loo_mask, loo, site)
            add_error(x, zloo, full_mask, loo_oracle, site)

        _update_code_moments(eval_moments, full, full_mask, active_site_values)

        for rank in range(1, b + 1):
            P2 = U2[..., :rank] @ U2[..., :rank].mT
            Pc = Uc[..., :rank] @ Uc[..., :rank].mT
            z2 = torch.einsum("ngb,gbc->ngc", selected, P2.float())
            zc = (
                torch.einsum("ngb,gbc->ngc", full - mean.float(), Pc.float())
                + mean.float()
            )
            zc = zc * full_mask.unsqueeze(-1)
            add_error(x, z2, full_mask, trunc_second[rank - 1])
            add_error(x, zc, full_mask, trunc_centered[rank - 1])
        n_tokens += x.shape[0]

    return {
        "energy": energy,
        "full_fvu": _fvu(full_error, energy),
        "full_pooled_fvu": _pooled_fvu(full_error, energy),
        "single_site_fvu": _fvu(raw, energy),
        "single_site_pooled_fvu": _pooled_fvu(raw, energy),
        "mapped_single_site_fvu": _fvu(mapped, energy),
        "mapped_single_site_pooled_fvu": _pooled_fvu(mapped, energy),
        "single_site_oracle_fvu": _fvu(raw_oracle, energy),
        "single_site_oracle_pooled_fvu": _pooled_fvu(raw_oracle, energy),
        "mapped_single_site_oracle_fvu": _fvu(mapped_oracle, energy),
        "mapped_single_site_oracle_pooled_fvu": _pooled_fvu(mapped_oracle, energy),
        "leave_one_out_fvu": _fvu(loo, energy),
        "leave_one_out_pooled_fvu": _pooled_fvu(loo, energy),
        "leave_one_out_oracle_fvu": _fvu(loo_oracle, energy),
        "leave_one_out_oracle_pooled_fvu": _pooled_fvu(loo_oracle, energy),
        "truncation_second_fvu": _fvu(trunc_second, energy),
        "truncation_second_pooled_fvu": _pooled_fvu(trunc_second, energy),
        "truncation_centered_fvu": _fvu(trunc_centered, energy),
        "truncation_centered_pooled_fvu": _pooled_fvu(trunc_centered, energy),
        "single_site_support_iou": raw_inter / raw_union.clamp_min(1),
        "mapped_single_site_support_iou": map_inter / map_union.clamp_min(1),
        "n_tokens": n_tokens,
    }, eval_moments


def _cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@torch.no_grad()
def run_endpoints(
    name: str,
    root: Path,
    device: str,
    out_dir: Path,
    batch_size: int,
    renorm_scalars: torch.Tensor,
    store: Path,
    fit_tokens: int | None,
    eval_tokens: int | None,
    min_map_activations: int,
    whitener_hash: str | None = None,
) -> None:
    model, report = load_model(root, device)
    if not bool(torch.isfinite(model.theta)):
        raise ValueError(f"{root} has no finite calibration threshold")
    S = model.cfg.n_sites
    scale = None
    if report.get("site_renorm_at_load", report.get("site_renorm")):
        scale = renorm_scalars.to(device).view(1, S, 1)

    cal = accumulate_code_moments(
        model,
        _scaled_batches(
            StoreReader(store, "calibration", expected_whitener_hash=whitener_hash),
            batch_size,
            device,
            scale,
            fit_tokens,
        ),
    )
    site_cal = _site_full_views(cal)
    maps, offsets, map_valid = fit_affine_maps(*site_cal, min_count=min_map_activations)
    pair_cal = _pair_views(cal)
    rotations, rotation_offsets, proc_valid = fit_procrustes_maps(
        *pair_cal, min_count=min_map_activations
    )

    recon, eval_m = reconstruction_endpoints(
        model,
        _scaled_batches(
            StoreReader(store, "eval", expected_whitener_hash=whitener_hash),
            batch_size,
            device,
            scale,
            eval_tokens,
        ),
        maps,
        offsets,
        cal,
    )
    site_eval = _site_full_views(eval_m)
    map_r2 = affine_r2(*site_eval, maps, offsets)
    pair_eval = _pair_views(eval_m)
    proc_r2 = affine_r2(*pair_eval, rotations, rotation_offsets)
    cca = canonical_correlations(*pair_eval)

    headline = eval_m.count >= 10_000
    payload: dict[str, object] = {
        "calibration_active_count": _cpu(cal.count),
        "eval_active_count": _cpu(eval_m.count),
        "site_full_map": _cpu(maps).astype(np.float32),
        "site_full_offset": _cpu(offsets).astype(np.float32),
        "site_full_map_valid": _cpu(map_valid),
        "site_full_map_r2": _cpu(map_r2).astype(np.float32),
        "site_pair_procrustes": _cpu(rotations).astype(np.float32),
        "site_pair_procrustes_offset": _cpu(rotation_offsets).astype(np.float32),
        "site_pair_procrustes_valid": _cpu(proc_valid),
        "site_pair_procrustes_r2": _cpu(proc_r2).astype(np.float32),
        "site_pair_canonical_correlations": _cpu(cca).astype(np.float32),
        "site_pairs": np.asarray(eval_m.pairs, dtype=np.int16),
        "headline_eligible": _cpu(headline),
    }
    for key, value in recon.items():
        payload[key] = (
            np.int64(value)
            if isinstance(value, int)
            else _cpu(value).astype(np.float32)
        )
    payload["meta"] = json.dumps(
        {
            "run": str(root),
            "model_cfg": {
                "n_sites": S,
                "n_blocks": model.cfg.n_blocks,
                "block_dim": model.cfg.block_dim,
                "d_model": model.cfg.d_model,
            },
            "calibration_tokens": cal.n_tokens,
            "eval_tokens": eval_m.n_tokens,
            "map_conditioning": "full-code threshold-active events",
            "map_form": "affine ridge regression, row code y=xA+t",
            "map_ridge_relative_trace": 1e-5,
            "procrustes_form": "centered orthogonal map with calibration translation",
            "cca_covariance_rtol": 1e-6,
            "headline_min_eval_activations": 10_000,
            "min_map_activations": min_map_activations,
            "single_site_fvu_support": "own threshold",
            "single_site_oracle_fvu_support": "full-code threshold",
            "truncation_fit_split": "calibration",
            "truncation_form": "joint code/decoder code-space projection",
        }
    )
    path = out_dir / f"trained_endpoints_{name}.npz"
    np.savez_compressed(path, **payload)
    eligible = headline & map_valid.all(0)
    med_r2 = (
        float(torch.nanmedian(map_r2[:, eligible]))
        if bool(eligible.any())
        else float("nan")
    )
    print(
        f"{name}: {cal.n_tokens:,} calibration, {eval_m.n_tokens:,} eval; "
        f"headline {int(headline.sum())}/{model.cfg.n_blocks}; median mapped R2 {med_r2:.3f} -> {path}",
        flush=True,
    )


def main() -> None:
    from .artifacts import analysis_dir, load_winner

    winner = load_winner()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--fit-tokens", type=int, default=None)
    ap.add_argument("--eval-tokens", type=int, default=None)
    ap.add_argument("--min-map-activations", type=int, default=1000)
    ap.add_argument("--runs", nargs="*", default=None, metavar="NAME=PATH")
    args = ap.parse_args()
    args.store = args.store or Path(winner["store"])
    args.out = args.out or analysis_dir(winner)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {
        "winner": str(Path(winner["ckpt"]).parent),
        "primary": winner["counterpart_primary"],
    }
    if args.runs:
        runs = dict(pair.split("=", 1) for pair in args.runs)
    whitener = Whitener.load(args.store / "whitener.pt")
    renorm = whitener.site_rms_scalars()
    for name, path in runs.items():
        root = Path(path)
        if not (root / "latest.pt").exists():
            print(f"{name}: missing {root}, skipped")
            continue
        run_endpoints(
            name,
            root,
            args.device,
            args.out,
            args.batch,
            renorm,
            args.store,
            args.fit_tokens,
            args.eval_tokens,
            args.min_map_activations,
            whitener.hash,
        )


if __name__ == "__main__":
    main()
