"""The preregistered rate–distortion codec (design §Rate–distortion
protocol; runbook-phase099 tranche 3).

Everything here is fit on the CALIBRATION split and frozen before the
eval split is touched (R15). Selection runs in threshold mode — the
codec prices the deployed inference path, variable per-token counts
included (finding 13). Both arms (block and scalar) flow through the
identical code path; b=1 makes the orientation trivial and the
amplitude obligation one coordinate.

Pipeline, per model:

1. **Calibration pass** (`fit_codec`): stream the calib split once,
   collecting every selected block-event (code vector + block id) plus
   the per-token count histogram and per-block firing frequencies.
2. **Active-count floor**: blocks with fewer than `floor` calib events
   are EXCLUDED from the codec — zeroed at decode, mask-stripped before
   counting, paying no bits — identically in both arms; exclusions and
   their calib/eval usage shares are reported openly (the pilot's 2M
   calib split makes the floor bite harder than production's 13M).
3. **Canonical orientation** (R13): per block, rotate the code space to
   diagonalize the calib active-code second moment (descending); sign
   fixed so the active-mean projection is nonnegative. Exploits the
   residual O(b) gauge; frozen thereafter. Without it, an arbitrary
   gauge rotation changes componentwise clipping while the model is
   unchanged (tested: gauge-rotated models produce matching R-D points).
4. **Quantizer**: per canonical coordinate, clip to the calib
   0.1%/99.9% quantiles, then 2^q uniform levels spanning the range
   (endpoints included: xhat = lo + round(t*(2^q-1)) * (hi-lo)/(2^q-1));
   out-of-range saturates. q swept per spec.
5. **Support bits/token**: -log2 P(k_t) + log2 C(G_included, k_t), with
   P the calib count histogram, add-one smoothed over [0, K_max],
   K_max = 2*max_calib_count + 8; eval counts beyond K_max (never seen
   in practice) price at P(K_max). The enumerative term is deliberately
   usage-agnostic (R17); the Bernoulli product model over per-block
   calib frequencies is computed alongside as the declared
   support-entropy sensitivity.
6. **Amplitude bits/token**: q * b * k_t — each selected block carries
   the obligation to transmit b coordinates (finding 12); the scalar
   arm pays q * l_t for its own realized l_t (R14).
7. **Distortion**: whitened FVU through the quantized codes, per site
   and pooled, centering by the CALIB-fit per-site mean (no eval-fit
   parameters anywhere).
8. **Uncertainty** (R18): bootstrap over stored SEQUENCES (contiguous
   `row_len`-token groups of the sequential eval stream), never tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = ["CodecSpec", "Codec", "fit_codec", "evaluate_rd"]


@dataclass
class CodecSpec:
    qs: tuple[int, ...] = (4, 6, 8)
    clip_lo: float = 0.001  # 0.1% quantile
    clip_hi: float = 0.999  # 99.9% quantile
    floor: int = 1000  # min calib active events for codec inclusion
    n_bootstrap: int = 1000
    bootstrap_seed: int = 0


@dataclass
class Codec:
    """Frozen codec metadata — everything fit on calibration."""

    spec: CodecSpec
    included: torch.Tensor  # [G] bool
    rotation: torch.Tensor  # [G, b, b] canonical frames (row-major: z_can = R z)
    lo: torch.Tensor  # [G, b] clip floor, canonical coords
    hi: torch.Tensor  # [G, b] clip ceiling, canonical coords
    count_log2p: torch.Tensor  # [K_max+1] log2 of smoothed count model
    bernoulli_log2p: torch.Tensor  # [G] log2 p_hat (smoothed firing freq)
    bernoulli_log2q: torch.Tensor  # [G] log2 (1 - p_hat)
    calib_events: torch.Tensor  # [G] active-event counts (reporting)
    calib_tokens: int
    calib_mean: torch.Tensor  # [S, d] fp64 per-site mean (FVU centering)
    meta: dict = field(default_factory=dict)

    @property
    def n_included(self) -> int:
        return int(self.included.sum())

    def log2_count_prob(self, k: torch.Tensor) -> torch.Tensor:
        return self.count_log2p[k.clamp(max=self.count_log2p.numel() - 1)]

    def quantize(self, z_can: torch.Tensor, q: int) -> torch.Tensor:
        """z_can: [n, G, b] canonical-frame codes -> quantized, same frame."""
        levels = (1 << q) - 1
        lo, hi = self.lo.to(z_can.device), self.hi.to(z_can.device)
        span = (hi - lo).clamp_min(1e-12)
        t = ((z_can - lo) / span).clamp(0.0, 1.0)
        return lo + torch.round(t * levels) / levels * span


def _log2_binom(n: int, k: torch.Tensor) -> torch.Tensor:
    """log2 C(n, k), elementwise over integer tensor k (values > n clamp)."""
    kf = k.clamp(max=n).double()
    nf = float(n)
    return (
        torch.lgamma(torch.tensor(nf + 1.0)).double()
        - torch.lgamma(kf + 1.0)
        - torch.lgamma(nf - kf + 1.0)
    ) / math.log(2.0)


@torch.no_grad()
def fit_codec(model, batches, spec: CodecSpec, *, device: str = "cpu") -> Codec:
    """One calibration pass. `batches` yields [B, S, d] (CPU, any float
    dtype except fp16); selection in threshold mode against the model's
    frozen theta."""
    G, b = model.cfg.n_blocks, model.cfg.block_dim
    S, d = model.cfg.n_sites, model.cfg.d_model

    ev_codes: list[torch.Tensor] = []
    ev_ids: list[torch.Tensor] = []
    count_hist = torch.zeros(0, dtype=torch.long)
    block_events = torch.zeros(G, dtype=torch.long)
    mean_acc = torch.zeros(S, d, dtype=torch.float64)
    n_tokens = 0

    for x in batches:
        x = x.to(device, torch.float32)
        out = model(x, mode="threshold")
        mask = out.mask
        z_sel = out.z_selected
        ev_codes.append(z_sel[mask].float().cpu())
        ev_ids.append(mask.nonzero()[:, 1].to(torch.int32).cpu())
        counts = mask.sum(dim=1).cpu()
        m = int(counts.max()) if counts.numel() else 0
        if m + 1 > count_hist.numel():
            grown = torch.zeros(m + 1, dtype=torch.long)
            grown[: count_hist.numel()] = count_hist
            count_hist = grown
        count_hist += torch.bincount(counts, minlength=count_hist.numel())
        block_events += mask.sum(dim=0).cpu()
        mean_acc += x.double().sum(dim=0).cpu()
        n_tokens += x.shape[0]

    codes = torch.cat(ev_codes) if ev_codes else torch.zeros(0, b)
    ids = torch.cat(ev_ids).long() if ev_ids else torch.zeros(0, dtype=torch.long)
    included = block_events >= spec.floor

    # Canonical orientation: batched second moments via index_add, eigh
    # descending, sign so the active-mean projection is >= 0.
    M = torch.zeros(G, b, b, dtype=torch.float64)
    M.index_add_(0, ids, torch.einsum("ni,nj->nij", codes.double(), codes.double()))
    mean_code = torch.zeros(G, b, dtype=torch.float64)
    mean_code.index_add_(0, ids, codes.double())
    denom = block_events.clamp_min(1).double()
    M /= denom.view(-1, 1, 1)
    mean_code /= denom.view(-1, 1)
    eye = torch.eye(b, dtype=torch.float64)
    safe_M = torch.where(included.view(-1, 1, 1), M, eye.expand(G, b, b))
    _, evecs = torch.linalg.eigh(safe_M)  # ascending
    evecs = evecs.flip(-1)  # descending eigenvalue order, columns
    R = evecs.transpose(1, 2)  # rows: z_can = R @ z
    sign = torch.sign(torch.einsum("gij,gj->gi", R, mean_code))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    R = R * sign.unsqueeze(-1)

    # Clip quantiles per canonical coordinate.
    codes_can = torch.einsum("nij,nj->ni", R[ids].float(), codes)
    lo = torch.zeros(G, b)
    hi = torch.ones(G, b)
    order = torch.argsort(ids)
    sorted_ids = ids[order]
    sorted_codes = codes_can[order]
    boundaries = torch.searchsorted(
        sorted_ids, torch.arange(G + 1, dtype=torch.long)
    )
    qs = torch.tensor([spec.clip_lo, spec.clip_hi])
    for g in included.nonzero().flatten().tolist():
        seg = sorted_codes[boundaries[g] : boundaries[g + 1]]
        ql = torch.quantile(seg, qs, dim=0)
        lo[g], hi[g] = ql[0], ql[1]

    # Count model: add-one smoothing over [0, K_max].
    k_max_obs = int(count_hist.nonzero().max()) if count_hist.sum() else 0
    K_max = 2 * k_max_obs + 8
    smoothed = torch.ones(K_max + 1, dtype=torch.float64)
    smoothed[: count_hist.numel()] += count_hist.double()
    count_log2p = torch.log2(smoothed / smoothed.sum())

    # Bernoulli support-entropy sensitivity model.
    p_hat = (block_events.double() + 1.0) / (n_tokens + 2.0)
    bernoulli_log2p = torch.log2(p_hat)
    bernoulli_log2q = torch.log2(1.0 - p_hat)

    return Codec(
        spec=spec,
        included=included,
        rotation=R.float(),
        lo=lo,
        hi=hi,
        count_log2p=count_log2p,
        bernoulli_log2p=bernoulli_log2p.float(),
        bernoulli_log2q=bernoulli_log2q.float(),
        calib_events=block_events,
        calib_tokens=n_tokens,
        calib_mean=mean_acc / max(n_tokens, 1),
        meta={
            "n_blocks": G, "block_dim": b, "k_max_obs": k_max_obs,
            "n_excluded": int((~included).sum()),
            "excluded_calib_event_share": float(
                block_events[~included].sum() / max(1, block_events.sum())
            ),
        },
    )


@torch.no_grad()
def evaluate_rd(
    model,
    codec: Codec,
    batches,
    *,
    row_len: int,
    device: str = "cpu",
) -> dict:
    """Eval pass: per-q distortion through quantized codes + rates, with
    per-sequence accumulators and a sequence bootstrap. `batches` must be
    the SEQUENTIAL eval stream (stored order) so `row_len`-token groups
    are genuine stored sequences."""
    spec = codec.spec
    b = model.cfg.block_dim
    S = model.cfg.n_sites
    inc = codec.included.to(device)
    R = codec.rotation.to(device)
    mu = codec.calib_mean.to(device).float()  # [S, d] calib-fit centering
    log2_1mq_total = float(codec.bernoulli_log2q.double().sum())

    rows_err = {q: [] for q in spec.qs}  # per-row sq err (pooled over sites)
    rows_err_site = {q: [] for q in spec.qs}  # per-row [S]
    rows_tot: list[float] = []
    rows_tot_site: list[torch.Tensor] = []
    rows_bits_sup: list[float] = []
    rows_bits_bern: list[float] = []
    rows_counts: list[float] = []
    rows_n: list[int] = []

    # Rolling row assembly across batch boundaries.
    pend = {
        "err": {q: torch.zeros(S, dtype=torch.float64) for q in spec.qs},
        "tot": torch.zeros(S, dtype=torch.float64),
        "sup": 0.0, "bern": 0.0, "cnt": 0.0, "n": 0,
    }

    def close_row():
        for q in spec.qs:
            rows_err[q].append(float(pend["err"][q].sum()))
            rows_err_site[q].append(pend["err"][q].clone())
            pend["err"][q].zero_()
        rows_tot.append(float(pend["tot"].sum()))
        rows_tot_site.append(pend["tot"].clone())
        pend["tot"].zero_()
        rows_bits_sup.append(pend["sup"])
        rows_bits_bern.append(pend["bern"])
        rows_counts.append(pend["cnt"])
        rows_n.append(pend["n"])
        pend["sup"] = pend["bern"] = pend["cnt"] = 0.0
        pend["n"] = 0

    excluded_events = 0
    total_events = 0
    for x in batches:
        x = x.to(device, torch.float32)
        z = model.encode(x)
        raw_mask = model.select(z, mode="threshold")
        mask = raw_mask & inc.unsqueeze(0)
        excluded_events += int((raw_mask & ~inc.unsqueeze(0)).sum())
        total_events += int(raw_mask.sum())
        z_sel = z * mask.unsqueeze(-1)
        z_can = torch.einsum("gij,ngj->ngi", R, z_sel)
        counts = mask.sum(dim=1)

        # Rates (support enumerative + Bernoulli sensitivity), per token.
        sup_bits = (
            -codec.log2_count_prob(counts.cpu()).double()
            + _log2_binom(codec.n_included, counts.cpu())
        )
        act_p = (codec.bernoulli_log2p.to(device) * mask.float()).sum(dim=1).double()
        act_q = (codec.bernoulli_log2q.to(device) * mask.float()).sum(dim=1).double()
        bern_bits = -(act_p.cpu() + (log2_1mq_total - act_q.cpu()))

        err_site = {}
        for q in spec.qs:
            z_hat_can = codec.quantize(z_can, q) * mask.unsqueeze(-1)
            z_hat = torch.einsum("gji,ngj->ngi", R, z_hat_can)  # R^T back
            xhat = model.decode(z_hat)
            err_site[q] = (x - xhat).double().pow(2).sum(dim=2).cpu()  # [n, S]
        tot_site = (x - mu).double().pow(2).sum(dim=2).cpu()  # [n, S]

        # Assemble rows.
        i = 0
        n = x.shape[0]
        while i < n:
            take = min(row_len - pend["n"], n - i)
            sl = slice(i, i + take)
            for q in spec.qs:
                pend["err"][q] += err_site[q][sl].sum(dim=0)
            pend["tot"] += tot_site[sl].sum(dim=0)
            pend["sup"] += float(sup_bits[sl].sum())
            pend["bern"] += float(bern_bits[sl].sum())
            pend["cnt"] += float(counts[sl].sum())
            pend["n"] += take
            i += take
            if pend["n"] == row_len:
                close_row()
    if pend["n"]:
        close_row()

    n_rows = len(rows_tot)
    tot = torch.tensor(rows_tot, dtype=torch.float64)
    tot_site_t = torch.stack(rows_tot_site)  # [rows, S]
    n_tok = torch.tensor(rows_n, dtype=torch.float64)
    sup = torch.tensor(rows_bits_sup, dtype=torch.float64)
    bern = torch.tensor(rows_bits_bern, dtype=torch.float64)
    cnt = torch.tensor(rows_counts, dtype=torch.float64)
    gen = torch.Generator().manual_seed(spec.bootstrap_seed)

    def boot(num: torch.Tensor, den: torch.Tensor) -> tuple[float, float]:
        idx = torch.randint(0, n_rows, (spec.n_bootstrap, n_rows), generator=gen)
        r = num[idx].sum(dim=1) / den[idx].sum(dim=1)
        lo_v, hi_v = torch.quantile(r, torch.tensor([0.025, 0.975], dtype=r.dtype))
        return float(lo_v), float(hi_v)

    results: dict = {
        "n_rows": n_rows,
        "n_tokens": int(n_tok.sum()),
        "row_len": row_len,
        "avg_count": float(cnt.sum() / n_tok.sum()),
        "codec_meta": dict(codec.meta),
        "eval_excluded_event_share": excluded_events / max(1, total_events),
        "support_bits_per_token": float(sup.sum() / n_tok.sum()),
        "support_bits_ci95": boot(sup, n_tok),
        "bernoulli_bits_per_token": float(bern.sum() / n_tok.sum()),
        "bernoulli_bits_ci95": boot(bern, n_tok),
        "points": {},
    }
    for q in spec.qs:
        err = torch.tensor(rows_err[q], dtype=torch.float64)
        err_site_t = torch.stack(rows_err_site[q])  # [rows, S]
        amp_bits = float(q * b * cnt.sum() / n_tok.sum())
        fvu_lo, fvu_hi = boot(err, tot)
        rate = results["support_bits_per_token"] + amp_bits
        results["points"][str(q)] = {
            "q": q,
            "fvu_pooled": float(err.sum() / tot.sum()),
            "fvu_ci95": [fvu_lo, fvu_hi],
            "fvu_per_site": (
                err_site_t.sum(dim=0) / tot_site_t.sum(dim=0)
            ).tolist(),
            "amplitude_bits_per_token": amp_bits,
            "rate_bits_per_token": rate,
            "rate_bits_bernoulli": results["bernoulli_bits_per_token"] + amp_bits,
        }
    return results
