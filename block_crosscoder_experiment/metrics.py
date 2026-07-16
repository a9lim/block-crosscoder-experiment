"""Recovery metrics for the Phase -1 harness (design v2.2, Phase -1).

Evaluation protocol, in the order the spec pins it:

1. **Match** learned to planted blocks by Hungarian assignment on the mean
   (over carrying sites) principal-angle overlap between *used* spans —
   the column spaces of K^{1/2} D^s where K is the active-code second
   moment. Never frames (parked capacity is unidentifiable), never raw
   code correlation (splitting would alias).
2. **Align** each matched pair by ONE global O(b) Procrustes rotation fit
   on the jointly-active codes — a single R per pair, shared by all sites.
   Per-site alignment is forbidden: it would falsely certify shared
   coordinates (D11).
3. **Score**: span overlap + principal angles, code R^2 after alignment,
   depth-profile share error (the lambda-veto metric), rank recovery from
   contribution spectra (participation ratio, 95%-energy rank), detection
   IoU, norm concentration (the hollow/thickened + bundle-null readout),
   and the Bhalla capture/shattering/dilution trio (P19): restricted R^2,
   support size, receptive-field spread.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from scipy.optimize import linear_sum_assignment

from .gram import site_frobenius_shares
from .model import BlockCrosscoder
from .synthetic import PlantedModel

__all__ = [
    "BlockRecovery",
    "RecoveryReport",
    "subspace_overlap",
    "procrustes",
    "participation_ratio",
    "energy_rank",
    "block_site_spans",
    "evaluate_recovery",
]


# -- primitives -------------------------------------------------------------


def psd_sqrt(K: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(K.float())
    return evecs @ torch.diag_embed(evals.clamp_min(0).sqrt()) @ evecs.mT


def block_site_spans(
    D_block: torch.Tensor, z_active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Used-span bases and contribution spectra for one block.

    D_block: [S, b, d]; z_active: [n, b] codes on active tokens.
    Returns (bases [S, d, b] with columns ordered by contribution,
    spectra [S, b] = eigenvalues of the per-site contribution second
    moment D^sT K D^s, via SVD of W = K^{1/2} D^s — no d x d work).
    """
    K = z_active.float().mT @ z_active.float() / max(1, z_active.shape[0])
    W = psd_sqrt(K) @ D_block.float()  # [S, b, d]
    _, sv, Vh = torch.linalg.svd(W, full_matrices=False)
    return Vh.mT, sv.pow(2)  # [S, d, b], [S, b]


def subspace_overlap(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Mean squared principal-angle cosine between span(P) and span(Q)
    (columns orthonormal), plus the angles in radians."""
    sv = torch.linalg.svdvals(P.float().mT @ Q.float()).clamp(0.0, 1.0)
    r = min(P.shape[1], Q.shape[1])
    return float(sv.pow(2).sum() / r), torch.acos(sv[:r])


def procrustes(
    z_hat: torch.Tensor, z_ref: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    """One global O(b) alignment: R minimizing ||z_hat - z_ref R^T||_F.

    Returns (R, r2, scale_ratio) with r2 = 1 - ||z_hat - z_ref R^T||^2 /
    ||z_hat||^2 (uncentered — a mean offset is a real recovery error) and
    scale_ratio = mean||z_hat|| / mean||z_ref|| as a diagnostic (both
    dictionaries are Gram-constrained, so faithful recovery has ratio ~1;
    the rotation deliberately absorbs no scale)."""
    z_hat, z_ref = z_hat.float(), z_ref.float()
    U, _, Vh = torch.linalg.svd(z_hat.mT @ z_ref)
    R = U @ Vh
    resid = z_hat - z_ref @ R.mT
    r2 = 1.0 - float(resid.pow(2).sum() / z_hat.pow(2).sum().clamp_min(1e-12))
    scale = float(
        z_hat.norm(dim=1).mean() / z_ref.norm(dim=1).mean().clamp_min(1e-12)
    )
    return R, r2, scale


def participation_ratio(evals: torch.Tensor) -> float:
    e = evals.float().clamp_min(0)
    return float(e.sum().pow(2) / e.pow(2).sum().clamp_min(1e-24))


def energy_rank(evals: torch.Tensor, q: float = 0.95) -> int:
    e = evals.float().clamp_min(0).sort(descending=True).values
    cum = e.cumsum(0) / e.sum().clamp_min(1e-24)
    return int((cum < q).sum().item()) + 1


def norm_cv(z_active: torch.Tensor) -> float:
    """Coefficient of variation of active code norms: ~0 for a hollow
    shell, large for full-support (gaussian / bundle) geometry."""
    n = z_active.float().norm(dim=1)
    return float(n.std() / n.mean().clamp_min(1e-12))


# -- report structures --------------------------------------------------------


@dataclass
class BlockRecovery:
    planted: int
    rank: int
    frequency: float
    matched: int | None  # learned block index, None if nothing eligible
    overlap: float = math.nan  # mean over carrying sites, squared-cosine
    mean_angle_deg: float = math.nan
    code_r2: float = math.nan  # after the single global Procrustes
    scale_ratio: float = math.nan
    share_error: float = math.nan  # max_s |share diff| — the lambda-veto metric
    rank_pr: float = math.nan  # participation ratio, mean over carrying sites
    rank_95: float = math.nan  # 95%-energy rank, mean over carrying sites
    detection_iou: float = math.nan
    norm_cv_planted: float = math.nan
    norm_cv_learned: float = math.nan
    capture_r2: float = math.nan  # Bhalla restricted R^2, matched block only
    capture_r2_support: float = math.nan  # restricted to the support set
    support_size: int = 0  # Bhalla shattering coordinate
    rf_spread: float = math.nan  # Bhalla dilution coordinate
    n_active_planted: int = 0
    n_active_matched: int = 0


@dataclass
class RecoveryReport:
    blocks: list[BlockRecovery]
    n_learned_eligible: int
    n_learned_dead: int  # below min_active on the eval sample
    n_learned_unmatched: int
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "blocks": [asdict(b) for b in self.blocks],
            "n_learned_eligible": self.n_learned_eligible,
            "n_learned_dead": self.n_learned_dead,
            "n_learned_unmatched": self.n_learned_unmatched,
            "config": self.config,
        }


# -- the evaluator ------------------------------------------------------------


@torch.no_grad()
def evaluate_recovery(
    truth: PlantedModel,
    learner: BlockCrosscoder,
    *,
    n_eval: int = 32768,
    seed: int = 0,
    mode: str = "topk",
    min_active: int = 50,
    min_joint: int = 30,
    associate_lift: float = 3.0,
    associate_floor: float = 0.05,
) -> RecoveryReport:
    """Sample eval data from the truth, run the learner, match, align, score.

    min_active is the R22 inclusion threshold scaled to harness size;
    learned blocks under it are counted as dead-on-eval and excluded.
    """
    batch = truth.sample(n_eval, seed=seed)
    device, dtype = learner.E.device, learner.E.dtype
    x = batch.x.to(device, dtype)
    out = learner(x, mode=mode)
    z_l = out.z_selected.float().cpu()
    mask_l = out.mask.cpu()
    S = truth.n_sites

    counts = mask_l.sum(dim=0)
    eligible = torch.nonzero(counts >= min_active).flatten().tolist()
    n_dead = int((counts < min_active).sum())

    # Used spans, both sides.
    learner_D = learner.D.detach().float().cpu()  # [S, Gl, b, d]
    spans_l, spectra_l = {}, {}
    for j in eligible:
        spans_l[j], spectra_l[j] = block_site_spans(
            learner_D[:, j], z_l[mask_l[:, j], j]
        )
    profiles = [
        spec.depth_profile or (1.0 / S,) * S for spec in truth.specs
    ]
    spans_p, active_p = {}, {}
    for g, spec in enumerate(truth.specs):
        active_p[g] = batch.active[:, g]
        z_pg = batch.z[active_p[g], g]
        if z_pg.shape[0] >= 2:
            spans_p[g], _ = block_site_spans(truth.D[:, g], z_pg)

    # Hungarian matching on mean carrying-site span overlap.
    matchable = [g for g in spans_p]
    cost = torch.ones(len(matchable), max(len(eligible), 1))
    for gi, g in enumerate(matchable):
        r = truth.specs[g].rank
        carrying = [s for s in range(S) if profiles[g][s] > 0]
        for ji, j in enumerate(eligible):
            ov = sum(
                subspace_overlap(spans_p[g][s, :, :r], spans_l[j][s, :, :r])[0]
                for s in carrying
            ) / len(carrying)
            cost[gi, ji] = 1.0 - ov
    match: dict[int, int] = {}
    if eligible and matchable:
        rows, cols = linear_sum_assignment(cost.numpy())
        match = {matchable[ri]: eligible[ci] for ri, ci in zip(rows, cols)}

    shares_l = site_frobenius_shares(learner_D)  # [S, Gl]

    blocks = []
    for g, spec in enumerate(truth.specs):
        rec = BlockRecovery(
            planted=g,
            rank=spec.rank,
            frequency=spec.frequency,
            matched=match.get(g),
            n_active_planted=int(active_p[g].sum()),
        )
        blocks.append(rec)
        if rec.matched is None or g not in spans_p:
            continue
        j = rec.matched
        r = spec.rank
        carrying = [s for s in range(S) if profiles[g][s] > 0]

        overlaps, angles = [], []
        for s in carrying:
            ov, ang = subspace_overlap(spans_p[g][s, :, :r], spans_l[j][s, :, :r])
            overlaps.append(ov)
            angles.append(ang.mean())
        rec.overlap = sum(overlaps) / len(overlaps)
        rec.mean_angle_deg = float(torch.stack(angles).mean() * 180 / math.pi)

        joint = active_p[g] & mask_l[:, j]
        if int(joint.sum()) >= min_joint:
            _, rec.code_r2, rec.scale_ratio = procrustes(
                z_l[joint, j], batch.z[joint, g]
            )
        rec.share_error = float(
            (shares_l[:, j] - torch.tensor(profiles[g])).abs().max()
        )
        rec.rank_pr = sum(participation_ratio(spectra_l[j][s]) for s in carrying) / len(carrying)
        rec.rank_95 = sum(energy_rank(spectra_l[j][s]) for s in carrying) / len(carrying)
        union = active_p[g] | mask_l[:, j]
        rec.detection_iou = float(joint.sum() / union.sum().clamp_min(1))
        rec.norm_cv_planted = norm_cv(batch.z[active_p[g], g])
        rec.norm_cv_learned = norm_cv(z_l[mask_l[:, j], j])
        rec.n_active_matched = int(counts[j])

        # Bhalla trio (P19). Association by conditional-rate lift over the
        # rate on planted-INACTIVE tokens — lift over the overall base rate
        # would be structurally unreachable for high-frequency blocks.
        cond_rate = mask_l[active_p[g]].float().mean(dim=0)  # [Gl]
        off_rate = mask_l[~active_p[g]].float().mean(dim=0)  # [Gl]
        associated = [
            jj
            for jj in eligible
            if cond_rate[jj] >= associate_floor
            and cond_rate[jj] >= associate_lift * float(off_rate[jj])
        ]
        rec.support_size = len(associated)
        if associated:
            rec.rf_spread = float(cond_rate[associated].mean())

        x_g = torch.einsum(
            "nb,sbd->nsd", batch.z[active_p[g], g], truth.D[:, g]
        )  # planted signal on its active tokens

        def restricted_r2(block_set: list[int]) -> float:
            keep = torch.zeros_like(z_l[active_p[g]])
            keep[:, block_set] = z_l[active_p[g]][:, block_set]
            xhat = learner.decode(
                keep.to(device, dtype), add_bias=False
            ).float().cpu()
            return 1.0 - float(
                (xhat - x_g).pow(2).sum() / x_g.pow(2).sum().clamp_min(1e-12)
            )

        rec.capture_r2 = restricted_r2([j])
        if associated:
            rec.capture_r2_support = restricted_r2(associated)

    return RecoveryReport(
        blocks=blocks,
        n_learned_eligible=len(eligible),
        n_learned_dead=n_dead,
        n_learned_unmatched=len(eligible) - len(set(match.values())),
        config={
            "n_eval": n_eval,
            "seed": seed,
            "mode": mode,
            "min_active": min_active,
            "min_joint": min_joint,
            "associate_lift": associate_lift,
            "associate_floor": associate_floor,
        },
    )
