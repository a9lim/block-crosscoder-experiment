"""Trainer for the block-sparse crosscoder.

Implements the load-bearing step ordering (design v2.2, R8-R11):

    optimizer step on fp32 master -> retract master decoders ->
    regenerate bf16 forward copy -> log post-cast Gram residual

with 8-bit Adam moments (bitsandbytes, CUDA), encoder-only weight decay
(decoder decay is 0 — uniform shrinkage is undone by retraction and only
injects noise), 1k-step linear warmup + cosine decay, and the AuxK
dead-block machinery in its three comparison variants (P8):

    "sasa"          SASA App. C.1 — dead = windowed activation frequency
                    <= threshold; per-token top-s_aux dead blocks by
                    residual energy re-encode the frozen residual.
    "long_horizon"  the former v2.1 rule — dead = zero activations over a
                    long accepted-token horizon; same selection mechanics.
    "fel"           Fel-style runner-up AuxK — no dead set; the next
                    s_aux runner-up blocks (by main-code norm, unselected)
                    explain the residual with the *main* code;
                    alpha = 1/s_aux. NB a hybrid (F5): Fel App. D uses the
                    next-l runner-ups with alpha = 1/l where l is the MAIN
                    block sparsity — faithful only when s_aux = k.

The data interface is any iterable of declared-coordinate [B, S, d] batches —
synthetic tensors or a normalized/raw disk store. The trainer owns
no data randomness: permutation seeds live with the data source (design:
the store's shuffle seed is recorded and shared by BSC and baseline).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch.optim.lr_scheduler import LambdaLR

from .gram import gram_residual, retract_, site_frobenius_shares
from .model import BlockCrosscoder, BSCConfig, BSCOutput, bsc_loss

__all__ = [
    "TrainConfig",
    "DeadTracker",
    "Trainer",
    "aux_loss",
    "tensor_batches",
    "validate_run_binding",
]

AUX_VARIANTS = ("none", "sasa", "long_horizon", "fel")
CHECKPOINT_FREE_FLOOR_FRAC = 0.15  # same D14 floor as store.ShardWriter


def _payload_nbytes(obj) -> int:
    """Tensor bytes in a (nested) checkpoint payload, for the pre-write
    free-space check. Non-tensor leaves are negligible and counted as 0."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_payload_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_payload_nbytes(v) for v in obj)
    return 0


@dataclass
class TrainConfig:
    total_steps: int
    lr: float = 3e-4
    warmup_steps: int = 1000
    # "cosine" = linear warmup + cosine decay to 0 (current default);
    # "linear_fifth" = SASA B.3 — warmup, constant, then linear decay to 0
    # (our final-fifth reading of SASA's "over one-fifth of the training").
    # Both arms of the 0.9.5 lr ladder (F2).
    schedule: str = "cosine"
    betas: tuple[float, float] = (0.9, 0.999)
    encoder_weight_decay: float = 0.0  # value is an open calibration item (F2)
    retract_every: int = 1  # >1 is the documented throughput ablation (P16)
    optimizer: str = "auto"  # "adamw8bit" (CUDA) | "adamw" | "auto"
    forward_dtype: str = "bf16"  # "bf16" (production) | "fp32" (exact/dev)
    # AuxK follows the validated SASA-style recovery path.
    aux_variant: str = "sasa"
    s_aux: int = 256
    alpha_aux: float = 1.0  # SASA lambda_aux; the Fel arm overrides to 1/s_aux
    dead_threshold: float = 1e-4
    # Deadness is a property of token exposure, not optimizer-step count.
    # These defaults preserve the former 100/500-batch production windows at
    # B=4096 while remaining invariant to batch size and partial batches.
    dead_window_tokens: int = 409_600
    dead_horizon_tokens: int = 2_048_000
    # Loss-spike guard: batch-skip with
    # corroboration. Trigger = grad_norm > guard_factor x trailing median
    # AND rec > guard_loss_factor x trailing median, medians over the last
    # guard_window ACCEPTED steps (unarmed until the window fills). A
    # grad-only anomaly is logged as a near-miss and never skipped — AuxK
    # engagement legitimately moves gradient scale. Non-finite grad/loss
    # always skips. More than guard_max_consecutive skips in a row raises:
    # a run that needs sustained skipping is not at a stable operating
    # point, and the guard must not censor that evidence (skip rate is a
    # reported run gate).
    guard: bool = False
    guard_factor: float = 20.0
    guard_loss_factor: float = 5.0
    guard_window: int = 50
    guard_max_consecutive: int = 5
    guard_max_skip_rate: float = 1e-3
    # AuxK caps; the production CLI pins the gradient-ratio cap at 1.0.
    # aux_frac_cap: revived blocks/step <= ceil(frac x live dead-set size)
    # (dead-set variants only — fel has no dead set). aux_ratio_cap:
    # rescale alpha so the aux gradient norm never exceeds ratio x the
    # main-loss gradient norm (two extra grad passes per step where the
    # dead set is nonempty). alpha_aux < 1 is the third candidate and
    # needs no mechanism — it is already a config field.
    aux_frac_cap: float | None = None
    aux_ratio_cap: float | None = None
    # BatchTopK budget annealing (capture-sweep finding: budget ratio drives
    # the capture-vs-tiling basin). When set, k is interpolated linearly from
    # k_anneal_from to the model config's k over k_anneal_steps (default:
    # total_steps), then held.
    k_anneal_from: float | None = None
    k_anneal_steps: int | None = None
    # Diagnostics
    ema_decay: float = 0.99  # EMA of batch-min selected score (D10: diagnostic only)
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.aux_variant not in AUX_VARIANTS:
            raise ValueError(f"aux_variant must be one of {AUX_VARIANTS}")
        if self.forward_dtype not in ("bf16", "fp32"):
            raise ValueError("forward_dtype must be 'bf16' or 'fp32'")
        if self.schedule not in ("cosine", "linear_fifth"):
            raise ValueError("schedule must be 'cosine' or 'linear_fifth'")
        if self.s_aux <= 0:
            raise ValueError("s_aux must be positive")
        if self.dead_threshold < 0.0:
            raise ValueError("dead_threshold must be non-negative")
        if self.dead_window_tokens <= 0 or self.dead_horizon_tokens <= 0:
            raise ValueError("dead token windows must be positive")
        if not (0.0 <= self.guard_max_skip_rate <= 1.0):
            raise ValueError("guard_max_skip_rate must be in [0, 1]")


def validate_run_binding(
    actual: dict | None,
    expected: dict,
    *,
    keys: Sequence[str] | None = None,
) -> None:
    """Fail closed when a checkpoint is not bound to the expected run.

    Exact resume validation compares the whole canonical binding. Consumers
    such as codec evaluation may compare only the fields relevant to their
    store/gauge view. Legacy checkpoints deliberately fail whenever an
    expected binding is supplied.
    """
    if actual is None:
        raise ValueError("checkpoint has no run binding (legacy/unbound checkpoint)")
    names = list(keys) if keys is not None else sorted(set(actual) | set(expected))
    mismatches = {
        name: {"checkpoint": actual.get(name), "expected": expected.get(name)}
        for name in names
        if actual.get(name) != expected.get(name)
    }
    if mismatches:
        raise ValueError(
            "checkpoint run binding mismatch: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )


class DeadTracker:
    """Ring buffer of accepted-observation activation counts, on device.

    Supports both dead criteria: SASA windowed frequency (counts over the
    most recent whole observations covering at least `window_tokens`) and
    long-horizon (zero activations over `horizon_tokens`). A block is never
    flagged dead before enough accepted tokens have been observed — a fresh
    model has no dead blocks, only unobserved ones.

    ``capacity`` remains denominated in observation slots. When
    ``max_tokens`` is supplied, the ring grows rather than discarding history
    until it can cover that many tokens, so variable and partial batch sizes
    cannot silently shorten a token window.
    """

    def __init__(
        self,
        n_blocks: int,
        capacity: int,
        device,
        *,
        max_tokens: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.counts = torch.zeros(capacity, n_blocks, device=device)
        self.tokens = torch.zeros(capacity, device=device)
        self.max_tokens = max_tokens
        self.ptr = 0
        self.filled = 0

    def update(self, mask: torch.Tensor) -> None:
        """mask: [B, G] bool from the training forward."""
        if mask.ndim != 2 or mask.shape[1] != self.counts.shape[1]:
            raise ValueError(
                f"mask must have shape [B, {self.counts.shape[1]}], got "
                f"{tuple(mask.shape)}"
            )
        if mask.shape[0] <= 0:
            raise ValueError("dead-tracker observations must contain tokens")
        if self.filled == self.counts.shape[0] and self.max_tokens is not None:
            retained = float(self.tokens.sum() - self.tokens[self.ptr]) + mask.shape[0]
            if retained < self.max_tokens:
                self._grow()
        # The one-shot CUDA reduction requests a ~256 MiB workspace at
        # B=4096,G=8192, enough to cross the 4090 ceiling after optimizer
        # state is resident. Row chunks preserve the exact integer count in
        # fp32 (B is tiny relative to its exact range) with bounded workspace.
        total = torch.zeros(mask.shape[1], device=mask.device)
        for start in range(0, mask.shape[0], 256):
            total.add_(mask[start : start + 256].sum(dim=0, dtype=torch.float32))
        self.counts[self.ptr].copy_(total)
        self.tokens[self.ptr] = mask.shape[0]
        self.ptr = (self.ptr + 1) % self.counts.shape[0]
        self.filled = min(self.filled + 1, self.counts.shape[0])

    def _grow(self) -> None:
        """Double slot capacity while preserving oldest-to-newest order."""
        old_counts, old_tokens = self._last(self.filled)
        old_counts = old_counts.flip(0)
        old_tokens = old_tokens.flip(0)
        capacity = max(1, 2 * self.counts.shape[0])
        counts = self.counts.new_zeros((capacity, self.counts.shape[1]))
        tokens = self.tokens.new_zeros(capacity)
        counts[: self.filled].copy_(old_counts)
        tokens[: self.filled].copy_(old_tokens)
        self.counts = counts
        self.tokens = tokens
        self.ptr = self.filled

    def _last(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        cap = self.counts.shape[0]
        idx = (self.ptr - 1 - torch.arange(n, device=self.counts.device)) % cap
        return self.counts[idx], self.tokens[idx]

    def _recent_tokens(
        self, target_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Newest whole observations covering ``target_tokens`` when able."""
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.filled == 0:
            return self.counts[:0], self.tokens[:0]
        counts, tokens = self._last(self.filled)
        cumulative = tokens.cumsum(dim=0)
        n = min(self.filled, int((cumulative < target_tokens).sum().item()) + 1)
        return counts[:n], tokens[:n]

    @property
    def history_tokens(self) -> int:
        if self.filled == 0:
            return 0
        _, tokens = self._last(self.filled)
        return int(tokens.sum().item())

    def frequency(self, window_tokens: int) -> torch.Tensor:
        """Per-block frequency over the last ``window_tokens`` accepted tokens.

        The oldest included observation may cross the boundary; aggregate
        masks cannot split a batch after the fact, so its tokens are included
        whole and form the denominator.
        """
        counts, tokens = self._recent_tokens(window_tokens)
        if tokens.numel() == 0:
            return torch.zeros_like(self.counts[0])
        return counts.sum(dim=0) / tokens.sum().clamp_min(1.0)

    def dead(
        self,
        variant: str,
        *,
        threshold: float,
        window_tokens: int,
        horizon_tokens: int,
    ) -> torch.Tensor:
        """Bool [G]. All-False until the token-denominated history is full."""
        G = self.counts.shape[1]
        device = self.counts.device
        if variant == "sasa":
            if self.history_tokens < window_tokens:
                return torch.zeros(G, dtype=torch.bool, device=device)
            return self.frequency(window_tokens) <= threshold
        if variant == "long_horizon":
            if self.history_tokens < horizon_tokens:
                return torch.zeros(G, dtype=torch.bool, device=device)
            counts, _ = self._recent_tokens(horizon_tokens)
            return counts.sum(dim=0) == 0
        raise ValueError(f"no dead criterion for variant {variant!r}")

    def state_dict(self) -> dict:
        return {
            "counts": self.counts,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "ptr": self.ptr,
            "filled": self.filled,
        }

    def load_state_dict(self, state: dict) -> None:
        counts = state["counts"].to(self.counts.device)
        tokens = state["tokens"].to(self.tokens.device)
        if counts.shape != self.counts.shape:
            self.counts = counts.clone()
            self.tokens = tokens.clone()
        else:
            self.counts.copy_(counts)
            self.tokens.copy_(tokens)
        self.max_tokens = state.get("max_tokens", self.max_tokens)
        self.ptr = state["ptr"]
        self.filled = state["filled"]


def aux_loss(
    model: BlockCrosscoder,
    x: torch.Tensor,
    out: BSCOutput,
    variant: str,
    dead: torch.Tensor | None,
    s_aux: int,
) -> torch.Tensor | None:
    """L_aux under the same fp32 mean reduction as L_rec (R12).

    The residual is frozen (no gradient through it) in every variant.
    Returns None when the variant has nothing to train on this step.
    """
    B, G = out.scores.shape

    if variant == "fel":
        # Runner-up blocks by main-code norm among the unselected; the main
        # code (not a re-encoding) explains what the selected blocks missed.
        n_unselected = int(G - out.mask.sum(dim=1).max().item())
        keep = min(s_aux, n_unselected)
        if keep <= 0:
            return None
        p = out.scores.masked_fill(out.mask, float("-inf"))
        z_aux = out.z
    else:
        # SASA C.1 / long-horizon: re-encode the frozen residual through
        # dead blocks only; top s_aux dead blocks by residual energy.
        assert dead is not None
        n_dead = int(dead.sum().item())
        keep = min(s_aux, n_dead)
        if keep <= 0:
            return None
    residual = (x - out.xhat).detach()
    if variant != "fel":
        z_aux = model.encode(residual) * dead.view(1, -1, 1)
        p = model.scores(z_aux).masked_fill(~dead.view(1, -1), float("-inf"))

    top = p.topk(keep, dim=1, sorted=False).indices
    mask = torch.zeros(B, G, dtype=torch.bool, device=p.device)
    mask.scatter_(1, top, True)
    rhat = model.decode(z_aux * mask.unsqueeze(-1), add_bias=False)
    return (rhat.float() - residual.float()).pow(2).mean()


def build_optimizer(model: BlockCrosscoder, cfg: TrainConfig) -> tuple[torch.optim.Optimizer, str]:
    """AdamW over [encoders (decayed), decoders+bias (decay 0)] param groups.

    8-bit moments via bitsandbytes on CUDA (design: fp32 master weights,
    8-bit moments); plain AdamW elsewhere.
    """
    kind = cfg.optimizer
    device = next(model.parameters()).device
    if kind == "auto":
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        kind = "adamw8bit" if (device.type == "cuda" and has_bnb) else "adamw"
    no_decay_ids = {id(model.D), id(model.c)}
    encoder_params = [
        p for p in model.parameters() if id(p) not in no_decay_ids
    ]
    groups = []
    if encoder_params:
        groups.append(
            {"params": encoder_params, "weight_decay": cfg.encoder_weight_decay}
        )
    groups.append({"params": [model.D, model.c], "weight_decay": 0.0})
    if kind == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=cfg.lr, betas=cfg.betas), kind
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas), kind
    raise ValueError(f"unknown optimizer {kind!r}")


def _lr_factor(cfg: TrainConfig):
    def factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.schedule == "linear_fifth":
            # SASA B.3: constant after warmup, linear decay over the
            # final fifth of training.
            decay_start = cfg.total_steps * 4 // 5
            if step < decay_start:
                return 1.0
            span = max(1, cfg.total_steps - decay_start)
            return max(0.0, 1.0 - (step - decay_start) / span)
        span = max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def tensor_batches(
    x: torch.Tensor, batch_size: int, *, seed: int = 0, epochs: int | None = None
) -> Iterator[torch.Tensor]:
    """Shuffled minibatches from an in-memory [N, S, d] tensor, reshuffled
    each epoch. The seed is the caller's to record (design: the permutation
    seed is shared by BSC and baseline runs)."""
    n = x.shape[0]
    gen = torch.Generator().manual_seed(seed)
    epoch = 0
    while epochs is None or epoch < epochs:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - batch_size + 1, batch_size):
            yield x[perm[i : i + batch_size]]
        epoch += 1


def _floating_tensors(obj) -> Iterator[torch.Tensor]:
    if torch.is_tensor(obj):
        if obj.is_floating_point() or obj.is_complex():
            yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _floating_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _floating_tensors(value)


def _all_finite(obj) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in _floating_tensors(obj))


def _project_decoder_(model: BlockCrosscoder) -> int:
    project = getattr(model, "project_decoder_", None)
    if project is not None:
        result = project()
        return 0 if result is None else int(result)
    return retract_(model.D.data, eig_floor=model.cfg.eig_floor)


def _constraint_residual(model: BlockCrosscoder) -> float | None:
    measure = getattr(model, "decoder_constraint_residual", None)
    if measure is not None:
        value = measure()
        return float(value.max() if torch.is_tensor(value) else value)
    constraint = getattr(model.cfg, "decoder_constraint", "gram")
    if constraint == "gram":
        return float(gram_residual(model.D.float()).max())
    return None


class Trainer:
    """Owns the master/forward-copy pair, the optimizer, the retraction
    schedule, dead-block tracking, and diagnostics logging."""

    def __init__(
        self,
        model: BlockCrosscoder,
        cfg: TrainConfig,
        *,
        log_path: str | Path | None = None,
        run_binding: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_binding = copy.deepcopy(run_binding)
        self.master = model  # fp32 masters
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise TypeError("master model must be fp32")
        if cfg.forward_dtype == "bf16":
            self.fwd = copy.deepcopy(model).to(torch.bfloat16)
            for p in self.master.parameters():
                p.requires_grad_(False)
        else:
            self.fwd = model
        self.opt, self.optimizer_kind = build_optimizer(self.master, cfg)
        self.sched = LambdaLR(self.opt, _lr_factor(cfg))
        self.tracker = DeadTracker(
            model.cfg.n_blocks,
            capacity=128,
            device=next(model.parameters()).device,
            max_tokens=max(cfg.dead_window_tokens, cfg.dead_horizon_tokens),
        )
        self.step_idx = 0
        self._k_final = float(model.cfg.k)  # annealing target
        self.ema_min_score: float | None = None  # diagnostic only (D10)
        # Guard state (E2). Histories hold ACCEPTED-step values only, so a
        # spike never poisons its own reference median.
        self._guard_grad_hist: list[float] = []
        self._guard_rec_hist: list[float] = []
        self._guard_consecutive = 0
        self.skipped_steps = 0
        self.guard_events: list[dict] = []
        self._prev_shares = site_frobenius_shares(self.master.D).detach().clone()
        self.history: list[dict] = []
        self._log_file = Path(log_path).open("a") if log_path is not None else None

    # -- one training step -------------------------------------------------

    def step(self, x: torch.Tensor) -> dict:
        cfg = self.cfg
        fwd_param = next(self.fwd.parameters())
        x = x.to(device=fwd_param.device, dtype=fwd_param.dtype)
        log_step = self.step_idx % cfg.log_every == 0

        k_now = None
        if cfg.k_anneal_from is not None:
            span = max(1, cfg.k_anneal_steps or cfg.total_steps)
            frac = min(1.0, self.step_idx / span)
            k_now = cfg.k_anneal_from + (self._k_final - cfg.k_anneal_from) * frac
            self.master.cfg.k = k_now
            if self.fwd is not self.master:
                self.fwd.cfg.k = k_now

        out = self.fwd(x)
        parts = bsc_loss(out, x, self.fwd)

        trainable = [p for p in self.fwd.parameters() if p.requires_grad]

        def norm_of(loss: torch.Tensor) -> float:
            g = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
            return math.sqrt(sum(float(t.float().pow(2).sum()) for t in g if t is not None))

        l_aux = None
        alpha_eff = None
        grad_norm_aux = None
        s_aux_eff = None
        if cfg.aux_variant != "none":
            dead = None
            s_aux_eff = cfg.s_aux
            if cfg.aux_variant in ("sasa", "long_horizon"):
                dead = self.tracker.dead(
                    cfg.aux_variant,
                    threshold=cfg.dead_threshold,
                    window_tokens=cfg.dead_window_tokens,
                    horizon_tokens=cfg.dead_horizon_tokens,
                )
                if cfg.aux_frac_cap is not None:
                    # E3 frac cap: revival budget proportional to the live
                    # dead set, never the full s_aux slam.
                    n_dead = int(dead.sum().item())
                    s_aux_eff = min(
                        cfg.s_aux, max(1, math.ceil(cfg.aux_frac_cap * n_dead))
                    )
            l_aux = aux_loss(self.fwd, x, out, cfg.aux_variant, dead, s_aux_eff)
            if l_aux is not None:
                alpha = 1.0 / cfg.s_aux if cfg.aux_variant == "fel" else cfg.alpha_aux
                alpha_eff = alpha
                if cfg.aux_ratio_cap is not None:
                    # E3 ratio cap: bound the aux update energy relative to
                    # the main-loss gradient, exactly (two extra grad
                    # passes; the cascade signature is grad_norm_aux
                    # 3e-4 -> 108 while the main loss barely moves).
                    grad_norm_aux = norm_of(l_aux)
                    norm_main = norm_of(parts["total"])
                    if alpha * grad_norm_aux > cfg.aux_ratio_cap * norm_main:
                        alpha_eff = (
                            cfg.aux_ratio_cap * norm_main / max(grad_norm_aux, 1e-12)
                        )
                parts["aux"] = l_aux
                parts["total"] = parts["total"] + alpha_eff * l_aux

        # Aux/main gradient norms are a pilot logging requirement; the aux
        # norm is measured exactly on log steps via a separate grad pass
        # (already available on every step when the ratio cap is active).
        if log_step and l_aux is not None and grad_norm_aux is None:
            grad_norm_aux = norm_of(l_aux)

        self.opt.zero_grad(set_to_none=True)
        if self.fwd is not self.master:
            for p in self.fwd.parameters():
                p.grad = None
        parts["total"].backward()

        if self.fwd is not self.master:
            for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                if f.grad is None:
                    continue
                if m.grad is None:
                    m.grad = f.grad.detach().float()
                else:
                    m.grad.copy_(f.grad.detach())
        grad_norm = math.sqrt(
            sum(
                float(p.grad.float().pow(2).sum())
                for p in self.master.parameters()
                if p.grad is not None
            )
        )

        state_finite = _all_finite(tuple(self.master.parameters())) and _all_finite(
            self.opt.state
        )

        # E2 guard: decide before any state is touched. Non-finite always
        # skips; a corroborated spike (grad AND loss anomalous vs the
        # accepted-step trailing medians) skips; a grad-only anomaly is a
        # near-miss, logged and applied.
        rec_val = float(parts["rec"].detach())
        skip_reason = None
        pre_step_nonfinite = not (
            math.isfinite(grad_norm)
            and math.isfinite(rec_val)
            and math.isfinite(float(parts["total"].detach()))
            and state_finite
        )
        if pre_step_nonfinite and not cfg.guard:
            raise RuntimeError(
                "non-finite loss/gradient/parameter/optimizer state with spike guard disabled"
            )
        if cfg.guard:
            if pre_step_nonfinite:
                skip_reason = "nonfinite"
            elif len(self._guard_grad_hist) >= cfg.guard_window:
                g_med = statistics.median(self._guard_grad_hist)
                r_med = statistics.median(self._guard_rec_hist)
                grad_anom = grad_norm > cfg.guard_factor * g_med
                loss_anom = rec_val > cfg.guard_loss_factor * r_med
                if grad_anom and loss_anom:
                    skip_reason = "spike"
                elif grad_anom:
                    skip_reason = "near_miss"  # logged below, not skipped
        skipped = skip_reason in ("nonfinite", "spike")
        if cfg.guard and skip_reason is not None:
            event = {
                "step": self.step_idx,
                "reason": skip_reason,
                "skipped": skipped,
                "grad_norm": grad_norm,
                "rec": rec_val,
                "lr": self.sched.get_last_lr()[0],
                # Cheap deterministic batch identity for the postmortem
                # (the step-1600 lesson: spikes can be data-driven).
                "batch_hash": f"{x.shape[0]}x{float(x.float().sum()):.6e}",
            }
            self.guard_events.append(event)
            if self._log_file is not None:
                self._log_file.write(json.dumps({"guard_event": event}) + "\n")
                self._log_file.flush()

        # The load-bearing ordering: step on master -> retract master ->
        # regenerate the forward copy -> measure the post-cast residual.
        # A skipped step drops the update but still advances the schedule,
        # so run length and the lr curve stay deterministic.
        floor_hits = 0
        if skipped:
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.skipped_steps += 1
            self._guard_consecutive += 1
            if self._guard_consecutive > cfg.guard_max_consecutive:
                raise RuntimeError(
                    f"spike guard skipped {self._guard_consecutive} consecutive "
                    f"steps at step {self.step_idx} (lr "
                    f"{self.sched.get_last_lr()[0]:.2e}) — this operating point "
                    "is not stable; the guard refuses to censor it (E2)"
                )
        else:
            self.opt.step()
            if not (
                _all_finite(tuple(self.master.parameters()))
                and _all_finite(self.opt.state)
            ):
                self.opt.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "optimizer produced non-finite parameter/state; refusing to "
                    "continue (reload the last atomic checkpoint)"
                )
            self.sched.step()
            if self.step_idx % cfg.retract_every == 0:
                floor_hits = _project_decoder_(self.master)
            if not _all_finite(tuple(self.master.parameters())):
                raise RuntimeError(
                    "decoder projection produced non-finite parameters; refusing "
                    "to continue (reload the last atomic checkpoint)"
                )
            if self.fwd is not self.master:
                with torch.no_grad():
                    for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                        f.copy_(m)
                    self.fwd.theta.copy_(self.master.theta)
            # Dead-frequency history is accepted-step state. A guarded poison
            # batch must not influence which blocks receive future AuxK.
            self.tracker.update(out.mask)
            self._guard_consecutive = 0
            if cfg.guard and math.isfinite(grad_norm) and math.isfinite(rec_val):
                self._guard_grad_hist.append(grad_norm)
                self._guard_rec_hist.append(rec_val)
                if len(self._guard_grad_hist) > cfg.guard_window:
                    self._guard_grad_hist.pop(0)
                    self._guard_rec_hist.pop(0)

        selected = out.scores.detach()[out.mask]
        if selected.numel() > 0 and not skipped:
            batch_min = float(selected.min())
            self.ema_min_score = (
                batch_min
                if self.ema_min_score is None
                else cfg.ema_decay * self.ema_min_score + (1 - cfg.ema_decay) * batch_min
            )

        record = {
            "step": self.step_idx,
            "rec": rec_val,
            "total": float(parts["total"].detach()),
            "lr": self.sched.get_last_lr()[0],
            "grad_norm": grad_norm,
            "floor_hits": floor_hits,
        }
        if "rank" in parts:
            record["rank"] = float(parts["rank"].detach())
        if k_now is not None:
            record["k"] = k_now
        if l_aux is not None:
            record["aux"] = float(l_aux.detach())
        if grad_norm_aux is not None:
            record["grad_norm_aux"] = grad_norm_aux
        if alpha_eff is not None and cfg.aux_ratio_cap is not None:
            record["alpha_aux_eff"] = alpha_eff
        if s_aux_eff is not None and cfg.aux_frac_cap is not None:
            record["s_aux_eff"] = s_aux_eff
        if skip_reason is not None:
            record["skip_reason"] = skip_reason
            record["skipped"] = skipped
        if log_step:
            record.update(self._diagnostics())
            self.history.append(record)
            if self._log_file is not None:
                self._log_file.write(json.dumps(record) + "\n")
                self._log_file.flush()
        self.step_idx += 1
        return record

    @torch.no_grad()
    def _diagnostics(self) -> dict:
        d = {
            "dead_frac_window": float(
                (
                    self.tracker.frequency(self.cfg.dead_window_tokens)
                    <= self.cfg.dead_threshold
                )
                .float()
                .mean()
            ),
        }
        master_residual = _constraint_residual(self.master)
        if master_residual is not None:
            d["decoder_constraint_residual_master"] = master_residual
            # Compatibility key for existing Gram-constrained reports.
            if getattr(self.master.cfg, "decoder_constraint", "gram") == "gram":
                d["gram_residual_master"] = master_residual
        if self.ema_min_score is not None:
            d["ema_min_score"] = self.ema_min_score
        if self.fwd is not self.master:
            postcast_residual = _constraint_residual(self.fwd)
            if postcast_residual is not None:
                d["decoder_constraint_residual_postcast"] = postcast_residual
                if getattr(self.fwd.cfg, "decoder_constraint", "gram") == "gram":
                    d["gram_residual_postcast"] = postcast_residual
        shares = site_frobenius_shares(self.master.D).detach()
        d["share_jump"] = float((shares - self._prev_shares).abs().max())
        self._prev_shares = shares.clone()
        return d

    # -- driving loop -------------------------------------------------------

    def fit(self, batches: Iterable[torch.Tensor]) -> list[dict]:
        for x in batches:
            if self.step_idx >= self.cfg.total_steps:
                break
            self.step(x)
        if self._log_file is not None:
            self._log_file.flush()
        return self.history

    def validate_run_gates(self) -> None:
        """Enforce terminal run-quality gates before evaluation/publication."""
        if not self.cfg.guard:
            return
        skip_rate = self.skipped_steps / max(1, self.step_idx)
        if skip_rate > self.cfg.guard_max_skip_rate:
            raise RuntimeError(
                f"spike-guard skip rate {skip_rate:.4%} exceeds the "
                f"{self.cfg.guard_max_skip_rate:.4%} run gate "
                f"({self.skipped_steps}/{self.step_idx}); refusing evaluation"
            )

    # -- checkpointing (exercised by the Phase -1 battery and the pilot) ----

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "model": self.master.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "tracker": self.tracker.state_dict(),
            "step_idx": self.step_idx,
            "k_final": self._k_final,  # cfg.k may be mid-anneal at save time
            "ema_min_score": self.ema_min_score,
            # Guard state (E2): histories feed future skip decisions, so
            # resume must restore them for bit-compatible continuation.
            "guard": {
                "grad_hist": list(self._guard_grad_hist),
                "rec_hist": list(self._guard_rec_hist),
                "consecutive": self._guard_consecutive,
                "skipped_steps": self.skipped_steps,
                "events": list(self.guard_events),
            },
            "model_cfg": asdict(self.master.cfg),
            "train_cfg": asdict(self.cfg),
            "optimizer_kind": self.optimizer_kind,
            "run_binding": copy.deepcopy(self.run_binding),
        }
        path = Path(path)
        # Free-space check that aborts *before* the write (D14) — the atomic
        # replace transiently doubles the footprint, and the freed space of
        # an overwritten checkpoint is deliberately not credited.
        nbytes = _payload_nbytes(payload)
        usage = shutil.disk_usage(path.parent if path.parent.name else ".")
        if usage.free - nbytes < CHECKPOINT_FREE_FLOOR_FRAC * usage.total:
            raise RuntimeError(
                f"checkpoint write would breach the "
                f"{CHECKPOINT_FREE_FLOOR_FRAC:.0%} free-space floor "
                f"({usage.free / 1e9:.1f} GB free, payload {nbytes / 1e9:.2f} GB)"
            )
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)  # atomic write-then-rename (D14)
        tmp.rename(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        expected_binding: dict | None = None,
    ) -> "Trainer":
        payload = torch.load(path, map_location=device, weights_only=True)
        stored_binding = payload.get("run_binding")
        if stored_binding is not None:
            bound_model_cfg = dict(payload["model_cfg"])
            # The live cfg.k is temporarily annealed; the binding records the
            # run's final target, which is serialized separately for resume.
            bound_model_cfg["k"] = payload.get("k_final", bound_model_cfg["k"])
            validate_run_binding(
                stored_binding,
                {
                    "model_cfg": bound_model_cfg,
                    "train_cfg": payload["train_cfg"],
                },
                keys=("model_cfg", "train_cfg"),
            )
        if expected_binding is not None:
            validate_run_binding(stored_binding, expected_binding)
        model_cfg = BSCConfig(**payload["model_cfg"])
        cfg = TrainConfig(**{
            **payload["train_cfg"],
            "betas": tuple(payload["train_cfg"]["betas"]),
        })
        model = BlockCrosscoder(model_cfg).to(device)
        model.load_state_dict(payload["model"])
        trainer = cls(model, cfg, run_binding=stored_binding)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.sched.load_state_dict(payload["scheduler"])
        trainer.tracker.load_state_dict(payload["tracker"])
        trainer.step_idx = payload["step_idx"]
        trainer._k_final = payload.get("k_final", float(model_cfg.k))
        trainer.ema_min_score = payload["ema_min_score"]
        guard_state = payload.get("guard")  # absent in pre-E2 checkpoints
        if guard_state is not None:
            trainer._guard_grad_hist = list(guard_state["grad_hist"])
            trainer._guard_rec_hist = list(guard_state["rec_hist"])
            trainer._guard_consecutive = guard_state["consecutive"]
            trainer.skipped_steps = guard_state["skipped_steps"]
            trainer.guard_events = list(guard_state["events"])
        if trainer.fwd is not trainer.master:
            with torch.no_grad():
                for m, f in zip(trainer.master.parameters(), trainer.fwd.parameters()):
                    f.copy_(m)
        trainer._prev_shares = site_frobenius_shares(trainer.master.D).detach().clone()
        return trainer
