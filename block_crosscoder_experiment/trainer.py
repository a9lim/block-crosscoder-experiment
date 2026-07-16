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
                    long batch horizon; same selection mechanics.
    "fel"           Fel App. D — no dead set; the next s_aux runner-up
                    blocks (by main-code norm, unselected) explain the
                    residual with the *main* code; alpha = 1/s_aux.

The data interface is any iterable of whitened [B, S, d] batches —
synthetic tensors in Phase -1, the disk store in Phase 1. The trainer owns
no data randomness: permutation seeds live with the data source (design:
the store's shuffle seed is recorded and shared by BSC and baseline).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.optim.lr_scheduler import LambdaLR

from .gram import gram_residual, retract_, site_frobenius_shares
from .model import BlockCrosscoder, BSCConfig, BSCOutput, bsc_loss

__all__ = ["TrainConfig", "DeadTracker", "Trainer", "aux_loss", "tensor_batches"]

AUX_VARIANTS = ("none", "sasa", "long_horizon", "fel")


@dataclass
class TrainConfig:
    total_steps: int
    lr: float = 3e-4
    warmup_steps: int = 1000
    betas: tuple[float, float] = (0.9, 0.999)
    encoder_weight_decay: float = 0.0  # value is a Phase-0.9 calibration item
    retract_every: int = 1  # >1 is the documented throughput ablation (P16)
    optimizer: str = "auto"  # "adamw8bit" (CUDA) | "adamw" | "auto"
    forward_dtype: str = "bf16"  # "bf16" (production) | "fp32" (exact/dev)
    # AuxK (P8; s_aux/alpha/window/threshold are Phase -1/0.9 calibration
    # items — SASA's values are the starting spec).
    aux_variant: str = "sasa"
    s_aux: int = 256
    alpha_aux: float = 1.0  # SASA lambda_aux; the Fel arm overrides to 1/s_aux
    dead_threshold: float = 1e-4
    dead_window_batches: int = 100
    dead_horizon_batches: int = 500
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


class DeadTracker:
    """Ring buffer of per-batch activation counts, on device.

    Supports both dead criteria: SASA windowed frequency (counts over the
    last `window` batches divided by tokens seen) and long-horizon (zero
    activations over the last `horizon` batches). A block is never flagged
    dead before the relevant history is full — a fresh model has no dead
    blocks, only unobserved ones.
    """

    def __init__(self, n_blocks: int, capacity: int, device) -> None:
        self.counts = torch.zeros(capacity, n_blocks, device=device)
        self.tokens = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.filled = 0

    def update(self, mask: torch.Tensor) -> None:
        """mask: [B, G] bool from the training forward."""
        self.counts[self.ptr] = mask.sum(dim=0).float()
        self.tokens[self.ptr] = mask.shape[0]
        self.ptr = (self.ptr + 1) % self.counts.shape[0]
        self.filled = min(self.filled + 1, self.counts.shape[0])

    def _last(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        cap = self.counts.shape[0]
        idx = (self.ptr - 1 - torch.arange(n, device=self.counts.device)) % cap
        return self.counts[idx], self.tokens[idx]

    def frequency(self, window: int) -> torch.Tensor:
        """Per-block activation frequency over the last `window` batches."""
        n = min(window, self.filled)
        if n == 0:
            return torch.zeros_like(self.counts[0])
        counts, tokens = self._last(n)
        return counts.sum(dim=0) / tokens.sum().clamp_min(1.0)

    def dead(self, variant: str, *, threshold: float, window: int, horizon: int) -> torch.Tensor:
        """Bool [G]. All-False until the relevant history is full."""
        G = self.counts.shape[1]
        device = self.counts.device
        if variant == "sasa":
            if self.filled < window:
                return torch.zeros(G, dtype=torch.bool, device=device)
            return self.frequency(window) <= threshold
        if variant == "long_horizon":
            if self.filled < horizon:
                return torch.zeros(G, dtype=torch.bool, device=device)
            counts, _ = self._last(horizon)
            return counts.sum(dim=0) == 0
        raise ValueError(f"no dead criterion for variant {variant!r}")

    def state_dict(self) -> dict:
        return {
            "counts": self.counts,
            "tokens": self.tokens,
            "ptr": self.ptr,
            "filled": self.filled,
        }

    def load_state_dict(self, state: dict) -> None:
        self.counts.copy_(state["counts"])
        self.tokens.copy_(state["tokens"])
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
    residual = (x - out.xhat).detach()
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
    device = model.E.device
    if kind == "auto":
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        kind = "adamw8bit" if (device.type == "cuda" and has_bnb) else "adamw"
    groups = [
        {"params": [model.E], "weight_decay": cfg.encoder_weight_decay},
        {"params": [model.D, model.c], "weight_decay": 0.0},
    ]
    if kind == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=cfg.lr, betas=cfg.betas), kind
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas), kind
    raise ValueError(f"unknown optimizer {kind!r}")


def _warmup_cosine(cfg: TrainConfig):
    def factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
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


class Trainer:
    """Owns the master/forward-copy pair, the optimizer, the retraction
    schedule, dead-block tracking, and diagnostics logging."""

    def __init__(
        self,
        model: BlockCrosscoder,
        cfg: TrainConfig,
        *,
        log_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
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
        self.sched = LambdaLR(self.opt, _warmup_cosine(cfg))
        self.tracker = DeadTracker(
            model.cfg.n_blocks,
            max(cfg.dead_window_batches, cfg.dead_horizon_batches),
            device=model.E.device,
        )
        self.step_idx = 0
        self._k_final = float(model.cfg.k)  # annealing target
        self.ema_min_score: float | None = None  # diagnostic only (D10)
        self._prev_shares = site_frobenius_shares(self.master.D).detach().clone()
        self.history: list[dict] = []
        self._log_file = Path(log_path).open("a") if log_path is not None else None

    # -- one training step -------------------------------------------------

    def step(self, x: torch.Tensor) -> dict:
        cfg = self.cfg
        x = x.to(device=self.fwd.E.device, dtype=self.fwd.E.dtype)
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
        self.tracker.update(out.mask)

        l_aux = None
        if cfg.aux_variant != "none":
            dead = None
            if cfg.aux_variant in ("sasa", "long_horizon"):
                dead = self.tracker.dead(
                    cfg.aux_variant,
                    threshold=cfg.dead_threshold,
                    window=cfg.dead_window_batches,
                    horizon=cfg.dead_horizon_batches,
                )
            l_aux = aux_loss(self.fwd, x, out, cfg.aux_variant, dead, cfg.s_aux)
            if l_aux is not None:
                alpha = 1.0 / cfg.s_aux if cfg.aux_variant == "fel" else cfg.alpha_aux
                parts["aux"] = l_aux
                parts["total"] = parts["total"] + alpha * l_aux

        # Aux/main gradient norms are a pilot logging requirement; the aux
        # norm is measured exactly on log steps via a separate grad pass.
        grad_norm_aux = None
        trainable = [p for p in self.fwd.parameters() if p.requires_grad]
        if log_step and l_aux is not None:
            g = torch.autograd.grad(l_aux, trainable, retain_graph=True, allow_unused=True)
            grad_norm_aux = math.sqrt(
                sum(float(t.float().pow(2).sum()) for t in g if t is not None)
            )

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

        # The load-bearing ordering: step on master -> retract master ->
        # regenerate the forward copy -> measure the post-cast residual.
        self.opt.step()
        self.sched.step()
        floor_hits = 0
        if self.step_idx % cfg.retract_every == 0:
            floor_hits = retract_(self.master.D.data, eig_floor=self.master.cfg.eig_floor)
        if self.fwd is not self.master:
            with torch.no_grad():
                for m, f in zip(self.master.parameters(), self.fwd.parameters()):
                    f.copy_(m)
                self.fwd.theta.copy_(self.master.theta)

        selected = out.scores.detach()[out.mask]
        if selected.numel() > 0:
            batch_min = float(selected.min())
            self.ema_min_score = (
                batch_min
                if self.ema_min_score is None
                else cfg.ema_decay * self.ema_min_score + (1 - cfg.ema_decay) * batch_min
            )

        record = {
            "step": self.step_idx,
            "rec": float(parts["rec"].detach()),
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
            "gram_residual_master": float(gram_residual(self.master.D).max()),
            "dead_frac_window": float(
                (self.tracker.frequency(self.cfg.dead_window_batches) <= self.cfg.dead_threshold)
                .float()
                .mean()
            ),
        }
        if self.ema_min_score is not None:
            d["ema_min_score"] = self.ema_min_score
        if self.fwd is not self.master:
            d["gram_residual_postcast"] = float(gram_residual(self.fwd.D.float()).max())
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
            "model_cfg": asdict(self.master.cfg),
            "train_cfg": asdict(self.cfg),
            "optimizer_kind": self.optimizer_kind,
        }
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)  # atomic write-then-rename (D14)
        tmp.rename(path)

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, *, device: torch.device | str = "cpu"
    ) -> "Trainer":
        payload = torch.load(path, map_location=device, weights_only=True)
        model_cfg = BSCConfig(**payload["model_cfg"])
        cfg = TrainConfig(**{
            **payload["train_cfg"],
            "betas": tuple(payload["train_cfg"]["betas"]),
        })
        model = BlockCrosscoder(model_cfg).to(device)
        model.load_state_dict(payload["model"])
        trainer = cls(model, cfg)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.sched.load_state_dict(payload["scheduler"])
        trainer.tracker.load_state_dict(payload["tracker"])
        trainer.step_idx = payload["step_idx"]
        trainer._k_final = payload.get("k_final", float(model_cfg.k))
        trainer.ema_min_score = payload["ema_min_score"]
        if trainer.fwd is not trainer.master:
            with torch.no_grad():
                for m, f in zip(trainer.master.parameters(), trainer.fwd.parameters()):
                    f.copy_(m)
        trainer._prev_shares = site_frobenius_shares(trainer.master.D).detach().clone()
        return trainer
