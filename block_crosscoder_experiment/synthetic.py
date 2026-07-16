"""Gauge-correct synthetic ground truth for the Phase -1 harness (D11).

The planted truth is itself a Gram-constraint-satisfying block dictionary:
per block, D_g^s = sqrt(share_s) * V_g^s with V_g^s a random b x d
orthonormal-row frame, so sum_s D_g^s D_g^s^T = I_b holds *exactly* by
construction (no retraction needed) and per-site frames are independent —
cross-site frame rotation, the thing the BSC exists for, comes free.

Planted rank never lives in decoder rows (the constraint forces every frame
to rank b; parked capacity is unidentifiable and is not scored): intrinsic
coordinates u_g in R^r enter as block codes z_g = A_g u_g through a random
b x r Stiefel map, so rank r is carried by the *contribution operators*
D_g^s^T A_g, which is exactly where the H4 readouts look.

Geometries follow the Michaud regime split (P18): "shell" with
radial_spread=0 is the hollow manifold (a ring at r=2), radial_spread>0 is
the radially-thickened version of the same geometry; "gaussian" is generic
full-support linear structure — including the weakened bundle null (D11):
perfectly co-active scalars with full-dimensional joint support are
observationally equivalent to a block, so the learner may legitimately
bundle them; what must not happen is the coherence battery calling them a
*curved manifold*. Gate groups plant that null as separate rank-1 blocks
with coupled activation gates.

Everything is generated directly in whitened per-site coordinates — the
generator sits downstream of where whitening lives in the real pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, NamedTuple

import torch

__all__ = ["BlockSpec", "SyntheticBatch", "PlantedModel"]

GEOMETRIES = ("gaussian", "shell")


@dataclass(frozen=True)
class BlockSpec:
    """One planted block. depth_profile None = exactly flat (1/S per site);
    a one-hot profile is a site-specific decoy. gate_group couples
    activation gates across blocks (coupling 1.0 = perfect co-activation,
    the bundle null)."""

    rank: int
    frequency: float
    geometry: str = "gaussian"
    radial_spread: float = 0.0  # shell only; 0 = hollow
    spectrum: tuple[float, ...] | None = None  # gaussian only; len == rank
    depth_profile: tuple[float, ...] | None = None
    scale: float = 1.0
    gate_group: int | None = None
    gate_coupling: float = 1.0


class SyntheticBatch(NamedTuple):
    x: torch.Tensor  # [n, S, d] whitened activations
    z: torch.Tensor  # [n, G_true, b] planted block codes (A_g u_g)
    active: torch.Tensor  # [n, G_true] bool gates


class PlantedModel:
    """Planted dictionary + code distribution; the sampling side of the
    Phase -1 harness. Recovery metrics live in metrics.py."""

    def __init__(
        self,
        specs: list[BlockSpec],
        *,
        n_sites: int,
        d_model: int,
        block_dim: int,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.specs = list(specs)
        self.n_sites = n_sites
        self.d_model = d_model
        self.block_dim = block_dim
        self.noise_std = noise_std
        G, S, b, d = len(specs), n_sites, block_dim, d_model
        if b > d:
            raise ValueError("block_dim must not exceed d_model")

        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.D = torch.zeros(S, G, b, d)
        self.A: list[torch.Tensor] = []  # per block, [b, r] Stiefel
        for g, spec in enumerate(specs):
            self._validate(spec)
            profile = spec.depth_profile or (1.0 / S,) * S
            for s in range(S):
                if profile[s] == 0.0:
                    continue
                q, _ = torch.linalg.qr(torch.randn(d, b, generator=gen))
                self.D[s, g] = profile[s] ** 0.5 * q.T  # orthonormal rows
            qa, _ = torch.linalg.qr(torch.randn(b, spec.rank, generator=gen))
            self.A.append(qa)
        self._groups = sorted(
            {s.gate_group for s in specs if s.gate_group is not None}
        )

    def _validate(self, spec: BlockSpec) -> None:
        if not 1 <= spec.rank <= self.block_dim:
            raise ValueError(f"rank must be in [1, {self.block_dim}]")
        if spec.geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}")
        if spec.spectrum is not None and len(spec.spectrum) != spec.rank:
            raise ValueError("spectrum length must equal rank")
        if spec.depth_profile is not None:
            if len(spec.depth_profile) != self.n_sites:
                raise ValueError("depth_profile length must equal n_sites")
            if abs(sum(spec.depth_profile) - 1.0) > 1e-6:
                raise ValueError("depth_profile must sum to 1")
        if spec.gate_group is not None and not 0.0 <= spec.gate_coupling <= 1.0:
            raise ValueError("gate_coupling must be in [0, 1]")

    # -- sampling -----------------------------------------------------------

    def _gates(self, n: int, gen: torch.Generator) -> torch.Tensor:
        group_gate = {
            gid: torch.rand(n, generator=gen)
            < max(s.frequency for s in self.specs if s.gate_group == gid)
            for gid in self._groups
        }
        gates = torch.zeros(n, len(self.specs), dtype=torch.bool)
        for g, spec in enumerate(self.specs):
            own = torch.rand(n, generator=gen) < spec.frequency
            if spec.gate_group is None:
                gates[:, g] = own
            else:
                use_group = torch.rand(n, generator=gen) < spec.gate_coupling
                gates[:, g] = torch.where(use_group, group_gate[spec.gate_group], own)
        return gates

    def _intrinsic(self, spec: BlockSpec, n: int, gen: torch.Generator) -> torch.Tensor:
        """u in R^r per geometry. [n, r]"""
        r = spec.rank
        if spec.geometry == "gaussian":
            u = torch.randn(n, r, generator=gen)
            if spec.spectrum is not None:
                u = u * torch.tensor(spec.spectrum).sqrt()
            return u
        # shell: uniform direction; hollow (radius 1) or radially thickened.
        direction = torch.randn(n, r, generator=gen)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if spec.radial_spread == 0.0:
            return direction
        radius = 1.0 + spec.radial_spread * torch.randn(n, 1, generator=gen)
        return direction * radius.clamp_min(0.05)

    def sample(self, n: int, *, seed: int = 0) -> SyntheticBatch:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        G, b = len(self.specs), self.block_dim
        gates = self._gates(n, gen)
        z = torch.zeros(n, G, b)
        for g, spec in enumerate(self.specs):
            u = self._intrinsic(spec, n, gen) * spec.scale
            z[:, g] = (u @ self.A[g].T) * gates[:, g : g + 1]
        x = torch.einsum("ngb,sgbd->nsd", z, self.D)
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn(x.shape, generator=gen)
        return SyntheticBatch(x, z, gates)

    def batches(
        self, batch_size: int, n_batches: int | None = None, *, seed: int = 0
    ) -> Iterator[torch.Tensor]:
        """Stream of x batches for Trainer.fit; ground truth is re-derivable
        from the same seed via sample()."""
        i = 0
        while n_batches is None or i < n_batches:
            yield self.sample(batch_size, seed=seed + i).x
            i += 1

    # -- planted ground truth for metrics -----------------------------------

    def contribution_maps(self, g: int) -> torch.Tensor:
        """Per-site map from intrinsic u to the site contribution:
        C_g^s = D_g^s^T A_g, [S, d, r]. Its column space is the planted
        *used* span at site s; planted rank is its rank — never the frame's.
        """
        return torch.einsum("sbd,br->sdr", self.D[:, g], self.A[g])

    def planted_second_moment(self, g: int, *, n: int = 20000, seed: int = 0) -> torch.Tensor:
        """Empirical per-site contribution second moment
        E[c c^T | g active], [S, d, d] — the planted H4 readout target."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        spec = self.specs[g]
        u = self._intrinsic(spec, n, gen) * spec.scale
        c = torch.einsum("nr,sdr->snd", u, self.contribution_maps(g))
        return torch.einsum("snd,sne->sde", c, c) / n
