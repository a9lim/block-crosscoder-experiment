"""Disk-backed whitened activation store (design v2.3.1, D6/D8/D9/D14).

Three pieces, shared by the Phase-0.9 rehearsal and the Phase-1 store:

- ``WhitenerAccumulator`` / ``Whitener`` — the *training-side harvest-fit*
  whitener (not saklas's consumer-side neutral-fit ``LayerWhitener``; the
  two are never interchangeable). Per site: mean and covariance accumulated
  in fp64 from fp32 batches, ridge per the saklas convention
  (λ_s = mean-diag(Σ̂_s) × DEFAULT_RIDGE_SCALE), eigendecomposition in
  fp64, frozen W_s = (Σ_s + λ_s I)^{-1/2}. Immutable once fit: the exact
  μ, W, ridge, site list, and source manifest are hashed, and that hash
  rides in every shard header; mismatches are rejected at load.
- ``ShardWriter`` — atomic safetensors shards ([token, site, d] bf16,
  sequence-contiguous), whitener hash + content checksum in each header,
  free-space abort *before* every write, per-shard finiteness/zero-row
  audit at write time. fp16 is forbidden in this path.
- ``StoreReader`` — the only sanctioned access pattern: per-epoch
  shard-level shuffle, contiguous chunk reads into a RAM shuffle buffer,
  batches mixed within the buffer, permutation seed recorded (and shared
  by BSC and baseline runs). Token-random mmap access is deliberately
  not implemented. ``sequential_batches`` streams a split in stored order
  for eval (the eval store is never assumed RAM-resident).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch

__all__ = [
    "DEFAULT_RIDGE_SCALE",
    "Whitener",
    "WhitenerAccumulator",
    "ShardWriter",
    "StoreReader",
    "prefetch_batches",
]

DEFAULT_RIDGE_SCALE = 1.0  # saklas mahalanobis.py convention, pinned by design
FORBIDDEN_DTYPES = (torch.float16,)
STORE_DTYPE = torch.bfloat16
MANIFEST_NAME = "split.json"


class WhitenerAccumulator:
    """fp64 sufficient statistics for the per-site whitener.

    Batches arrive as [n, S, d] (any float dtype except fp16); statistics
    are accumulated per batch in fp64 — batch-granular pairwise
    accumulation per D9. Covariance GEMMs run in fp64, which sidesteps
    TF32 entirely.
    """

    def __init__(self, n_sites: int, d_model: int, device: torch.device | str = "cpu") -> None:
        self.n = 0
        self.sum = torch.zeros(n_sites, d_model, dtype=torch.float64, device=device)
        self.outer = torch.zeros(
            n_sites, d_model, d_model, dtype=torch.float64, device=device
        )

    def update(self, x: torch.Tensor) -> None:
        """x: [n, S, d] raw activations."""
        if x.dtype in FORBIDDEN_DTYPES:
            raise TypeError("fp16 is forbidden in the harvest/store path")
        x64 = x.to(device=self.sum.device, dtype=torch.float64)
        self.n += x64.shape[0]
        self.sum += x64.sum(dim=0)
        self.outer += torch.einsum("nsd,nse->sde", x64, x64)

    def merge(self, other: "WhitenerAccumulator") -> "WhitenerAccumulator":
        out = WhitenerAccumulator(self.sum.shape[0], self.sum.shape[1], self.sum.device)
        out.n = self.n + other.n
        out.sum = self.sum + other.sum.to(self.sum.device)
        out.outer = self.outer + other.outer.to(self.outer.device)
        return out

    def finalize(
        self,
        *,
        sites: Sequence[int],
        meta: dict,
        ridge_scale: float = DEFAULT_RIDGE_SCALE,
    ) -> "Whitener":
        """Eigendecompose in fp64 (on CPU) and freeze W_s = (Σ+λI)^{-1/2}."""
        if self.n < 2:
            raise ValueError("whitener needs at least 2 tokens")
        mean = (self.sum / self.n).cpu()
        cov = self.outer.cpu() / self.n - torch.einsum("sd,se->sde", mean, mean)
        d = mean.shape[1]
        ridge = cov.diagonal(dim1=1, dim2=2).mean(dim=1) * ridge_scale  # [S]
        W = torch.zeros_like(cov)
        eigs = torch.zeros(mean.shape[0], d, dtype=torch.float64)
        for s in range(mean.shape[0]):
            e, V = torch.linalg.eigh(cov[s] + ridge[s] * torch.eye(d, dtype=torch.float64))
            eigs[s] = e
            W[s] = (V * e.clamp_min(1e-12).rsqrt()) @ V.T
        return Whitener(
            mean=mean.float(),
            W=W.float(),
            ridge=ridge.float(),
            eigenvalues=eigs.float(),
            sites=tuple(int(s) for s in sites),
            n_fit_tokens=self.n,
            meta=dict(meta),
        )


@dataclass
class Whitener:
    """Frozen per-site whitening map x̃^s = W_s (x^s − μ_s)."""

    mean: torch.Tensor  # [S, d] fp32
    W: torch.Tensor  # [S, d, d] fp32
    ridge: torch.Tensor  # [S] fp32
    eigenvalues: torch.Tensor  # [S, d] fp32, regularized-covariance spectrum
    sites: tuple[int, ...]
    n_fit_tokens: int
    meta: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        """sha256 over μ, W, ridge, sites, and the source manifest."""
        h = hashlib.sha256()
        for t in (self.mean, self.W, self.ridge):
            h.update(t.contiguous().to(torch.float32).numpy().tobytes())
        h.update(json.dumps([self.sites, self.meta], sort_keys=True).encode())
        return h.hexdigest()

    def site_rms_scalars(self) -> torch.Tensor:
        """Per-site scalar RMS renormalization after shrinkage whitening
        (F7 decision arm, design v2.3.2): the whitened per-dim variance
        prediction is (e_j − λ_s)/e_j for e = eig(Σ+λI), so scaling site s
        by 1/sqrt(mean_j retained_j) restores ~unit mean per-dim power —
        directional rogue-dim suppression kept, equal total site power
        restored. Returns [S] fp32.
        """
        e = self.eigenvalues.double()
        lam = self.ridge.double().unsqueeze(1)
        retained = ((e - lam) / e).clamp_min(0.0).mean(dim=1)
        return retained.clamp_min(1e-12).rsqrt().float()

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n, S, d] raw -> whitened, computed in fp32."""
        if x.dtype in FORBIDDEN_DTYPES:
            raise TypeError("fp16 is forbidden in the harvest/store path")
        mean = self.mean.to(x.device)
        W = self.W.to(x.device)
        return torch.einsum("sde,nse->nsd", W, x.float() - mean)

    def unapply(self, xw: torch.Tensor) -> torch.Tensor:
        """Whitened -> raw (fp32): x = W^{-1} x̃ + μ. Export egress only."""
        Winv = torch.linalg.inv(self.W.double()).float().to(xw.device)
        return torch.einsum("sde,nse->nsd", Winv, xw.float()) + self.mean.to(xw.device)

    def save(self, path: str | Path) -> None:
        payload = {
            "mean": self.mean,
            "W": self.W,
            "ridge": self.ridge,
            "eigenvalues": self.eigenvalues,
            "sites": list(self.sites),
            "n_fit_tokens": self.n_fit_tokens,
            "meta": self.meta,
            "hash": self.hash,
        }
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.rename(path)

    @classmethod
    def load(cls, path: str | Path) -> "Whitener":
        p = torch.load(path, map_location="cpu", weights_only=True)
        w = cls(
            mean=p["mean"],
            W=p["W"],
            ridge=p["ridge"],
            eigenvalues=p["eigenvalues"],
            sites=tuple(p["sites"]),
            n_fit_tokens=p["n_fit_tokens"],
            meta=p["meta"],
        )
        if w.hash != p["hash"]:
            raise ValueError(f"whitener hash mismatch in {path} — file corrupted?")
        return w


class ShardWriter:
    """Sequence-contiguous whitened-bf16 shards with audited atomic writes.

    Layout: ``root/<split>/shard_00000.safetensors`` + a ``split.json``
    manifest. Every shard's safetensors metadata carries the whitener
    hash, token count, site list, and a sha256 content checksum.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        whitener_hash: str,
        sites: Sequence[int],
        d_model: int,
        meta: dict | None = None,
        tokens_per_shard: int = 150_000,
        free_space_floor_frac: float = 0.15,
        max_zero_row_frac: float = 1e-4,
    ) -> None:
        self.dir = Path(root) / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.whitener_hash = whitener_hash
        self.sites = tuple(int(s) for s in sites)
        self.d_model = d_model
        self.meta = dict(meta or {})
        self.tokens_per_shard = tokens_per_shard
        self.free_floor = free_space_floor_frac
        self.max_zero_row_frac = max_zero_row_frac
        self._buffer: list[torch.Tensor] = []
        self._buffered = 0
        self.shards: list[dict] = []

    def add(self, x: torch.Tensor) -> None:
        """x: [n, S, d] whitened, cast to bf16 here. CPU tensors expected."""
        if x.dtype in FORBIDDEN_DTYPES:
            raise TypeError("fp16 is forbidden in the harvest/store path")
        if x.shape[1] != len(self.sites) or x.shape[2] != self.d_model:
            raise ValueError(f"shape {tuple(x.shape)} does not match store config")
        self._buffer.append(x.to(device="cpu", dtype=STORE_DTYPE))
        self._buffered += x.shape[0]
        while self._buffered >= self.tokens_per_shard:
            self._flush(self.tokens_per_shard)

    def _flush(self, n_tokens: int) -> None:
        chunk = torch.cat(self._buffer, dim=0)
        out, rest = chunk[:n_tokens], chunk[n_tokens:]
        self._buffer = [rest] if rest.shape[0] else []
        self._buffered = rest.shape[0]
        self._write(out)

    def _write(self, acts: torch.Tensor) -> None:
        from safetensors.torch import save_file

        # Audit before bytes hit disk: all-finite, near-zero rows bounded.
        if not torch.isfinite(acts.float()).all():
            raise ValueError("non-finite activations reached the shard writer")
        zero_rows = (acts.float().abs().sum(dim=(1, 2)) == 0).float().mean()
        if float(zero_rows) > self.max_zero_row_frac:
            raise ValueError(
                f"zero-row fraction {float(zero_rows):.2e} exceeds "
                f"{self.max_zero_row_frac:.0e} — suspect the capture path"
            )
        nbytes = acts.numel() * acts.element_size()
        usage = shutil.disk_usage(self.dir)
        if usage.free - nbytes < self.free_floor * usage.total:
            raise RuntimeError(
                f"write would breach the {self.free_floor:.0%} free-space floor "
                f"({usage.free / 1e9:.1f} GB free, shard {nbytes / 1e9:.2f} GB)"
            )
        idx = len(self.shards)
        checksum = hashlib.sha256(
            acts.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        header = {
            "whitener_hash": self.whitener_hash,
            "split": self.split,
            "shard_index": str(idx),
            "n_tokens": str(acts.shape[0]),
            "sites": json.dumps(list(self.sites)),
            "d_model": str(self.d_model),
            "dtype": "bfloat16",
            "content_sha256": checksum,
            "meta": json.dumps(self.meta, sort_keys=True),
        }
        path = self.dir / f"shard_{idx:05d}.safetensors"
        tmp = path.with_suffix(".tmp")
        save_file({"acts": acts.contiguous()}, tmp, metadata=header)
        tmp.rename(path)
        self.shards.append({"file": path.name, "n_tokens": int(acts.shape[0])})

    def close(self) -> dict:
        """Flush the remainder and write the split manifest."""
        if self._buffered:
            self._flush(self._buffered)
        manifest = {
            "split": self.split,
            "whitener_hash": self.whitener_hash,
            "sites": list(self.sites),
            "d_model": self.d_model,
            "n_tokens": sum(s["n_tokens"] for s in self.shards),
            "shards": self.shards,
            "meta": self.meta,
        }
        tmp = self.dir / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.rename(self.dir / MANIFEST_NAME)
        return manifest


def prefetch_batches(
    it: Iterator[torch.Tensor], depth: int = 4
) -> Iterator[torch.Tensor]:
    """E5 (runbook-phase099 tranche 1): drive an I/O-bound batch iterator
    from a daemon thread, holding up to ``depth`` batches ahead of the
    consumer. Order-preserving, so determinism is untouched; worker
    exceptions are re-raised at the consumption point. The 0.9 rehearsal
    measured 55-70% data-wait on the training loop — this overlaps shard
    reads with GPU steps.

    If the consumer abandons the generator early (e.g. total_steps hit),
    the worker thread parks on a full queue holding ~depth batches + one
    shard until process exit — acceptable for one-training-per-process
    scripts, not for long-lived services.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=depth)
    end = object()

    def worker() -> None:
        try:
            for x in it:
                q.put(x)
            q.put(end)
        except BaseException as e:  # noqa: BLE001 — re-raised consumer-side
            q.put(e)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is end:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class StoreReader:
    """Sequential buffered-shuffle reads from a written split.

    Per epoch: seeded shard-order permutation; contiguous chunk reads fill
    a ``buffer_tokens`` RAM buffer; the buffer is permuted and emitted as
    [batch, S, d] batches (bf16, CPU); a sub-batch remainder carries into
    the next fill. The seed is the caller's to record — design: BSC and
    baseline share it verbatim.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        expected_whitener_hash: str | None = None,
        sites: Sequence[int] | None = None,
    ) -> None:
        self.dir = Path(root) / split
        manifest_path = self.dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        self.whitener_hash = self.manifest["whitener_hash"]
        if (
            expected_whitener_hash is not None
            and self.whitener_hash != expected_whitener_hash
        ):
            raise ValueError(
                f"store {self.dir} was written under whitener "
                f"{self.whitener_hash[:12]}…, expected {expected_whitener_hash[:12]}…"
            )
        self.n_tokens = self.manifest["n_tokens"]
        stored_sites = list(self.manifest["sites"])
        # E4 site-subset view (runbook-phase099 tranche 1): `sites` selects a
        # subset of the stored site axis by layer number, sliced AFTER shard
        # load so generator consumption (shard order, buffer permutations)
        # is byte-identical to the full-width read at the same seed — the
        # factorial's matched-data guarantee: a single-site cell sees exactly
        # the joint run's token stream, sliced. Stored order is preserved;
        # reordering is refused rather than silently permuting frames.
        if sites is None:
            self.sites = tuple(stored_sites)
            self._site_sel: torch.Tensor | None = None
        else:
            req = [int(s) for s in sites]
            missing = [s for s in req if s not in stored_sites]
            if missing:
                raise ValueError(f"sites {missing} not in store (has {stored_sites})")
            if len(set(req)) != len(req):
                raise ValueError(f"duplicate sites in {req}")
            idx = [stored_sites.index(s) for s in req]
            if idx != sorted(idx):
                raise ValueError(
                    f"sites {req} not in stored order {stored_sites} — "
                    "reordering the site axis is not supported"
                )
            self.sites = tuple(req)
            self._site_sel = torch.tensor(idx, dtype=torch.long)
        self.n_sites = len(self.sites)
        self.d_model = self.manifest["d_model"]

    def _subset(self, acts: torch.Tensor) -> torch.Tensor:
        if self._site_sel is None:
            return acts
        return acts.index_select(1, self._site_sel)

    def _shard_tokens(self, name: str, *, verify: bool = False) -> torch.Tensor:
        from safetensors import safe_open

        path = self.dir / name
        with safe_open(path, framework="pt", device="cpu") as f:
            header = f.metadata()
            if header["whitener_hash"] != self.whitener_hash:
                raise ValueError(f"whitener hash mismatch in shard {path}")
            acts = f.get_tensor("acts")
        if verify:
            checksum = hashlib.sha256(
                acts.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if checksum != header["content_sha256"]:
                raise ValueError(f"content checksum mismatch in shard {path}")
        return acts

    def verify(self) -> int:
        """Re-hash every shard against its header. Returns tokens verified."""
        total = 0
        for s in self.manifest["shards"]:
            total += self._shard_tokens(s["file"], verify=True).shape[0]
        if total != self.n_tokens:
            raise ValueError(
                f"manifest claims {self.n_tokens} tokens, shards hold {total}"
            )
        return total

    def sequential_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Stored-order stream, never RAM-resident beyond one shard (eval)."""
        carry: torch.Tensor | None = None
        for s in self.manifest["shards"]:
            acts = self._subset(self._shard_tokens(s["file"]))
            if carry is not None:
                acts = torch.cat([carry, acts], dim=0)
            n_full = acts.shape[0] // batch_size * batch_size
            for i in range(0, n_full, batch_size):
                yield acts[i : i + batch_size]
            carry = acts[n_full:] if acts.shape[0] > n_full else None
        if carry is not None and carry.shape[0]:
            yield carry

    def shuffled_batches(
        self,
        batch_size: int,
        *,
        seed: int,
        epochs: int | None = None,
        buffer_tokens: int = 131_072,
    ) -> Iterator[torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        epoch = 0
        while epochs is None or epoch < epochs:
            order = torch.randperm(len(self.manifest["shards"]), generator=gen)
            buffer: list[torch.Tensor] = []
            buffered = 0
            for shard_idx in order.tolist():
                acts = self._subset(
                    self._shard_tokens(self.manifest["shards"][shard_idx]["file"])
                )
                buffer.append(acts)
                buffered += acts.shape[0]
                while buffered >= buffer_tokens:
                    chunk = torch.cat(buffer, dim=0)
                    take, rest = chunk[:buffer_tokens], chunk[buffer_tokens:]
                    buffer = [rest] if rest.shape[0] else []
                    buffered = rest.shape[0]
                    perm = torch.randperm(take.shape[0], generator=gen)
                    take = take[perm]
                    n_full = take.shape[0] // batch_size * batch_size
                    for i in range(0, n_full, batch_size):
                        yield take[i : i + batch_size]
                    tail = take[n_full:]
                    if tail.shape[0]:
                        buffer.insert(0, tail)
                        buffered += tail.shape[0]
            # End of epoch: emit whatever remains (shuffled, last partial
            # batch dropped — batch shapes stay constant for the trainer).
            if buffered:
                chunk = torch.cat(buffer, dim=0)
                perm = torch.randperm(chunk.shape[0], generator=gen)
                chunk = chunk[perm]
                for i in range(0, chunk.shape[0] - batch_size + 1, batch_size):
                    yield chunk[i : i + batch_size]
            epoch += 1
