"""Sparse SAE-code harvest + store for the Phase-0 ring hunt.

Harvest once (model out of the loop afterwards), analyze many times: the
battery's access pattern is "all tokens where any cluster member fires",
so codes are stored feature-indexed (CSC) in shards. Values stay float32 —
fp16 is banned in the harvest/store path (workspace rule; gemma-3
late-layer channels overflow it), and code magnitudes ride on activation
norms.

BOS positions are dropped at harvest: they carry no class information and
their outlier norms distort every downstream PCA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import torch

__all__ = ["CodeStore", "harvest_codes", "pack_token_rows"]

_MANIFEST = "manifest.json"


class CodeStore:
    """Feature-indexed sparse code shards on disk.

    Layout: <root>/manifest.json + shard_%04d.pt, each shard a dict with
    CSC arrays (ccol (F+1,), row (nnz,), val (nnz,) f32) for the battery's
    feature-major access AND CSR arrays (crow, col, val) for the
    co-activation branch's token-major chunking, plus token_ids (T_s,).
    Token indices are shard-local; readers re-base to the global stream.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest = json.loads((self.root / _MANIFEST).read_text())
        self.n_features: int = manifest["n_features"]
        self.n_tokens: int = manifest["n_tokens"]
        self.meta: dict = manifest
        self._shards = [self.root / s for s in manifest["shards"]]
        self._csc: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def load_csc(self, device: torch.device | str = "cpu") -> None:
        """Consolidate all shards into one in-memory CSC (ccol, row, val).

        One-time cost that makes per-cluster selection O(members), not
        O(shard loads) — the affinity/ranking passes call select thousands
        of times.
        """
        cols, rows, vals = [], [], []
        base = 0
        for path in self._shards:
            shard = torch.load(path, weights_only=True)
            ccol = shard["ccol"]
            cols.append(
                torch.repeat_interleave(
                    torch.arange(self.n_features, dtype=torch.int32), ccol.diff()
                )
            )
            rows.append(shard["row"].to(torch.int64) + base)
            vals.append(shard["val"])
            base += int(shard["token_ids"].shape[0])
        col = torch.cat(cols).to(device)
        row = torch.cat(rows).to(device)
        val = torch.cat(vals).to(device)
        order = torch.argsort(col.to(torch.int64), stable=True)
        col, row, val = col[order], row[order], val[order]
        counts = torch.bincount(col.to(torch.int64), minlength=self.n_features)
        ccol = torch.zeros(self.n_features + 1, dtype=torch.int64, device=device)
        ccol[1:] = counts.cumsum(0)
        self._csc = (ccol, row, val)

    # -- writing -----------------------------------------------------------

    @staticmethod
    def open_writer(
        root: str | Path, n_features: int, meta: dict | None = None
    ) -> "_CodeStoreWriter":
        return _CodeStoreWriter(Path(root), n_features, meta or {})

    # -- reading -----------------------------------------------------------

    def token_ids(self) -> torch.Tensor:
        return torch.cat([torch.load(s, weights_only=True)["token_ids"] for s in self._shards])

    def firing_counts(self) -> torch.Tensor:
        counts = torch.zeros(self.n_features, dtype=torch.long)
        for path in self._shards:
            shard = torch.load(path, weights_only=True)
            counts += shard["ccol"].diff()
        return counts

    def select_members(
        self, members: torch.Tensor, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Dense (n_tokens, |members|) code submatrix on `device`."""
        if self._csc is not None:
            ccol, row, val = self._csc
            out = torch.zeros(self.n_tokens, members.shape[0], device=device)
            for j, f in enumerate(members.tolist()):
                lo, hi = int(ccol[f]), int(ccol[f + 1])
                if hi > lo:
                    out[row[lo:hi].to(device=device, dtype=torch.long), j] = val[
                        lo:hi
                    ].to(device=device, dtype=out.dtype)
            return out
        members = members.cpu()
        out = torch.zeros(self.n_tokens, members.shape[0], device=device)
        base = 0
        for path in self._shards:
            shard = torch.load(path, weights_only=True)
            ccol, row, val = shard["ccol"], shard["row"], shard["val"]
            for j, f in enumerate(members.tolist()):
                lo, hi = int(ccol[f]), int(ccol[f + 1])
                if hi > lo:
                    idx = (base + row[lo:hi]).to(device=device, dtype=torch.long)
                    out[idx, j] = val[lo:hi].to(device=device, dtype=out.dtype)
            base += int(shard["token_ids"].shape[0])
        return out

    def member_row_union(self, members: torch.Tensor) -> torch.Tensor:
        """Sorted unique token rows where any member fires. Needs load_csc()."""
        if self._csc is None:
            raise RuntimeError("member_row_union requires load_csc() first")
        ccol, row, _ = self._csc
        slices = [row[int(ccol[f]) : int(ccol[f + 1])] for f in members.tolist()]
        if not slices:
            return row.new_empty(0)
        return torch.cat(slices).unique()

    def member_firing_counts(self, members: torch.Tensor) -> torch.Tensor:
        """(n_tokens,) count of firing members per token. Consolidates on
        first use if load_csc() hasn't run."""
        if self._csc is None:
            self.load_csc()
        ccol, row, _ = self._csc
        slices = [row[int(ccol[f]) : int(ccol[f + 1])] for f in members.tolist()]
        if not slices:
            return torch.zeros(self.n_tokens, dtype=torch.long, device=row.device)
        return torch.bincount(
            torch.cat(slices).to(torch.int64), minlength=self.n_tokens
        )

    def select_member_rows(
        self,
        members: torch.Tensor,
        rows: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Dense (|rows|, |members|) submatrix restricted to the given token
        rows, in their order. Needs load_csc().

        The gate-then-densify path: full-width select_members on a
        production store is n_tokens × |members| (35 GB for the 2192-member
        spectral blob — the scan OOM of 2026-07-16, repeated per null
        draw); gating and subsampling on sparse counts first bounds this
        at max_tokens × |members|.
        """
        if self._csc is None:
            self.load_csc()
        ccol, row, val = self._csc
        out = torch.zeros(rows.shape[0], members.shape[0], device=device)
        if rows.numel() == 0 or members.numel() == 0:
            return out
        rows = rows.to(device=row.device, dtype=torch.int64)
        order = torch.argsort(rows)
        sorted_rows = rows[order]
        for j, f in enumerate(members.tolist()):
            lo, hi = int(ccol[f]), int(ccol[f + 1])
            if hi == lo:
                continue
            r = row[lo:hi].to(torch.int64)
            pos = torch.searchsorted(sorted_rows, r).clamp_(
                max=sorted_rows.shape[0] - 1
            )
            hit = sorted_rows[pos] == r
            if hit.any():
                out[order[pos[hit]].to(device=device, dtype=torch.long), j] = val[
                    lo:hi
                ][hit].to(device=device, dtype=out.dtype)
        return out

    def iter_dense_chunks(
        self, chunk: int = 65536, device: torch.device | str = "cpu"
    ) -> Iterator[torch.Tensor]:
        """Token-major dense (≤chunk, F) blocks — the co-activation path."""
        for path in self._shards:
            shard = torch.load(path, weights_only=True)
            crow, col, val = shard["crow"], shard["col"], shard["val_csr"]
            t_s = int(shard["token_ids"].shape[0])
            for start in range(0, t_s, chunk):
                stop = min(start + chunk, t_s)
                lo, hi = int(crow[start]), int(crow[stop])
                dense = torch.zeros(stop - start, self.n_features, device=device)
                rows = torch.repeat_interleave(
                    torch.arange(stop - start, device=device),
                    (crow[start + 1 : stop + 1] - crow[start:stop]).to(device),
                )
                dense[rows, col[lo:hi].to(device=device, dtype=torch.long)] = val[
                    lo:hi
                ].to(device=device, dtype=dense.dtype)
                yield dense


class _CodeStoreWriter:
    def __init__(self, root: Path, n_features: int, meta: dict):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.n_features = n_features
        self.meta = meta
        self.shards: list[str] = []
        self.n_tokens = 0
        self.nnz = 0

    def add_shard(self, codes: torch.Tensor, token_ids: torch.Tensor) -> None:
        """codes: dense (T_s, F) nonneg; token_ids: (T_s,). Small inputs only
        — production harvests must use add_shard_coo (a dense shard at
        store scale is ~100 GB; that OOM has been paid for once already)."""
        if codes.shape[1] != self.n_features:
            raise ValueError(f"expected F={self.n_features}, got {codes.shape[1]}")
        nz = codes.nonzero(as_tuple=False)
        self.add_shard_coo(
            nz[:, 0], nz[:, 1], codes[nz[:, 0], nz[:, 1]], token_ids
        )

    def add_shard_coo(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        vals: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        """Assemble one shard from COO triplets — never densifies."""
        t_s = int(token_ids.shape[0])
        rows = rows.to(torch.int64).cpu()
        cols = cols.to(torch.int64).cpu()
        vals = vals.to(torch.float32).cpu()

        by_row = torch.argsort(rows, stable=True)
        crow = torch.zeros(t_s + 1, dtype=torch.int64)
        crow[1:] = torch.bincount(rows, minlength=t_s).cumsum(0)

        by_col = torch.argsort(cols, stable=True)
        ccol = torch.zeros(self.n_features + 1, dtype=torch.int64)
        ccol[1:] = torch.bincount(cols, minlength=self.n_features).cumsum(0)

        name = f"shard_{len(self.shards):04d}.pt"
        torch.save(
            {
                "ccol": ccol,
                "row": rows[by_col].to(torch.int32),
                "val": vals[by_col],
                "crow": crow,
                "col": cols[by_row].to(torch.int32),
                "val_csr": vals[by_row],
                "token_ids": token_ids.to(torch.int32).cpu(),
            },
            self.root / name,
        )
        self.shards.append(name)
        self.n_tokens += t_s
        self.nnz += int(vals.shape[0])

    def finalize(self) -> CodeStore:
        manifest = {
            "n_features": self.n_features,
            "n_tokens": self.n_tokens,
            "nnz": self.nnz,
            "shards": self.shards,
            **self.meta,
        }
        (self.root / _MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        return CodeStore(self.root)


def pack_token_rows(
    token_iter: Iterator[list[int]],
    *,
    ctx: int,
    bos_id: int,
    n_rows: int,
) -> Iterator[torch.Tensor]:
    """Pack a token stream into (ctx,) rows: BOS + ctx−1 content tokens.

    Documents are concatenated without boundary tokens (the Bloom/SAELens
    harvest convention for gpt2-small-res-jb).
    """
    buffer: list[int] = []
    produced = 0
    for doc in token_iter:
        buffer.extend(doc)
        while len(buffer) >= ctx - 1 and produced < n_rows:
            row = [bos_id] + buffer[: ctx - 1]
            buffer = buffer[ctx - 1 :]
            produced += 1
            yield torch.tensor(row, dtype=torch.long)
        if produced >= n_rows:
            return


@torch.no_grad()
def harvest_codes(
    model,
    sae,
    rows: Iterable[torch.Tensor],
    *,
    hook_name: str,
    out_root: str | Path,
    batch_size: int = 32,
    shard_tokens: int = 1_000_000,
    meta: dict | None = None,
    device: torch.device | str = "cuda",
) -> CodeStore:
    """Run rows through model → hook activations → SAE codes → CodeStore.

    Position 0 (BOS) is dropped from every row. Guards: every batch's code
    matrix is checked for all-zero rows (an SAE with L0≈60 should produce
    ~none; a spike means the hook or encode path silently broke).
    """
    writer = CodeStore.open_writer(out_root, int(sae.cfg.d_sae), meta)
    pending_rows: list[torch.Tensor] = []
    pending_cols: list[torch.Tensor] = []
    pending_vals: list[torch.Tensor] = []
    pending_ids: list[torch.Tensor] = []
    pending = 0
    zero_rows = 0
    total = 0

    def flush() -> None:
        nonlocal pending, pending_rows, pending_cols, pending_vals, pending_ids
        if pending_ids:
            writer.add_shard_coo(
                torch.cat(pending_rows),
                torch.cat(pending_cols),
                torch.cat(pending_vals),
                torch.cat(pending_ids),
            )
            pending_rows, pending_cols, pending_vals, pending_ids = [], [], [], []
            pending = 0

    batch: list[torch.Tensor] = []

    def run_batch() -> None:
        nonlocal zero_rows, total, pending
        toks = torch.stack(batch).to(device)
        _, cache = model.run_with_cache(
            toks, stop_at_layer=int(hook_name.split(".")[1]) + 1, names_filter=hook_name
        )
        acts = cache[hook_name][:, 1:, :]  # drop BOS
        codes = sae.encode(acts.to(sae.dtype)).reshape(-1, int(sae.cfg.d_sae))
        zero = (codes.gt(0).sum(dim=1) == 0).sum()
        zero_rows += int(zero)
        total += codes.shape[0]
        # Sparsify on-device per batch: a dense shard at store scale is
        # ~100 GB of RAM (the OOM that killed harvest run 1).
        nz = codes.nonzero(as_tuple=False)
        pending_rows.append((nz[:, 0] + pending).cpu())
        pending_cols.append(nz[:, 1].cpu())
        pending_vals.append(codes[nz[:, 0], nz[:, 1]].float().cpu())
        pending_ids.append(toks[:, 1:].reshape(-1).cpu())
        pending += codes.shape[0]
        if pending >= shard_tokens:
            flush()

    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            run_batch()
            batch = []
    if batch:
        run_batch()
    flush()

    writer.meta["zero_code_rows"] = zero_rows
    writer.meta["zero_code_fraction"] = zero_rows / max(total, 1)
    store = writer.finalize()
    if total and zero_rows / total > 0.05:
        raise RuntimeError(
            f"{zero_rows}/{total} tokens produced all-zero codes — "
            "suspect the hook/encode path before trusting this store"
        )
    return store


def decoder_hash(sae) -> str:
    data = sae.W_dec.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
