"""Phase-0.9 toy manifold export: one block -> saklas-shaped folder.

Exercises the Phase-2 producer contract shape (manifold.json + per-model
safetensors) without claiming the saklas `discovered`-source schema, which
is Phase-2 work that lands in saklas. Per site the export preserves the
*coordinate map*, not just the span (design Phase 2, finding 10): truncated
SVD of the whitened decoder D_g^s gives layer_<L>.basis (r x d orthonormal
rows, saklas orientation), layer_<L>.singular_values, and the b x r
right-factor, so code -> activation reconstruction is exactly
basis^T @ diag(sigma) @ right_factor^T @ z_g. share = per-site
contribution-energy shares over the eval split (never Frobenius decoder
norms — Phase -1 findings §2.5). The whitener seam is worn openly: shares
are in the training-side harvest-fit whitener, NOT re-expressed in a
consumer-side neutral-fit whitener (that re-expression is the Phase-2
bridge's job).

  python -u scripts/export_phase09_toy_manifold.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "google/gemma-3-1b-pt"
SVD_RTOL = 1e-6  # keep sigma > rtol * sigma_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("/data/runs/bcc-phase09/bsc_lam0.001_seed0"),
    )
    parser.add_argument(
        "--store", type=Path,
        default=Path("/data/stores/bcc-phase09/gemma3_1b_6site_fineweb"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/data/runs/bcc-phase09/toy_manifold"),
    )
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"config: run_dir={args.run_dir} store={args.store} out={args.out}",
          flush=True)

    import torch
    from safetensors.torch import load_file, save_file

    from block_crosscoder_experiment.store import StoreReader, Whitener
    from block_crosscoder_experiment.trainer import Trainer

    whitener = Whitener.load(args.store / "whitener.pt")
    sites = list(whitener.sites)
    trainer = Trainer.load_checkpoint(args.run_dir / "latest.pt", device=args.device)
    model = trainer.master.to(args.device)
    model.eval()
    cfg = model.cfg
    report = json.loads((args.run_dir / "report.json").read_text())
    if report["whitener_hash"] != whitener.hash:
        raise SystemExit("run/store whitener mismatch")

    # ---- per-block per-site contribution energy over the eval split ------
    # Under the Gram constraint sum_s D_g^s D_g^s^T = I_b the site-summed
    # energy per firing is exactly ||z_g||^2 — checked below as an invariant.
    reader = StoreReader(args.store, "eval", expected_whitener_hash=whitener.hash)
    gram = torch.einsum("sgbd,sgcd->sgbc", model.D, model.D)  # [S, G, b, b]
    energy = torch.zeros(cfg.n_sites, cfg.n_blocks, dtype=torch.float64,
                         device=args.device)
    firings = torch.zeros(cfg.n_blocks, dtype=torch.float64, device=args.device)
    z_sq = 0.0
    n = 0
    with torch.no_grad():
        for x in reader.sequential_batches(args.batch):
            out = model(x.to(args.device, torch.float32), mode="topk")
            energy += torch.einsum(
                "ngb,sgbc,ngc->sg", out.z_selected, gram, out.z_selected
            ).double()
            firings += out.mask.sum(dim=0).double()
            z_sq += float(out.z_selected.pow(2).sum())
            n += x.shape[0]
    gram_gap = abs(float(energy.sum()) - z_sq) / max(z_sq, 1e-12)
    print(f"energy accounting over {n:,} eval tokens: "
          f"gram-invariant gap {gram_gap:.2e}", flush=True)

    total = energy.sum(dim=0)  # [G]
    top = total.topk(5)
    print("top blocks by contribution energy:", flush=True)
    for e, g in zip(top.values.tolist(), top.indices.tolist()):
        share = (energy[:, g] / energy[:, g].sum()).tolist()
        freq = float(firings[g]) / n
        print(f"  block {g}: energy {e:,.0f}, firing freq {freq:.4f}, "
              f"share/site {[round(v, 3) for v in share]}", flush=True)
    g_star = int(top.indices[0])
    shares = (energy[:, g_star] / energy[:, g_star].sum()).cpu()

    # ---- per-site truncated SVD of the whitened decoder ------------------
    tensors: dict[str, torch.Tensor] = {}
    site_meta = {}
    for s, layer in enumerate(sites):
        m = model.D[s, g_star].detach().float().T.cpu()  # [d, b]
        u, sv, vh = torch.linalg.svd(m, full_matrices=False)
        r = int((sv > SVD_RTOL * sv.max()).sum())
        u_r, sv_r, v_r = u[:, :r], sv[:r], vh.T[:, :r]
        ortho = float((u_r.T @ u_r - torch.eye(r)).abs().max())
        recon = float((u_r @ torch.diag(sv_r) @ v_r.T - m).norm() / m.norm())
        if ortho > 1e-5 or recon > 1e-5:
            raise SystemExit(f"site {layer}: SVD validation failed "
                             f"(ortho {ortho:.2e}, recon {recon:.2e})")
        tensors[f"layer_{layer}.basis"] = u_r.T.contiguous()  # (r, d) rows
        tensors[f"layer_{layer}.mean"] = model.c[s].detach().float().cpu()
        tensors[f"layer_{layer}.singular_values"] = sv_r
        tensors[f"layer_{layer}.right_factor"] = v_r.contiguous()  # (b, r)
        sigma_frac = (sv_r.pow(2) / sv_r.pow(2).sum()).tolist()
        site_meta[str(layer)] = {
            "rank": r,
            "sigma_energy_fractions": [round(v, 4) for v in sigma_frac],
        }
        print(f"  site {layer}: rank {r}, sigma {[round(float(v), 4) for v in sv_r]}, "
              f"ortho {ortho:.1e}, recon {recon:.1e}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    tensor_file = f"{MODEL.split('/')[-1]}.safetensors"
    save_file(tensors, str(args.out / tensor_file))
    manifest = {
        "schema": "bcc-discovered-toy-v0",
        "name": f"phase09-toy-block-{g_star}",
        "source": "block-crosscoder-experiment Phase 0.9 rehearsal",
        "model": MODEL,
        "run": {k: report[k] for k in ("arm", "lam", "seed", "total_steps")},
        "block": g_star,
        "block_dim": cfg.block_dim,
        "sites": sites,
        "hook_names": whitener.meta["hook_names"],
        "whitener_hash": whitener.hash,
        "whitener_seam": (
            "training-side harvest-fit whitener; shares NOT re-expressed in a "
            "consumer-side neutral-fit whitener (Phase-2 saklas bridge item)"
        ),
        "share_per_site": {
            str(layer): round(float(shares[s]), 6)
            for s, layer in enumerate(sites)
        },
        "share_kind": "contribution-energy over eval split (topk mode)",
        "firing_frequency": round(float(firings[g_star]) / n, 6),
        "eval_tokens": n,
        "site_svd": site_meta,
        "files": [tensor_file],
    }
    (args.out / "manifold.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Reload round trip: the folder must reproduce the decoder exactly.
    back = load_file(str(args.out / tensor_file))
    for s, layer in enumerate(sites):
        rebuilt = (
            back[f"layer_{layer}.basis"].T
            @ torch.diag(back[f"layer_{layer}.singular_values"])
            @ back[f"layer_{layer}.right_factor"].T
        )
        err = float(
            (rebuilt - model.D[s, g_star].detach().float().T.cpu()).norm()
            / rebuilt.norm()
        )
        if err > 1e-5:
            raise SystemExit(f"site {layer}: reload round trip failed ({err:.2e})")
    print(f"reload round trip: all {len(sites)} sites < 1e-5", flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
