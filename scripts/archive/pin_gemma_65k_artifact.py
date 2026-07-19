"""Pin the width_65k reroute artifact + decoder geometry (Phase 0 reroute #1).

The 16k discovery null's mechanism story says ring recovery needs the
dictionary to split cyclic families into per-class features, and splitting
is an overcompleteness effect — so the reroute moves to width_65k (25×,
the regime of Engels' non-GPT-2 results). This script makes that move
auditable before any harvest:

1. Loads `layer_22_width_65k_l0_medium` from `gemma-scope-2-4b-pt-res`,
   records its identity (d_sae, hook, W_dec sha256, repo sha) under
   `reroute_width_65k` in data/phase0/artifact_provenance.json.
2. Computes decoder geometry — median max-neighbor cosine and the fraction
   of features with a >0.7 cosine neighbor — for BOTH 16k and 65k with the
   same code, so the does-25x-create-fans comparison is method-matched
   (and the quoted 16k numbers get an independent recomputation).

  python scripts/pin_gemma_65k_artifact.py
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

RELEASE = "gemma-scope-2-4b-pt-res"
SAE_IDS = ("layer_22_width_16k_l0_medium", "layer_22_width_65k_l0_medium")
EXPECTED = {"layer_22_width_65k_l0_medium": {"d_in": 2560, "d_sae": 65536,
            "hook_name": "blocks.22.hook_resid_post"}}


def max_neighbor_cosine(decoder, device: str, chunk: int = 4096):
    """Per-feature max off-diagonal cosine to any other decoder row."""
    import torch

    u = decoder.to(device, torch.float32)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    out = torch.empty(u.shape[0], device=device)
    for lo in range(0, u.shape[0], chunk):
        hi = min(lo + chunk, u.shape[0])
        cos = u[lo:hi] @ u.T
        cos[torch.arange(hi - lo, device=device), torch.arange(lo, hi, device=device)] = -1.0
        out[lo:hi] = cos.max(dim=1).values
        del cos
    return out.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/phase0/artifact_provenance.json")
    )
    args = parser.parse_args()

    import torch
    from huggingface_hub import HfApi
    from sae_lens import SAE
    from sae_lens.loading.pretrained_saes_directory import (
        get_pretrained_saes_directory,
    )

    from block_crosscoder_experiment.phase0.harvest import decoder_hash

    device = "cuda" if torch.cuda.is_available() else "cpu"
    entry = get_pretrained_saes_directory()[RELEASE]
    record: dict = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo_id": entry.repo_id,
        "repo_sha": HfApi().repo_info(entry.repo_id).sha,
        "geometry": {},
    }

    ok = True
    for sae_id in SAE_IDS:
        sae = SAE.from_pretrained(RELEASE, sae_id, device="cpu", dtype="float32")
        meta = getattr(sae.cfg, "metadata", None)
        hook = getattr(meta, "hook_name", None) or getattr(sae.cfg, "hook_name", None)
        mnc = max_neighbor_cosine(sae.W_dec.detach(), device)
        geo = {
            "d_in": int(sae.cfg.d_in),
            "d_sae": int(sae.cfg.d_sae),
            "hook_name": hook,
            "W_dec_sha256": decoder_hash(sae),
            "overcompleteness": round(int(sae.cfg.d_sae) / int(sae.cfg.d_in), 2),
            "median_max_neighbor_cos": round(float(mnc.median()), 4),
            "frac_neighbor_gt_0.7": round(float(mnc.gt(0.7).float().mean()), 4),
            "frac_neighbor_gt_0.5": round(float(mnc.gt(0.5).float().mean()), 4),
        }
        record["geometry"][sae_id] = geo
        for key, want in EXPECTED.get(sae_id, {}).items():
            good = geo[key] == want
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {sae_id} {key} == {want} — {geo[key]}")
        print(
            f"{sae_id}: {geo['overcompleteness']}x, median max-neighbor cos "
            f"{geo['median_max_neighbor_cos']}, >0.7 frac {geo['frac_neighbor_gt_0.7']}",
            flush=True,
        )
        del sae

    provenance = json.loads(args.out.read_text()) if args.out.exists() else {}
    provenance["reroute_width_65k"] = record
    args.out.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{'PASS' if ok else 'FAIL'}  -> {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
