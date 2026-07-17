"""Phase-0 artifact identity verification (design §Phase 0, P13).

Pins the two SAE artifacts the phase runs on and records their identity so
the phase verdict is auditable:

1. Positive control: Bloom 2024 GPT-2-small layer-7 residual SAE — the exact
   artifact Engels et al. clustered (`gpt2-small-res-jb`,
   `blocks.7.hook_resid_pre`, d_sae 24576). A mismatch here downgrades the
   control to a labeled transfer control per the design.
2. Target: `gemma-scope-2-4b-pt-res` residual SAE at the saklas-convention
   analysis depth (select_runtime_layer: nearest 65% depth in the 40–90%
   band), width pinned nearest the Engels ~24k scale.

Zero model forwards; loads SAE weights only. Writes
data/phase0/artifact_provenance.json and prints a summary.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

CONTROL_RELEASE = "gpt2-small-res-jb"
CONTROL_SAE_ID = "blocks.7.hook_resid_pre"
CONTROL_EXPECTED = {"d_in": 768, "d_sae": 24576}

TARGET_RELEASE = "gemma-scope-2-4b-pt-res"
# Engels demonstrated ring recovery at ~24k features; 16k is the nearest
# available width (1m would need a ~4 TB dense similarity matrix at fp32).
TARGET_WIDTH = "16k"
# Middle of the released L0 family; recorded so a swap is a visible decision.
TARGET_L0 = "medium"

SAE_ID_RE = re.compile(r"layer_(\d+)_width_([0-9a-z]+)_l0_([a-z]+)")


def _tensor_sha256(t) -> str:
    import torch

    data = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _repo_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().repo_info(repo_id).sha


def _load_sae(release: str, sae_id: str):
    from sae_lens import SAE

    return SAE.from_pretrained(release, sae_id, device="cpu", dtype="float32")


def _sae_identity(sae, release: str, sae_id: str) -> dict[str, Any]:
    cfg = sae.cfg
    meta = getattr(cfg, "metadata", None)
    hook_name = getattr(meta, "hook_name", None) or getattr(cfg, "hook_name", None)
    arch = getattr(cfg, "architecture", None)
    if callable(arch):
        arch = arch()
    return {
        "release": release,
        "sae_id": sae_id,
        "d_in": int(cfg.d_in),
        "d_sae": int(cfg.d_sae),
        "hook_name": hook_name,
        "architecture": str(arch) if arch else type(sae).__name__,
        "W_dec_shape": list(sae.W_dec.shape),
        "W_dec_sha256": _tensor_sha256(sae.W_dec),
    }


def _gemma_n_layers(model_id: str) -> int:
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    text_cfg = cfg.get("text_config", cfg)
    return int(text_cfg["num_hidden_layers"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/phase0/artifact_provenance.json"),
    )
    args = parser.parse_args()

    import sae_lens
    from sae_lens.loading.pretrained_saes_directory import (
        get_pretrained_saes_directory,
    )
    from saklas.core.sae import select_runtime_layer

    directory = get_pretrained_saes_directory()
    report: dict[str, Any] = {
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sae_lens_version": sae_lens.__version__,
        "checks": [],
    }

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))

    # --- positive control -------------------------------------------------
    control_entry = directory[CONTROL_RELEASE]
    control = _load_sae(CONTROL_RELEASE, CONTROL_SAE_ID)
    control_id = _sae_identity(control, CONTROL_RELEASE, CONTROL_SAE_ID)
    control_id["repo_id"] = control_entry.repo_id
    control_id["repo_sha"] = _repo_sha(control_entry.repo_id)
    control_id["model"] = control_entry.model
    report["control"] = control_id

    check(
        "control is Bloom 2024 reformatted release",
        control_entry.repo_id == "jbloom/GPT2-Small-SAEs-Reformatted",
        control_entry.repo_id,
    )
    for key, want in CONTROL_EXPECTED.items():
        check(f"control {key} == {want}", control_id[key] == want, str(control_id[key]))
    check(
        "control hook is layer-7 resid_pre",
        control_id["hook_name"] == CONTROL_SAE_ID,
        str(control_id["hook_name"]),
    )

    # --- gemma target -----------------------------------------------------
    target_entry = directory[TARGET_RELEASE]
    model_id = target_entry.model
    n_layers = _gemma_n_layers(model_id)

    parsed = {}
    for sae_id in target_entry.saes_map:
        m = SAE_ID_RE.fullmatch(sae_id)
        if m:
            layer, width, l0 = int(m.group(1)), m.group(2), m.group(3)
            parsed.setdefault(layer, []).append((width, l0, sae_id))
    check("target sae_ids parse", len(parsed) > 0, f"{len(parsed)} layers")

    layer = select_runtime_layer(frozenset(parsed), n_layers=n_layers)
    at_layer = sorted(parsed[layer])
    pick = [s for w, l, s in at_layer if w == TARGET_WIDTH and l == TARGET_L0]
    check(
        f"width_{TARGET_WIDTH}/l0_{TARGET_L0} exists at layer {layer}",
        bool(pick),
        f"available: {[f'{w}/{l}' for w, l, _ in at_layer]}",
    )
    target_sae_id = pick[0]

    target = _load_sae(TARGET_RELEASE, target_sae_id)
    target_id = _sae_identity(target, TARGET_RELEASE, target_sae_id)
    target_id["repo_id"] = target_entry.repo_id
    target_id["repo_sha"] = _repo_sha(target_entry.repo_id)
    target_id["model"] = model_id
    target_id["model_n_layers"] = n_layers
    target_id["selected_layer"] = layer
    target_id["layer_fraction"] = round(layer / (n_layers - 1), 4)
    target_id["available_layers"] = sorted(parsed)
    target_id["available_at_layer"] = [f"{w}/{l}" for w, l, _ in at_layer]
    target_id["width_rationale"] = (
        "nearest available width to the Engels ~24k control scale "
        "(16k vs 65k/262k/1m); larger widths also inflate the dense "
        "similarity matrix spectral clustering needs"
    )
    report["target"] = target_id

    check(
        "target d_in matches gemma-3-4b hidden size",
        target_id["d_in"] == 2560,
        str(target_id["d_in"]),
    )
    check(
        "selected layer in 40-90% workspace band",
        0.4 <= target_id["layer_fraction"] <= 0.9,
        f"layer {layer}/{n_layers - 1} = {target_id['layer_fraction']:.0%}",
    )

    # Phase 0.5 lookahead: coherence pre-test needs SAEs at 2-3 depths.
    report["phase05_depth_coverage"] = sorted(parsed)
    check(
        "phase 0.5 has >= 3 depths available",
        len(parsed) >= 3,
        f"layers {sorted(parsed)}",
    )

    ok = all(c["ok"] for c in report["checks"])
    report["verdict"] = "PASS" if ok else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nverdict: {report['verdict']}  ->  {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
