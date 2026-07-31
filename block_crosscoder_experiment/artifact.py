"""Load and use the compact reviewer-facing BSC artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .codec import Codec, EncodedBatch, decode_batch, encode_batch
from .model import BSCConfig, BlockCrosscoder

ARTIFACT_SCHEMA = "bsc-winning-formula-artifact-v1"
DEFAULT_ARTIFACT_NAME = "bsc-16m.pt"


@dataclass(slots=True)
class BSCArtifact:
    """A trained model, calibrated codec, and fitted scalar-RMS transform."""

    model: BlockCrosscoder
    codec: Codec
    normalization: dict[str, Any]
    raw_calibration_mean: torch.Tensor
    metadata: dict[str, Any]

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        mean = self.normalization["mean"].to(raw.device)
        scale = torch.diagonal(
            self.normalization["W"].to(raw.device),
            dim1=-2,
            dim2=-1,
        ).unsqueeze(0)
        return (raw.float() - mean) * scale

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        mean = self.normalization["mean"].to(normalized.device)
        scale = torch.diagonal(
            self.normalization["W"].to(normalized.device),
            dim1=-2,
            dim2=-1,
        ).unsqueeze(0)
        return normalized.float() / scale.clamp_min(1e-30) + mean

    @torch.no_grad()
    def encode(self, raw: torch.Tensor, *, q: int = 8) -> EncodedBatch:
        """Normalize raw four-site activations and encode a deployable packet."""

        return encode_batch(self.model, self.codec, self.normalize(raw), q)

    @torch.no_grad()
    def decode(self, packet: EncodedBatch) -> torch.Tensor:
        """Decode a packet back into raw four-site activation coordinates."""

        return self.denormalize(decode_batch(self.model, self.codec, packet))


def load_artifact(
    path: str | Path = DEFAULT_ARTIFACT_NAME,
    *,
    device: str | torch.device = "cuda",
) -> BSCArtifact:
    """Load the single-file artifact emitted by ``bsc replicate``."""

    artifact_path = Path(path)
    if artifact_path.is_dir():
        artifact_path = artifact_path / DEFAULT_ARTIFACT_NAME
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"{artifact_path} is not a BSC winning-formula artifact")
    model = BlockCrosscoder(BSCConfig(**payload["model_cfg"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device).eval()
    codec = Codec.from_payload(payload["codec"], source=str(artifact_path))
    normalization = dict(payload["normalization"])
    if normalization.get("mode") != "scalar_rms":
        raise ValueError("winning-formula artifact requires scalar_rms normalization")
    return BSCArtifact(
        model=model,
        codec=codec,
        normalization=normalization,
        raw_calibration_mean=payload["raw_calibration_mean"],
        metadata={
            "claim": payload["claim"],
            "formula": dict(payload["formula"]),
            "training": dict(payload["training"]),
        },
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "BSCArtifact",
    "DEFAULT_ARTIFACT_NAME",
    "load_artifact",
]
