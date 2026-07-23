"""Canonical activation-capture and derived-view content identities."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from .studies import canonical_json


ACTIVATION_CONTENT_IDENTITY_SCHEMA = "bsc-activation-content-identity-v1"

_REAL_PREPARATION_DATA_FIELDS = {
    "kind",
    "root",
    "splits",
    "bindings",
    "row_intervals",
    "row_intervals_disjoint",
    "declared_split_contract",
    "raw_root",
    "raw_bindings",
    "raw_declared_split_contract",
    "source_contract",
    "store_view_policy",
    "training_row_policy",
    "normalization",
}
_SOURCE_CONTRACT_FIELDS = {
    "path",
    "sha256",
    "source_hash",
    "source",
    "declared",
    "split_order",
    "split_plan",
    "capture_binding_sha256",
    "capture_binding",
    "capture_implementation",
    "capture_content_sha256",
    "splits",
    "capture",
}
_SPLIT_RECORD_FIELDS = {
    "requested_tokens",
    "actual_tokens",
    "manifest_sha256",
    "row_stream_sha256",
    "content_stream_sha256",
}
_DERIVED_NORMALIZATION_FIELDS = {
    "mode",
    "transform_sha256",
    "transform_hash",
    "view_manifest_sha256",
    "view_manifest_file_sha256",
}
_ON_THE_FLY_NORMALIZATION_FIELDS = {
    "mode",
    "application",
    "transform_path",
    "transform_sha256",
    "transform_hash",
    "transform_manifest",
    "transform_manifest_sha256",
    "selected_site_indices",
    "source_capture_sha256",
    "source_fit_manifest",
    "source_fit_manifest_file_sha256",
    "source_fit_manifest_sha256",
    "source_fit_row_stream_sha256",
    "source_fit_requested_tokens",
}
_NORMALIZATION_LOCATOR_FIELDS = {
    "transform_path",
    "transform_manifest",
    "source_fit_manifest",
}


def _validate_split_contract(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"real activation {label} split identity is incomplete")
    for split, record in value.items():
        if (
            not isinstance(split, str)
            or not split
            or not isinstance(record, Mapping)
            or set(record) != _SPLIT_RECORD_FIELDS
            or type(record.get("requested_tokens")) is not int
            or type(record.get("actual_tokens")) is not int
            or record["requested_tokens"] <= 0
            or record["actual_tokens"] < record["requested_tokens"]
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(record.get(name, ""))) is None
                for name in (
                    "manifest_sha256",
                    "row_stream_sha256",
                    "content_stream_sha256",
                )
            )
        ):
            raise ValueError(
                f"real activation {label} split {split!r} is noncanonical"
            )
    return value


def activation_content_identity(data: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the only valid activation identity from preparation data."""

    if set(data) != _REAL_PREPARATION_DATA_FIELDS or data.get("kind") != "activation_store":
        raise ValueError("real activation preparation uses a noncanonical field set")
    source = data.get("source_contract")
    normalization = data.get("normalization")
    raw_splits = data.get("raw_declared_split_contract")
    view_splits = data.get("declared_split_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (source, normalization, raw_splits, view_splits)
    ):
        raise ValueError(
            "real activation preparation lacks a complete content identity"
        )
    if set(source) != _SOURCE_CONTRACT_FIELDS:
        raise ValueError("real activation source contract is noncanonical")
    required_source_fields = (
        "sha256",
        "capture_binding_sha256",
        "source_hash",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(source.get(name, ""))) is None
        for name in required_source_fields
    ):
        raise ValueError("real activation source identity has an invalid digest")
    raw_splits = _validate_split_contract(raw_splits, label="raw")
    view_splits = _validate_split_contract(view_splits, label="view")
    if set(raw_splits) != set(view_splits):
        raise ValueError("real activation raw/view split grids differ")
    for split in raw_splits:
        if (
            raw_splits[split]["actual_tokens"]
            != view_splits[split]["actual_tokens"]
            or raw_splits[split]["row_stream_sha256"]
            != view_splits[split]["row_stream_sha256"]
        ):
            raise ValueError(
                f"real activation raw/view split {split!r} rows differ"
            )
    store_view_policy = data.get("store_view_policy")
    expected_normalization_fields = (
        _DERIVED_NORMALIZATION_FIELDS
        if store_view_policy == "content_addressed_derived_view"
        else (
            _ON_THE_FLY_NORMALIZATION_FIELDS
            if store_view_policy
            == "single_bf16_raw_view_on_the_fly_invertible_normalization"
            else None
        )
    )
    if expected_normalization_fields is None or set(normalization) != expected_normalization_fields:
        raise ValueError("real activation normalization contract is noncanonical")
    normalization_mode = normalization.get("mode")
    transform_hash = normalization.get("transform_hash")
    transform_sha256 = normalization.get("transform_sha256")
    if (
        not isinstance(store_view_policy, str)
        or not store_view_policy
        or not isinstance(normalization_mode, str)
        or not normalization_mode
        or re.fullmatch(r"[0-9a-f]{64}", str(transform_hash or "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(transform_sha256 or "")) is None
    ):
        raise ValueError("real activation view identity is incomplete")
    raw_contract = {
        "source_capture_sha256": source.get("sha256"),
        "capture_binding_sha256": source.get("capture_binding_sha256"),
        "source_hash": source.get("source_hash"),
        "splits": raw_splits,
    }
    raw_digest = hashlib.sha256(
        canonical_json(raw_contract).encode("utf-8")
    ).hexdigest()
    view_contract = {
        "raw_content_identity_sha256": raw_digest,
        "store_view_policy": store_view_policy,
        "normalization_mode": normalization_mode,
        "transform_hash": transform_hash,
        "transform_sha256": transform_sha256,
        "normalization": {
            name: normalization[name]
            for name in sorted(normalization)
            if name not in _NORMALIZATION_LOCATOR_FIELDS
        },
        "splits": view_splits,
    }
    return {
        "schema": ACTIVATION_CONTENT_IDENTITY_SCHEMA,
        "raw_content_identity_sha256": raw_digest,
        "view_key": normalization_mode,
        "view_content_identity_sha256": hashlib.sha256(
            canonical_json(view_contract).encode("utf-8")
        ).hexdigest(),
        "raw_contract": raw_contract,
        "view_contract": view_contract,
    }
