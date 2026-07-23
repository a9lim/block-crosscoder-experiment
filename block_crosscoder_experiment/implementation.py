"""Canonical executable, numerical-runtime, and hardware identity.

Git metadata is retained as provenance, but it is deliberately excluded from
the execution digest.  The executable package bytes, imported dependency
versions, numerical backend state, and physical CUDA devices are the things
that can change scientific results and therefore form the campaign pin.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from contextlib import contextmanager
import fcntl
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping

import torch


CANONICAL_EXECUTOR_SCHEMA = "bsc-cell-executor-v13"
CANONICAL_EXECUTOR_PROCESS_MODEL = "persistent_exact_snapshot_lineage_v5"
IMPLEMENTATION_IDENTITY_SCHEMA = "bsc-implementation-identity-v2"

IMPLEMENTATION_DEPENDENCIES = (
    "block-crosscoder-experiment",
    "datasets",
    "huggingface-hub",
    "numpy",
    "sae-lens",
    "safetensors",
    "torch",
    "transformer-lens",
    "transformers",
    "triton",
)

_EXECUTION_FIELDS = (
    "schema",
    "executor_schema",
    "executor_process_model",
    "python_source_sha256",
    "python_source_files",
    "python",
    "platform",
    "torch",
    "torch_cuda_build",
    "dependencies",
    "numerical_runtime",
    "cuda_runtime",
)

_NUMERICAL_RUNTIME_FIELDS = (
    "float32_matmul_precision",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cudnn_allow_tf32",
    "cuda_matmul_allow_tf32",
    "cuda_matmul_allow_fp16_reduced_precision_reduction",
    "cuda_matmul_allow_bf16_reduced_precision_reduction",
    "environment",
)

_NUMERICAL_ENVIRONMENT_FIELDS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_identity(package_root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    source_files = sorted(package_root.rglob("*.py"))
    for path in source_files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(source_files)


def _git_provenance(package_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "source_dirty": None}
    try:
        top = Path(
            subprocess.run(
                ["git", "-C", str(package_root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        result["commit"] = subprocess.run(
            ["git", "-C", str(top), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        relative = package_root.relative_to(top)
        status = subprocess.run(
            [
                "git",
                "-C",
                str(top),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                relative.as_posix(),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result["source_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return result


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in IMPLEMENTATION_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _cuda_driver_version() -> str | None:
    try:
        rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    versions = sorted({row.strip() for row in rows if row.strip()})
    return ",".join(versions) if versions else None


def _cuda_devices() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, "uuid", None)
        devices.append(
            {
                "visible_index": index,
                "uuid": None if uuid is None else str(uuid),
                "name": str(properties.name),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory": int(properties.total_memory),
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )
    return devices


def implementation_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    source_sha256, source_count = _source_identity(package_root)
    numerical_runtime = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            )
        },
    }
    return {
        "schema": IMPLEMENTATION_IDENTITY_SCHEMA,
        "executor_schema": CANONICAL_EXECUTOR_SCHEMA,
        "executor_process_model": CANONICAL_EXECUTOR_PROCESS_MODEL,
        "python_source_sha256": source_sha256,
        "python_source_files": source_count,
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "dependencies": _dependency_versions(),
        "numerical_runtime": numerical_runtime,
        "cuda_runtime": {
            "available": bool(torch.cuda.is_available()),
            "driver": _cuda_driver_version(),
            "cudnn": None
            if torch.backends.cudnn.version() is None
            else str(torch.backends.cudnn.version()),
            "devices": _cuda_devices(),
        },
        "provenance": {"git": _git_provenance(package_root)},
    }


def execution_identity_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {name: identity[name] for name in _EXECUTION_FIELDS}


def execution_identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(execution_identity_payload(identity)).encode("utf-8")
    ).hexdigest()


def validate_implementation_identity(
    identity: Mapping[str, Any],
    *,
    scientific: bool,
) -> str:
    """Validate the current identity schema and return its execution digest.

    Provenance is authenticated and validated but intentionally does not enter
    the digest: the exact executable bytes and numerical runtime are invariant
    under a clean commit of an already-pinned tree.
    """

    expected_top = {*_EXECUTION_FIELDS, "provenance"}
    if set(identity) != expected_top:
        raise ValueError("implementation identity has a noncanonical field set")
    if (
        identity.get("schema") != IMPLEMENTATION_IDENTITY_SCHEMA
        or identity.get("executor_schema") != CANONICAL_EXECUTOR_SCHEMA
        or identity.get("executor_process_model") != CANONICAL_EXECUTOR_PROCESS_MODEL
    ):
        raise ValueError("implementation identity uses the wrong current schema")
    source_digest = identity.get("python_source_sha256")
    source_files = identity.get("python_source_files")
    if not (
        isinstance(source_digest, str)
        and len(source_digest) == 64
        and all(character in "0123456789abcdef" for character in source_digest)
        and type(source_files) is int
        and source_files > 0
    ):
        raise ValueError("implementation source identity is malformed")
    if not isinstance(identity.get("python"), str) or not identity["python"]:
        raise ValueError("implementation Python identity is malformed")
    platform_identity = identity.get("platform")
    if (
        not isinstance(platform_identity, Mapping)
        or set(platform_identity)
        != {
            "system",
            "machine",
            "release",
        }
        or any(
            not isinstance(platform_identity[field], str)
            or not platform_identity[field]
            for field in platform_identity
        )
    ):
        raise ValueError("implementation platform identity is malformed")
    if not isinstance(identity.get("torch"), str) or not identity["torch"]:
        raise ValueError("implementation torch identity is malformed")
    cuda_build = identity.get("torch_cuda_build")
    if cuda_build is not None and not isinstance(cuda_build, str):
        raise ValueError("implementation CUDA build identity is malformed")

    dependencies = identity.get("dependencies")
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies) != set(IMPLEMENTATION_DEPENDENCIES)
        or any(
            version is not None and not isinstance(version, str)
            for version in dependencies.values()
        )
    ):
        raise ValueError("implementation dependency identity is malformed")

    numerical = identity.get("numerical_runtime")
    if not isinstance(numerical, Mapping) or set(numerical) != set(
        _NUMERICAL_RUNTIME_FIELDS
    ):
        raise ValueError("implementation numerical runtime is noncanonical")
    if not isinstance(numerical.get("float32_matmul_precision"), str) or any(
        type(numerical.get(field)) is not bool
        for field in _NUMERICAL_RUNTIME_FIELDS
        if field not in {"float32_matmul_precision", "environment"}
    ):
        raise ValueError("implementation numerical flags are malformed")
    environment = numerical.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != set(_NUMERICAL_ENVIRONMENT_FIELDS)
        or any(
            value is not None and not isinstance(value, str)
            for value in environment.values()
        )
    ):
        raise ValueError("implementation numerical environment is malformed")

    cuda_runtime = identity.get("cuda_runtime")
    if not isinstance(cuda_runtime, Mapping) or set(cuda_runtime) != {
        "available",
        "driver",
        "cudnn",
        "devices",
    }:
        raise ValueError("implementation CUDA runtime is noncanonical")
    if type(cuda_runtime.get("available")) is not bool or any(
        cuda_runtime.get(field) is not None
        and not isinstance(cuda_runtime.get(field), str)
        for field in ("driver", "cudnn")
    ):
        raise ValueError("implementation CUDA runtime fields are malformed")
    devices = cuda_runtime.get("devices")
    if not isinstance(devices, list):
        raise ValueError("implementation CUDA device list is malformed")
    for device in devices:
        if not isinstance(device, Mapping) or set(device) != {
            "visible_index",
            "uuid",
            "name",
            "compute_capability",
            "total_memory",
            "multi_processor_count",
        }:
            raise ValueError("implementation CUDA device identity is noncanonical")
        capability = device.get("compute_capability")
        if not (
            type(device.get("visible_index")) is int
            and device["visible_index"] >= 0
            and (device.get("uuid") is None or isinstance(device["uuid"], str))
            and isinstance(device.get("name"), str)
            and device["name"]
            and isinstance(capability, list)
            and len(capability) == 2
            and all(type(value) is int and value >= 0 for value in capability)
            and type(device.get("total_memory")) is int
            and device["total_memory"] > 0
            and type(device.get("multi_processor_count")) is int
            and device["multi_processor_count"] > 0
        ):
            raise ValueError("implementation CUDA device identity is malformed")
    if bool(devices) != bool(cuda_runtime["available"]):
        raise ValueError("implementation CUDA availability/device list disagrees")

    provenance = identity.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {"git"}:
        raise ValueError("implementation provenance is noncanonical")
    git = provenance.get("git")
    if not isinstance(git, Mapping) or set(git) != {"commit", "source_dirty"}:
        raise ValueError("implementation Git provenance is noncanonical")
    commit = git.get("commit")
    dirty = git.get("source_dirty")
    if commit is not None and not (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("implementation Git commit is malformed")
    if dirty is not None and type(dirty) is not bool:
        raise ValueError("implementation Git dirty state is malformed")
    if scientific and (commit is None or dirty is not False):
        raise ValueError("scientific execution requires a clean committed source tree")
    return execution_identity_sha256(identity)


def physical_cuda_device_key(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("physical CUDA identity requires a CUDA device")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        return str(uuid)
    return (
        f"{properties.name}|cc={properties.major}.{properties.minor}|"
        f"mem={properties.total_memory}|mp={properties.multi_processor_count}"
    )


def cuda_execution_lock_path(device: torch.device | str) -> Path:
    """Return a per-user, physical-device lock path outside mutable temp roots."""

    physical_identity = physical_cuda_device_key(device)
    identity_hash = hashlib.sha256(physical_identity.encode("utf-8")).hexdigest()[:16]
    root = Path(f"/var/tmp/block-crosscoder-experiment-gpu-locks-{os.getuid()}")
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    root_stat = root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.getuid()
        or stat.S_IMODE(root_stat.st_mode) & 0o077
    ):
        raise RuntimeError(f"unsafe host GPU lock directory: {root}")
    return root / f"gpu-{identity_hash}.lock"


@contextmanager
def host_cuda_execution_lock(
    device: torch.device | str,
    *,
    operation: str,
    owner_id: str,
) -> Iterator[None]:
    """Serialize all project CUDA work on one physical device and host."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        yield
        return
    path = cuda_execution_lock_path(resolved)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.getuid()
            or status.st_nlink != 1
        ):
            raise RuntimeError(f"unsafe host GPU lock file: {path}")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        payload = {
            "schema": "bsc-host-gpu-lock-v2",
            "operation": operation,
            "owner_id": owner_id,
            "device": str(resolved),
            "physical_device": physical_cuda_device_key(resolved),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        body = (_canonical_json(payload) + "\n").encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(body):
            written = os.write(descriptor, body[offset:])
            if written <= 0:
                raise RuntimeError(f"short write to host GPU lock {path}")
            offset += written
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
