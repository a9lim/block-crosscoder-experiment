import pytest
import torch


def _devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_devices())
def device(request) -> torch.device:
    return torch.device(request.param)
