import pytest
import torch

from tlab.device import get_device


def test_falls_back_to_cpu_when_no_accelerator_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert get_device() == torch.device("cpu")


def test_selects_mps_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_device() == torch.device("mps")


@pytest.mark.parametrize("mps_available", [False, True])
def test_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch, mps_available: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert get_device() == torch.device("cuda")
