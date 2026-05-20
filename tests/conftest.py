"""Shared pytest fixtures and CUDA skip-marker."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: requires a CUDA-capable GPU")
    config.addinivalue_line("markers", "triton: requires triton + CUDA")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords or "triton" in item.keywords:
            item.add_marker(skip_cuda)
