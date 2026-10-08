"""Shared test helpers: GPU skip, tiny fixtures."""

from __future__ import annotations

import pytest


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


skip_if_no_gpu = pytest.mark.skipif(
    not has_cuda(),
    reason="CUDA GPU not available",
)


@pytest.fixture
def tiny_float_vector():
    """Small numpy vector for CPU math checks."""
    import numpy as np

    return np.array([1.0, -2.0, 0.5, 3.25], dtype=np.float32)
