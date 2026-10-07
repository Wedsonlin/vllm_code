"""C1: vector add — numpy reference (default for tests)."""
from __future__ import annotations
import numpy as np


def vector_add_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add; broadcast not required — same shape."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return (a.astype(np.float32) + b.astype(np.float32)).astype(np.float32)


def vector_add_torch(a, b):
    """Optional torch path."""
    import torch
    return torch.as_tensor(a) + torch.as_tensor(b)
