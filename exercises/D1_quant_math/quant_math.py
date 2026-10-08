"""D1: symmetric int8 quantize / dequantize."""
from __future__ import annotations
import numpy as np

QMAX = 127


def symmetric_scale(x: np.ndarray) -> float:
    m = float(np.max(np.abs(x)))
    if m == 0.0:
        return 1.0
    return m / QMAX


def quantize_symmetric(x: np.ndarray) -> tuple:
    """Return (q_int8 ndarray, scale float)."""
    x = np.asarray(x, dtype=np.float32)
    scale = symmetric_scale(x)
    q = np.rint(np.clip(x / scale, -QMAX, QMAX)).astype(np.int8)
    return q, scale


def dequantize_symmetric(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * np.float32(scale)
