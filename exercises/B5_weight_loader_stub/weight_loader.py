"""B5: shard full weight tensor by TP rank (numpy)."""
from __future__ import annotations
import numpy as np


def shard_column_parallel(weight: np.ndarray, tp_size: int, tp_rank: int) -> np.ndarray:
    """Split axis 0 into tp_size equal shards; return shard for tp_rank."""
    if weight.ndim != 2:
        raise ValueError("expect 2D weight")
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError("bad tp")
    rows = weight.shape[0]
    if rows % tp_size != 0:
        raise ValueError("rows not divisible by tp_size")
    chunk = rows // tp_size
    start = tp_rank * chunk
    return weight[start : start + chunk].copy()


def shard_row_parallel(weight: np.ndarray, tp_size: int, tp_rank: int) -> np.ndarray:
    """Split axis 1 into tp_size equal shards."""
    if weight.ndim != 2:
        raise ValueError("expect 2D weight")
    if tp_size <= 0 or not (0 <= tp_rank < tp_size):
        raise ValueError("bad tp")
    cols = weight.shape[1]
    if cols % tp_size != 0:
        raise ValueError("cols not divisible by tp_size")
    chunk = cols // tp_size
    start = tp_rank * chunk
    return weight[:, start : start + chunk].copy()
