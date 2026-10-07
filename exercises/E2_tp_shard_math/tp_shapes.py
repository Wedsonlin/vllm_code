"""E2: compute local shapes for column/row parallel linear."""
from __future__ import annotations
from typing import Tuple


def column_parallel_shape(out_features: int, in_features: int, tp_size: int) -> Tuple[int, int]:
    """Local weight shape for column-parallel: (out/tp, in)."""
    if out_features % tp_size != 0:
        raise ValueError("out_features not divisible by tp_size")
    return (out_features // tp_size, in_features)


def row_parallel_shape(out_features: int, in_features: int, tp_size: int) -> Tuple[int, int]:
    """Local weight shape for row-parallel: (out, in/tp)."""
    if in_features % tp_size != 0:
        raise ValueError("in_features not divisible by tp_size")
    return (out_features, in_features // tp_size)
