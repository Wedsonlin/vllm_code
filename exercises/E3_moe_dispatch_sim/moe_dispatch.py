"""E3: count tokens per expert from top-k routing indices."""
from __future__ import annotations
from typing import List, Sequence
import numpy as np


def expert_token_counts(topk_indices: np.ndarray, num_experts: int) -> List[int]:
    """topk_indices: shape [num_tokens, k] with expert ids in [0, num_experts)."""
    if num_experts <= 0:
        raise ValueError("num_experts")
    counts = [0] * num_experts
    flat = np.asarray(topk_indices).reshape(-1)
    for e in flat:
        e = int(e)
        if e < 0 or e >= num_experts:
            raise ValueError(f"expert id out of range: {e}")
        counts[e] += 1
    return counts


def load_imbalance_ratio(counts: Sequence[int]) -> float:
    """max/mean; 1.0 if all zero or empty."""
    if not counts:
        return 1.0
    s = sum(counts)
    if s == 0:
        return 1.0
    mean = s / len(counts)
    return max(counts) / mean
