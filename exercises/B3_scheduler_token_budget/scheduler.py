"""B3: token-budget admission."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Request:
    req_id: str
    num_tokens: int


def admit(requests: Sequence[Request], budget: int) -> List[Request]:
    """Greedy admit in order while sum(num_tokens) <= budget.

    Skip a request that does not fit; continue to later ones (optional packing).
    Spec for course: **first-fit in order, skip if does not fit**.
    """
    if budget < 0:
        raise ValueError("budget must be >= 0")
    used = 0
    out: List[Request] = []
    for r in requests:
        if r.num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        if used + r.num_tokens <= budget:
            out.append(r)
            used += r.num_tokens
    return out


def remaining_budget(requests: Sequence[Request], budget: int) -> int:
    admitted = admit(requests, budget)
    return budget - sum(r.num_tokens for r in admitted)
