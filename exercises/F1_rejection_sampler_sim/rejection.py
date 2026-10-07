"""F1: speculative decoding rejection sampler (simplified)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RejectResult:
    accepted: List[int]
    accepted_len: int
    needs_bonus: bool


def apply_rejection(draft_tokens: Sequence[int], accept_mask: Sequence[bool]) -> RejectResult:
    """Accept longest prefix where accept_mask[i] is True; stop at first False.

    If all draft accepted, needs_bonus=True (target samples one extra).
    If draft empty, accepted=[], needs_bonus=True.
    """
    if len(draft_tokens) != len(accept_mask):
        raise ValueError("length mismatch")
    accepted: List[int] = []
    for tok, ok in zip(draft_tokens, accept_mask):
        if not ok:
            return RejectResult(accepted=accepted, accepted_len=len(accepted), needs_bonus=False)
        accepted.append(int(tok))
    return RejectResult(accepted=accepted, accepted_len=len(accepted), needs_bonus=True)
