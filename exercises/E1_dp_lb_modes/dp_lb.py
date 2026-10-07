"""E1: classify DP load-balance modes from flags."""
from __future__ import annotations
from typing import Dict


def classify_dp_lb_mode(flags: Dict[str, bool]) -> str:
    """
    Rules:
      external=True -> "external"  (front LB owns routing)
      external=False and internal_lb=True and cross_node=True -> "hybrid"
      external=False and internal_lb=True -> "internal"
      else -> raise ValueError
    """
    external = bool(flags.get("external", False))
    internal = bool(flags.get("internal_lb", False))
    cross = bool(flags.get("cross_node", False))
    if external:
        return "external"
    if internal and cross:
        return "hybrid"
    if internal:
        return "internal"
    raise ValueError("cannot classify DP LB mode from flags")
