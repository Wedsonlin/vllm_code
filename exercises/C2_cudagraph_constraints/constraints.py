"""C2: validate CUDA graph capture constraint checklist."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple


REQUIRED_TRUE = (
    "static_shapes",
    "no_cpu_sync_in_region",
    "no_dynamic_control_flow",
)
REQUIRED_FALSE = (
    "uses_graph_incompatible_ops",
)


def validate_capture_config(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (ok, list_of_violation_messages).

    Expected keys (bool): static_shapes, no_cpu_sync_in_region,
    no_dynamic_control_flow, uses_graph_incompatible_ops.
    Also require max_batch > 0 if present.
    """
    violations: List[str] = []
    for k in REQUIRED_TRUE:
        if not cfg.get(k, False):
            violations.append(f"{k} must be True")
    for k in REQUIRED_FALSE:
        if cfg.get(k, True):
            violations.append(f"{k} must be False")
    if "max_batch" in cfg and cfg["max_batch"] <= 0:
        violations.append("max_batch must be > 0")
    return (len(violations) == 0, violations)
