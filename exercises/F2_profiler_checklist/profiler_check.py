"""F2: validate profiler result dict has required fields."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple

REQUIRED = (
    "trace_path",
    "device",
    "num_steps",
    "avg_latency_ms",
    "framework",
)


def validate_profiler_report(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing = [k for k in REQUIRED if k not in report]
    bad: List[str] = [f"missing:{k}" for k in missing]
    if "num_steps" in report and int(report["num_steps"]) <= 0:
        bad.append("num_steps must be > 0")
    if "avg_latency_ms" in report and float(report["avg_latency_ms"]) < 0:
        bad.append("avg_latency_ms must be >= 0")
    if "framework" in report and str(report["framework"]).lower() not in {"pytorch", "nsight", "vllm"}:
        bad.append("framework must be pytorch|nsight|vllm")
    return (len(bad) == 0, bad)
