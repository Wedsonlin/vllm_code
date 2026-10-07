"""A1: fake vLLM serve args parser + version check stub.

Learners: re-implement the bodies marked TODO; tests import this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ServeArgs:
    model: str
    port: int = 8000
    tensor_parallel_size: int = 1


def parse_serve_args(argv: List[str]) -> ServeArgs:
    """Parse a minimal subset of `vllm serve` style argv (no real CLI lib).

    TODO(learner): walk argv; support:
      --model / -m <str>
      --port / -p <int>
      --tensor-parallel-size / -tp <int>
    Raise ValueError if model missing.
    """
    model = None
    port = 8000
    tp = 1
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--model", "-m") and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
            continue
        if a in ("--port", "-p") and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
            continue
        if a in ("--tensor-parallel-size", "-tp") and i + 1 < len(argv):
            tp = int(argv[i + 1])
            i += 2
            continue
        i += 1
    if not model:
        raise ValueError("missing --model")
    return ServeArgs(model=model, port=port, tensor_parallel_size=tp)


def _parse_semver(s: str) -> tuple:
    parts = s.strip().lstrip("v").split(".")
    nums = []
    for p in parts[:3]:
        nums.append(int("".join(c for c in p if c.isdigit()) or "0"))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def check_version_compatible(current: str, minimum: str) -> bool:
    """Return True if current >= minimum (simple semver).

    TODO(learner): compare (major, minor, patch) tuples.
    """
    return _parse_semver(current) >= _parse_semver(minimum)
