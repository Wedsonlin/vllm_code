"""STARTER — fill in TODOs until tests pass."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class ServeArgs:
    model: str
    port: int = 8000
    tensor_parallel_size: int = 1

def parse_serve_args(argv: List[str]) -> ServeArgs:
    raise NotImplementedError("TODO: parse --model/--port/--tensor-parallel-size")

def check_version_compatible(current: str, minimum: str) -> bool:
    raise NotImplementedError("TODO: semver compare")
