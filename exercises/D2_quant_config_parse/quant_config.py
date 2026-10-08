"""D2: parse simple quant config JSON."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List


ALLOWED_METHODS = {"fp8", "awq", "gptq", "smoothquant", "int8"}


@dataclass
class QuantConfig:
    method: str
    bits: int
    ignore_layers: List[str]


def parse_quant_config(data: Any) -> QuantConfig:
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise TypeError("config must be dict or JSON str")
    method = str(data.get("method", "")).lower()
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported method: {method}")
    bits = int(data.get("bits", 8))
    if bits not in (4, 8, 16):
        raise ValueError("bits must be 4, 8, or 16")
    ignore = data.get("ignore_layers", [])
    if not isinstance(ignore, list) or not all(isinstance(x, str) for x in ignore):
        raise ValueError("ignore_layers must be list[str]")
    return QuantConfig(method=method, bits=bits, ignore_layers=list(ignore))
