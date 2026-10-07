"""F3: KV transfer state machine for PD disaggregation (simplified)."""
from __future__ import annotations
from enum import Enum, auto


class KVState(Enum):
    IDLE = auto()
    ALLOCATED = auto()
    TRANSFERRING = auto()
    READY = auto()
    RELEASED = auto()


class IllegalKVTransition(Exception):
    pass


class KVTransferSM:
    _EDGES = {
        (KVState.IDLE, "allocate"): KVState.ALLOCATED,
        (KVState.ALLOCATED, "start_transfer"): KVState.TRANSFERRING,
        (KVState.TRANSFERRING, "complete"): KVState.READY,
        (KVState.READY, "release"): KVState.RELEASED,
        (KVState.ALLOCATED, "release"): KVState.RELEASED,
        (KVState.TRANSFERRING, "abort"): KVState.RELEASED,
    }

    def __init__(self):
        self.state = KVState.IDLE

    def step(self, event: str) -> KVState:
        key = (self.state, event)
        if key not in self._EDGES:
            raise IllegalKVTransition(f"{self.state} + {event}")
        self.state = self._EDGES[key]
        return self.state
