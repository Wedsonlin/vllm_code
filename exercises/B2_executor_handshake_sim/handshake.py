"""B2: Executor handshake state machine."""
from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    INIT = auto()
    HELLO_SENT = auto()
    READY = auto()
    CLOSED = auto()


class IllegalTransition(Exception):
    pass


class Handshake:
    """TODO(learner): enforce allowed edges only."""

    _EDGES = {
        (State.INIT, "hello"): State.HELLO_SENT,
        (State.HELLO_SENT, "ack"): State.READY,
        (State.READY, "close"): State.CLOSED,
        (State.INIT, "close"): State.CLOSED,
        (State.HELLO_SENT, "close"): State.CLOSED,
    }

    def __init__(self):
        self.state = State.INIT

    def step(self, event: str) -> State:
        key = (self.state, event)
        if key not in self._EDGES:
            raise IllegalTransition(f"{self.state} + {event}")
        self.state = self._EDGES[key]
        return self.state

    @property
    def is_ready(self) -> bool:
        return self.state is State.READY
