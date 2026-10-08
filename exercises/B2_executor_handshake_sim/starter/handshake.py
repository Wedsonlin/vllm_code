from enum import Enum, auto
class State(Enum):
    INIT = auto(); HELLO_SENT = auto(); READY = auto(); CLOSED = auto()
class IllegalTransition(Exception): pass
class Handshake:
    def __init__(self):
        self.state = State.INIT
    def step(self, event: str):
        raise NotImplementedError
    @property
    def is_ready(self):
        return False
