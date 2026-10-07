from enum import Enum, auto
class KVState(Enum):
    IDLE=auto(); ALLOCATED=auto(); TRANSFERRING=auto(); READY=auto(); RELEASED=auto()
class IllegalKVTransition(Exception): pass
class KVTransferSM:
    def __init__(self):
        self.state = KVState.IDLE
    def step(self, event):
        raise NotImplementedError
