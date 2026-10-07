"""STARTER"""
from __future__ import annotations
import queue, struct
from typing import Callable, Optional

def frame_message(payload: bytes) -> bytes:
    raise NotImplementedError

def unframe_message(framed: bytes) -> bytes:
    raise NotImplementedError

class ReqRepBus:
    def __init__(self):
        self.to_server = queue.Queue()
        self.to_client = queue.Queue()
    def serve_once(self, handler):
        raise NotImplementedError
    def request(self, payload, handler=None):
        raise NotImplementedError
