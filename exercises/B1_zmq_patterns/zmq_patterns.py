"""B1: REQ/REP framing over queues (no real ZMQ required)."""
from __future__ import annotations

import queue
import struct
from typing import Callable, Optional


def frame_message(payload: bytes) -> bytes:
    """TODO: 4-byte big-endian length + payload."""
    return struct.pack(">I", len(payload)) + payload


def unframe_message(framed: bytes) -> bytes:
    """TODO: validate length header and return payload."""
    if len(framed) < 4:
        raise ValueError("frame too short")
    (n,) = struct.unpack(">I", framed[:4])
    body = framed[4:]
    if len(body) != n:
        raise ValueError(f"length mismatch: header={n} body={len(body)}")
    return body


class ReqRepBus:
    """In-process REQ/REP using two queues."""

    def __init__(self):
        self.to_server: queue.Queue[bytes] = queue.Queue()
        self.to_client: queue.Queue[bytes] = queue.Queue()

    def serve_once(self, handler: Callable[[bytes], bytes]) -> None:
        framed = self.to_server.get(timeout=1.0)
        req = unframe_message(framed)
        resp = handler(req)
        self.to_client.put(frame_message(resp))

    def request(self, payload: bytes, handler: Optional[Callable[[bytes], bytes]] = None) -> bytes:
        """Send request; if handler given, process one serve_once inline."""
        self.to_server.put(frame_message(payload))
        if handler is not None:
            self.serve_once(handler)
        framed = self.to_client.get(timeout=1.0)
        return unframe_message(framed)
