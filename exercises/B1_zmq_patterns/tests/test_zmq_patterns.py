from exercises.B1_zmq_patterns.zmq_patterns import frame_message, unframe_message, ReqRepBus

def test_frame_roundtrip():
    raw = b"hello-vllm"
    assert unframe_message(frame_message(raw)) == raw

def test_empty():
    assert unframe_message(frame_message(b"")) == b""

def test_reqrep():
    bus = ReqRepBus()
    out = bus.request(b"ping", handler=lambda m: m.upper())
    assert out == b"PING"
