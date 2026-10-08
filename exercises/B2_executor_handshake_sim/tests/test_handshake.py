import pytest
from exercises.B2_executor_handshake_sim.handshake import Handshake, State, IllegalTransition

def test_happy_path():
    h = Handshake()
    assert h.step("hello") is State.HELLO_SENT
    assert h.step("ack") is State.READY
    assert h.is_ready
    assert h.step("close") is State.CLOSED

def test_illegal():
    h = Handshake()
    with pytest.raises(IllegalTransition):
        h.step("ack")
