import pytest
from exercises.F3_kv_state_machine.kv_sm import KVTransferSM, KVState, IllegalKVTransition

def test_happy():
    sm = KVTransferSM()
    assert sm.step("allocate") is KVState.ALLOCATED
    assert sm.step("start_transfer") is KVState.TRANSFERRING
    assert sm.step("complete") is KVState.READY
    assert sm.step("release") is KVState.RELEASED

def test_abort():
    sm = KVTransferSM()
    sm.step("allocate")
    sm.step("start_transfer")
    assert sm.step("abort") is KVState.RELEASED

def test_illegal():
    sm = KVTransferSM()
    with pytest.raises(IllegalKVTransition):
        sm.step("complete")
