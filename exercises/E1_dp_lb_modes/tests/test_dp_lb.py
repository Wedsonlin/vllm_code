import pytest
from exercises.E1_dp_lb_modes.dp_lb import classify_dp_lb_mode

def test_modes():
    assert classify_dp_lb_mode({"external": True}) == "external"
    assert classify_dp_lb_mode({"internal_lb": True, "cross_node": False}) == "internal"
    assert classify_dp_lb_mode({"internal_lb": True, "cross_node": True}) == "hybrid"

def test_bad():
    with pytest.raises(ValueError):
        classify_dp_lb_mode({})
