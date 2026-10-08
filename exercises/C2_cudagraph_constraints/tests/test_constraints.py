from exercises.C2_cudagraph_constraints.constraints import validate_capture_config

def test_ok():
    ok, v = validate_capture_config({
        "static_shapes": True,
        "no_cpu_sync_in_region": True,
        "no_dynamic_control_flow": True,
        "uses_graph_incompatible_ops": False,
        "max_batch": 8,
    })
    assert ok and v == []

def test_bad():
    ok, v = validate_capture_config({
        "static_shapes": False,
        "no_cpu_sync_in_region": True,
        "no_dynamic_control_flow": True,
        "uses_graph_incompatible_ops": True,
    })
    assert not ok
    assert any("static_shapes" in x for x in v)
    assert any("uses_graph_incompatible_ops" in x for x in v)
