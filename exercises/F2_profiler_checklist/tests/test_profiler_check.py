from exercises.F2_profiler_checklist.profiler_check import validate_profiler_report

def test_ok():
    ok, bad = validate_profiler_report({
        "trace_path": "/tmp/t.json",
        "device": "cuda:0",
        "num_steps": 10,
        "avg_latency_ms": 1.5,
        "framework": "pytorch",
    })
    assert ok and bad == []

def test_missing():
    ok, bad = validate_profiler_report({"trace_path": "x"})
    assert not ok
    assert any("missing:device" in b for b in bad)
