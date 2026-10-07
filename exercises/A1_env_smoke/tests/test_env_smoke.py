from exercises.A1_env_smoke.env_smoke import ServeArgs, parse_serve_args, check_version_compatible
import pytest

def test_parse_full():
    a = parse_serve_args(["--model", "meta/llama", "--port", "9000", "-tp", "4"])
    assert a == ServeArgs(model="meta/llama", port=9000, tensor_parallel_size=4)

def test_parse_short_model():
    a = parse_serve_args(["-m", "x", "-p", "8080"])
    assert a.model == "x" and a.port == 8080 and a.tensor_parallel_size == 1

def test_missing_model():
    with pytest.raises(ValueError):
        parse_serve_args(["--port", "8000"])

def test_version_ok():
    assert check_version_compatible("0.30.1", "0.30.0")
    assert check_version_compatible("0.30.0", "0.30.0")
    assert not check_version_compatible("0.29.9", "0.30.0")
