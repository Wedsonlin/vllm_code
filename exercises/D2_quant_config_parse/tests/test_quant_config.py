import pytest
from exercises.D2_quant_config_parse.quant_config import parse_quant_config

def test_parse_dict():
    c = parse_quant_config({"method": "FP8", "bits": 8, "ignore_layers": ["lm_head"]})
    assert c.method == "fp8" and c.bits == 8 and c.ignore_layers == ["lm_head"]

def test_parse_json_str():
    c = parse_quant_config('{"method": "awq", "bits": 4}')
    assert c.method == "awq" and c.bits == 4

def test_bad_method():
    with pytest.raises(ValueError):
        parse_quant_config({"method": "nope", "bits": 8})
