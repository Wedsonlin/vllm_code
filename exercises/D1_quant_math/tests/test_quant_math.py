import numpy as np
from exercises.D1_quant_math.quant_math import quantize_symmetric, dequantize_symmetric

def test_roundtrip_close():
    x = np.array([-1.0, 0.0, 0.5, 2.0], dtype=np.float32)
    q, s = quantize_symmetric(x)
    y = dequantize_symmetric(q, s)
    assert np.max(np.abs(x - y)) < 0.02

def test_zero():
    q, s = quantize_symmetric(np.zeros(4, dtype=np.float32))
    assert np.all(q == 0)
