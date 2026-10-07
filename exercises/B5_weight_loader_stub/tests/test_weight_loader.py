import numpy as np
from exercises.B5_weight_loader_stub.weight_loader import shard_column_parallel, shard_row_parallel

def test_column():
    w = np.arange(12).reshape(4, 3)
    s0 = shard_column_parallel(w, 2, 0)
    s1 = shard_column_parallel(w, 2, 1)
    assert s0.shape == (2, 3)
    assert np.array_equal(s0, w[:2])
    assert np.array_equal(s1, w[2:])

def test_row():
    w = np.arange(12).reshape(3, 4)
    s0 = shard_row_parallel(w, 2, 0)
    assert s0.shape == (3, 2)
    assert np.array_equal(s0, w[:, :2])
