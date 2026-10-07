import pytest
from exercises.E2_tp_shard_math.tp_shapes import column_parallel_shape, row_parallel_shape

def test_shapes():
    assert column_parallel_shape(4096, 1024, 4) == (1024, 1024)
    assert row_parallel_shape(4096, 1024, 4) == (4096, 256)

def test_bad():
    with pytest.raises(ValueError):
        column_parallel_shape(10, 4, 3)
