import pytest
from exercises.B4_block_pool.block_pool import BlockPool, BlockPoolExhausted

def test_alloc_free():
    p = BlockPool(2)
    a = p.allocate()
    b = p.allocate()
    with pytest.raises(BlockPoolExhausted):
        p.allocate()
    p.free(a)
    c = p.allocate()
    assert c == a or p.free_count == 0

def test_refcount():
    p = BlockPool(1)
    x = p.allocate()
    p.retain(x)
    assert p.refcount(x) == 2
    p.free(x)
    assert p.refcount(x) == 1
    p.free(x)
    assert p.free_count == 1
