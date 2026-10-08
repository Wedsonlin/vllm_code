import numpy as np
import pytest
from exercises.C1_triton_vector_add.vector_add import vector_add_numpy
from common.helpers import has_cuda, skip_if_no_gpu

def test_numpy_add():
    a = np.array([1, 2, 3], dtype=np.float32)
    b = np.array([4, 5, 6], dtype=np.float32)
    assert np.allclose(vector_add_numpy(a, b), np.array([5, 7, 9], dtype=np.float32))

def test_shape_mismatch():
    with pytest.raises(ValueError):
        vector_add_numpy(np.zeros(2), np.zeros(3))

@skip_if_no_gpu
@pytest.mark.gpu
def test_torch_optional():
    import torch
    from exercises.C1_triton_vector_add.vector_add import vector_add_torch
    a = torch.tensor([1.0, 2.0], device="cuda")
    b = torch.tensor([3.0, 4.0], device="cuda")
    assert torch.allclose(vector_add_torch(a, b), torch.tensor([4.0, 6.0], device="cuda"))
