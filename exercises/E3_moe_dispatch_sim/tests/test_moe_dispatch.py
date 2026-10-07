import numpy as np
from exercises.E3_moe_dispatch_sim.moe_dispatch import expert_token_counts, load_imbalance_ratio

def test_counts():
    idx = np.array([[0, 1], [0, 2], [1, 2]])
    assert expert_token_counts(idx, 4) == [2, 2, 2, 0]

def test_imbalance():
    assert load_imbalance_ratio([2, 2, 2, 0]) == 2 / 1.5
