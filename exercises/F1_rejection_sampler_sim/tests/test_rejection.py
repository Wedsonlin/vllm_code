from exercises.F1_rejection_sampler_sim.rejection import apply_rejection

def test_all_accept():
    r = apply_rejection([10, 11, 12], [True, True, True])
    assert r.accepted == [10, 11, 12] and r.needs_bonus

def test_reject_mid():
    r = apply_rejection([1, 2, 3], [True, False, True])
    assert r.accepted == [1] and not r.needs_bonus and r.accepted_len == 1

def test_empty():
    r = apply_rejection([], [])
    assert r.accepted == [] and r.needs_bonus
