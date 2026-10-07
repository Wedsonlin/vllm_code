from exercises.B3_scheduler_token_budget.scheduler import Request, admit, remaining_budget

def test_admit_pack():
    rs = [Request("a", 3), Request("b", 5), Request("c", 2)]
    got = admit(rs, 7)
    assert [r.req_id for r in got] == ["a", "c"]
    assert remaining_budget(rs, 7) == 2

def test_empty_budget():
    assert admit([Request("a", 1)], 0) == []

def test_all_fit():
    rs = [Request("a", 1), Request("b", 1)]
    assert admit(rs, 10) == list(rs)
