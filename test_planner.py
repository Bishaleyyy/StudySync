import itertools, pytest
from planner import generate_study_plan as g

LEVELS = ["High", "Medium", "Low"]

@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 12])
@pytest.mark.parametrize("hours", [1, 2, 3, 4, 5, 6, 2.5])
def test_sum_exact_and_nonnegative(n, hours):
    for combo in itertools.islice(itertools.product(LEVELS, repeat=n), 200):
        plan = g({f"S{i}": p for i, p in enumerate(combo)}, hours)
        assert round(sum(plan.values()), 2) == hours
        assert all(v >= 0 for v in plan.values())

def test_higher_priority_gets_more():
    p = g({"A": "High", "B": "Medium", "C": "Low"}, 5)
    assert p["A"] > p["B"] > p["C"]

def test_readme_example():
    p = g({"DAA": "High", "DBMS": "Low", "OS": "Medium", "Statistics": "High"}, 5)
    assert round(sum(p.values()), 2) == 5.0

def test_bad_input():
    assert g({}, 5) == {}
    with pytest.raises(ValueError): g({"A": "High"}, 0)
    with pytest.raises(ValueError): g({"A": "Urgent"}, 5)
