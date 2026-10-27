import itertools, pytest
from planner import generate_study_plan as g, generate_multi_day_plan as mdp, total_hours_per_subject as tot

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


def test_two_days_three_subjects_uses_all_time():
    pr = {"A": "High", "B": "Medium", "C": "Low"}
    sched = mdp(pr, {"A": 2, "B": 2, "C": 2}, 5)
    assert len(sched) == 2
    assert all(round(sum(d.values()), 2) == 5 for d in sched)
    assert round(sum(tot(sched).values()), 2) == 10

def test_subject_not_scheduled_after_its_exam():
    sched = mdp({"A": "High", "B": "High"}, {"A": 1, "B": 3}, 4)
    assert "A" in sched[0] and "A" not in sched[1] and "A" not in sched[2]
    assert all(round(sum(d.values()), 2) == 4 for d in sched)   # B takes the full day later

def test_total_equals_hours_times_days():
    pr = {"A": "High", "B": "Low", "C": "Medium"}
    sched = mdp(pr, {"A": 10, "B": 12, "C": 14}, 5)
    assert len(sched) == 14
    assert round(sum(tot(sched).values()), 2) == 70

def test_multi_day_bad_input():
    with pytest.raises(ValueError): mdp({"A": "High"}, {"A": 0}, 5)
    with pytest.raises(ValueError): mdp({"A": "High"}, {"B": 3}, 5)
