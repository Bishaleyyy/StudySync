"""
planner.py
-----------
NOT a neural network. A simple, explainable rule-based scheduler that turns
the DNN's priority predictions into hours per subject.

Logic:
    - Each priority level has a weight: High = 3, Medium = 2, Low = 1.
    - Total hours are split across subjects in proportion to the weights.

Rounding policy (documented):
    - Hours are rounded to 0.01 hr (36 seconds).
    - Rounding uses the "largest remainder" method, so the allocated hours
      ALWAYS add up exactly to the available hours (no 5.01 or 0.98).
"""

PRIORITY_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}


def generate_study_plan(subject_priorities, total_hours_per_day):
    """
    subject_priorities: dict like {"DAA": "High", "DBMS": "Medium", ...}
    total_hours_per_day: number > 0, hours available TODAY (not across days)

    Returns: dict {"DAA": 2.5, ...}; values are >= 0 and sum to total_hours_per_day.
    """
    if not subject_priorities:
        return {}
    if total_hours_per_day <= 0:
        raise ValueError("total_hours_per_day must be greater than 0")
    unknown = {p for p in subject_priorities.values() if p not in PRIORITY_WEIGHTS}
    if unknown:
        raise ValueError(f"Unknown priority level(s): {sorted(unknown)}")

    weights = {s: PRIORITY_WEIGHTS[p] for s, p in subject_priorities.items()}
    total_weight = sum(weights.values())

    total_units = round(total_hours_per_day * 100)          # work in 0.01-hr units
    exact = {s: w / total_weight * total_units for s, w in weights.items()}
    units = {s: int(x) for s, x in exact.items()}            # floor
    leftover = total_units - sum(units.values())
    # give the leftover units to the largest fractional remainders
    for s in sorted(exact, key=lambda k: exact[k] - units[k], reverse=True)[:leftover]:
        units[s] += 1

    return {s: u / 100 for s, u in units.items()}


def print_study_plan(plan, subject_priorities):
    """Pretty-print the generated plan, sorted by allocated hours (desc)."""
    print("\n===== Personalized Study Plan =====")
    for subject, hours in sorted(plan.items(), key=lambda item: item[1], reverse=True):
        print(f"  {subject:<10} | Priority: {subject_priorities[subject]:<6} | Study Time: {hours} hrs")
    print(f"  {'TOTAL':<10} | {'':<16} | Study Time: {round(sum(plan.values()), 2)} hrs")
    print("====================================\n")
