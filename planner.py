"""
planner.py
-----------
This is NOT a neural network. It is a simple, explainable rule-based
scheduling step that turns the DNN's priority predictions into an actual
study plan (hours per subject).

Logic:
    - Each priority level is given a "weight":
          High   = 3 parts
          Medium = 2 parts
          Low    = 1 part
    - The student's total available study hours are split across subjects
      in proportion to these weights.

This keeps the project simple: the DNN's job is only to figure out WHICH
subjects need more attention; a plain weighted-allocation formula (not a
second neural network) decides HOW MANY HOURS each subject gets. This
separation makes the system much easier to explain in a viva.
"""

PRIORITY_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}


def generate_study_plan(subject_priorities, total_hours_per_day):
    """
    subject_priorities: dict like {"DAA": "High", "DBMS": "Medium", ...}
    total_hours_per_day: float/int, total hours the student has today

    Returns: dict {"DAA": 2.5, "DBMS": 1.5, ...} (hours allocated)
    """
    weights = {subj: PRIORITY_WEIGHTS[level] for subj, level in subject_priorities.items()}
    total_weight = sum(weights.values())

    plan = {}
    for subject, weight in weights.items():
        allocated_hours = (weight / total_weight) * total_hours_per_day
        plan[subject] = round(allocated_hours, 2)

    return plan


def print_study_plan(plan, subject_priorities):
    """Pretty-print the generated plan, sorted by allocated hours (desc)."""
    print("\n===== Personalized Study Plan =====")
    sorted_subjects = sorted(plan.items(), key=lambda item: item[1], reverse=True)
    for subject, hours in sorted_subjects:
        priority = subject_priorities[subject]
        print(f"  {subject:<10} | Priority: {priority:<6} | Study Time: {hours} hrs")
    print("====================================\n")
