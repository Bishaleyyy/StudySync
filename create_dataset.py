"""
create_dataset.py
------------------
Generates a small, SYNTHETIC dataset for the StudySync project.

Why synthetic?
Real student data (marks, attendance, subject difficulty per student) is
private and hard to collect for a class project in one week. So we create
a rule-based synthetic dataset: we decide a sensible "priority score"
formula based on real study-planning logic (low marks + high difficulty +
few days left + low attendance => higher priority), then add random noise
so the pattern is not perfectly linear. This keeps the DNN's job realistic
(it has to learn a noisy pattern) without pretending the data is real.

Run:
    python create_dataset.py
Output:
    data/study_data.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducibility

N_SAMPLES = 300

# ---- 1. Generate raw feature values -------------------------------------

study_hours_available = np.random.randint(1, 7, N_SAMPLES)          # 1-6 hrs/day
previous_marks = np.random.randint(20, 100, N_SAMPLES)              # marks out of 100
attendance = np.random.randint(40, 100, N_SAMPLES)                  # % attendance
assignment_score = np.random.randint(0, 100, N_SAMPLES)             # assignment %
difficulty = np.random.choice(["Low", "Medium", "High"], N_SAMPLES,
                               p=[0.3, 0.4, 0.3])
days_left = np.random.randint(1, 30, N_SAMPLES)                     # days before exam

# Map difficulty text to a numeric weight just for LABEL CREATION
# (the actual model will encode difficulty separately in preprocessing.py)
difficulty_weight = {"Low": 0, "Medium": 1, "High": 2}
difficulty_num = np.array([difficulty_weight[d] for d in difficulty])

# ---- 2. Build a "need_score": higher = subject needs more attention -----
# This is a hand-designed formula reflecting common-sense study planning:
#   - low marks           -> needs more attention
#   - low attendance      -> needs more attention
#   - low assignment score-> needs more attention
#   - high difficulty     -> needs more attention
#   - few days left       -> needs more attention (urgency)
# Each term is scaled to roughly the same 0-1 range before combining.

need_score = (
    (100 - previous_marks) / 100 * 0.30 +
    (100 - attendance) / 100 * 0.20 +
    (100 - assignment_score) / 100 * 0.15 +
    (difficulty_num / 2) * 0.20 +
    (1 - days_left / 30) * 0.15
)

# add small random noise so the relationship is not perfectly clean
# (this is what makes the DNN's learning task realistic, not trivial)
noise = np.random.normal(0, 0.05, N_SAMPLES)
need_score = need_score + noise
need_score = np.clip(need_score, 0, 1)

# ---- 3. Convert the continuous need_score into 3 priority classes -------
# We use quantile-based cuts (qcut) so the 3 classes come out reasonably
# balanced (~1/3 each), which makes classification metrics easier to
# interpret for a beginner project.

priority = pd.qcut(
    need_score,
    q=3,
    labels=["Low", "Medium", "High"]
)

# ---- 4. Assemble the final dataframe ------------------------------------

df = pd.DataFrame({
    "Study_Hours_Available": study_hours_available,
    "Previous_Marks": previous_marks,
    "Attendance": attendance,
    "Assignment_Score": assignment_score,
    "Subject_Difficulty": difficulty,
    "Days_Left": days_left,
    "Priority_Level": priority.astype(str)
})

df.to_csv("data/study_data.csv", index=False)
print("Dataset created: data/study_data.csv")
print(df["Priority_Level"].value_counts())
print(df.head())
