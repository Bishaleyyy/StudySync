"""
main.py
--------
This is the file you run to DEMONSTRATE the whole StudySync pipeline
end-to-end, using a trained model:

    Student subject data -> DNN priority prediction -> Study plan

Run AFTER train.py has been run at least once (so the trained weights
and scaler values exist in model/).

Run:
    python main.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import StudySyncDNN
from preprocessing import DIFFICULTY_MAP, PRIORITY_MAP_INV, scale_new_sample
from planner import generate_study_plan, print_study_plan

# ---- 1. Load the trained model and the scaler values ----------------------
model = StudySyncDNN(input_size=6)
model.load_state_dict(torch.load("model/studysync_model.pth"))
model.eval()

mean = np.load("model/scaler_mean.npy")
std = np.load("model/scaler_std.npy")

# ---- 2. Example student data: several subjects with their own stats -------
# In a real deployment this would come from a form filled by the student.
# Feature order MUST match FEATURE_COLUMNS in preprocessing.py:
# [Study_Hours_Available, Previous_Marks, Attendance, Assignment_Score,
#  Subject_Difficulty, Days_Left]

student_subjects = {
    "DAA":     {"Study_Hours_Available": 4, "Previous_Marks": 45, "Attendance": 70,
                "Assignment_Score": 55, "Subject_Difficulty": "High", "Days_Left": 10},
    "DBMS":    {"Study_Hours_Available": 4, "Previous_Marks": 80, "Attendance": 90,
                "Assignment_Score": 85, "Subject_Difficulty": "Low", "Days_Left": 10},
    "OS":      {"Study_Hours_Available": 4, "Previous_Marks": 60, "Attendance": 75,
                "Assignment_Score": 65, "Subject_Difficulty": "Medium", "Days_Left": 10},
    "Statistics": {"Study_Hours_Available": 4, "Previous_Marks": 50, "Attendance": 60,
                "Assignment_Score": 50, "Subject_Difficulty": "Medium", "Days_Left": 10},
}

TOTAL_HOURS_TODAY = 5  # student says: "I have 5 hours to study today"

# ---- 3. Predict priority for each subject using the DNN --------------------
subject_priorities = {}

print("===== DNN Priority Predictions =====")
for subject, data in student_subjects.items():
    feature_row = [
        data["Study_Hours_Available"],
        data["Previous_Marks"],
        data["Attendance"],
        data["Assignment_Score"],
        DIFFICULTY_MAP[data["Subject_Difficulty"]],
        data["Days_Left"],
    ]
    scaled_input = scale_new_sample(feature_row, mean, std)

    with torch.no_grad():
        logits = model(scaled_input)
        predicted_class = torch.argmax(logits, dim=1).item()

    priority_label = PRIORITY_MAP_INV[predicted_class]
    subject_priorities[subject] = priority_label
    print(f"  {subject:<10} -> Priority: {priority_label}")

# ---- 4. Generate the study plan using the simple scheduler -----------------
study_plan = generate_study_plan(subject_priorities, TOTAL_HOURS_TODAY)
print_study_plan(study_plan, subject_priorities)

# ---- 5. Visualize the allocated study plan (Matplotlib bar chart) ----------
subjects = list(study_plan.keys())
hours = list(study_plan.values())
colors = ["#e74c3c" if subject_priorities[s] == "High"
          else "#f1c40f" if subject_priorities[s] == "Medium"
          else "#2ecc71" for s in subjects]

plt.figure(figsize=(7, 4))
plt.bar(subjects, hours, color=colors)
plt.title(f"Today's Study Plan ({TOTAL_HOURS_TODAY} hrs total)")
plt.ylabel("Allocated Hours")
plt.xlabel("Subject")
for i, h in enumerate(hours):
    plt.text(i, h + 0.05, str(h), ha="center")
plt.tight_layout()
plt.savefig("outputs/study_plan.png")
plt.close()
print("Saved outputs/study_plan.png")
