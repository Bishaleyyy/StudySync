# StudySync – A Neural Network Based Personalized Study Planning System

A beginner-friendly BSc. CSIT Neural Networks project. StudySync uses a small
PyTorch DNN to predict how much **priority** a subject needs (Low / Medium /
High) based on a student's marks, attendance, difficulty, available hours,
and days left before the exam. A simple rule-based scheduler then turns
those priorities into an actual **hours-per-subject study plan**.

> **Note:** The dataset used here is a small, synthetic, rule-based dataset
> created for academic demonstration. It is a prototype, not real student
> data, and the reported accuracy should be read in that light.

## Folder Structure

```
StudySync/
│
├── data/
│   └── study_data.csv          # generated synthetic dataset
├── model/
│   ├── studysync_model.pth     # trained DNN weights (created by train.py)
│   ├── scaler_mean.npy         # feature scaling values (created by train.py)
│   └── scaler_std.npy
├── outputs/
│   ├── loss_curve.png          # created by train.py
│   ├── confusion_matrix.png    # created by train.py
│   └── study_plan.png          # created by main.py
├── create_dataset.py           # generates data/study_data.csv
├── preprocessing.py            # encoding, scaling, train/test split
├── model.py                    # DNN architecture (PyTorch)
├── train.py                    # trains + evaluates + saves the model
├── planner.py                  # rule-based hour allocation
├── main.py                     # full demo: predict priorities + build plan
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Run these three scripts **in order**:

```bash
# 1. Create the synthetic dataset
python create_dataset.py

# 2. Train the DNN (saves weights + plots)
python train.py

# 3. Run the full demo: predict priorities and build a study plan
python main.py
```

## Expected Output

- `create_dataset.py` prints the class distribution and a preview of the data.
- `train.py` prints the loss every 20 epochs, then the test accuracy (around
  70–80% on this synthetic dataset — not a production accuracy claim), and
  saves `outputs/loss_curve.png` and `outputs/confusion_matrix.png`.
- `main.py` prints predicted priority for each sample subject, prints the
  generated study plan, and saves `outputs/study_plan.png`.

### Example Output (main.py)

```
===== DNN Priority Predictions =====
  DAA        -> Priority: High
  DBMS       -> Priority: Low
  OS         -> Priority: Medium
  Statistics -> Priority: High

===== Personalized Study Plan =====
  DAA        | Priority: High   | Study Time: 1.67 hrs
  Statistics | Priority: High   | Study Time: 1.67 hrs
  OS         | Priority: Medium | Study Time: 1.11 hrs
  DBMS       | Priority: Low    | Study Time: 0.56 hrs
```

This matches the example in the project brief: DAA (low marks, low
attendance, high difficulty) is correctly flagged as High priority and
receives more study time.

## Suggested Screenshots / Figures for the Report

1. `outputs/loss_curve.png` – training loss decreasing over epochs.
2. `outputs/confusion_matrix.png` – seaborn heatmap of predictions vs actual.
3. `outputs/study_plan.png` – bar chart of the generated study plan.
4. A screenshot of the terminal output of `train.py` (shows loss + accuracy).
5. A screenshot of the terminal output of `main.py` (shows predictions + plan).
6. A screenshot/table of `data/study_data.csv` (first 10–15 rows).
