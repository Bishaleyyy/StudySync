"""
preprocessing.py
------------------
All data preparation steps for StudySync, kept in plain NumPy/Pandas
(no scikit-learn) so every line is easy to explain in viva.

Steps performed:
1. Load the CSV.
2. Encode 'Subject_Difficulty' (Low/Medium/High) -> numbers (0/1/2).
3. Encode target 'Priority_Level' (Low/Medium/High) -> numbers (0/1/2).
4. Standardize numeric features (mean 0, std 1) using values learned
   from the TRAIN split only, then apply the same values to test data.
5. Split into train/test sets (80/20).
6. Convert everything into PyTorch tensors ready for the DNN.
"""

import numpy as np
import pandas as pd
import torch

DIFFICULTY_MAP = {"Low": 0, "Medium": 1, "High": 2}
PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2}
PRIORITY_MAP_INV = {v: k for k, v in PRIORITY_MAP.items()}

FEATURE_COLUMNS = [
    "Study_Hours_Available",
    "Previous_Marks",
    "Attendance",
    "Assignment_Score",
    "Subject_Difficulty",   # will be numeric after encoding
    "Days_Left",
]


def load_data(csv_path="data/study_data.csv"):
    """Read the raw CSV into a pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def encode_columns(df):
    """Turn text columns into numbers the network can understand."""
    df = df.copy()
    df["Subject_Difficulty"] = df["Subject_Difficulty"].map(DIFFICULTY_MAP)
    df["Priority_Level"] = df["Priority_Level"].map(PRIORITY_MAP)
    return df


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """
    Simple manual train/test split (kept dependency-free on purpose).
    Shuffles indices, then slices them into train/test.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    """
    Standardize features to mean 0 / std 1.
    IMPORTANT: mean & std are computed on TRAIN data only, then reused
    on test data. This avoids "leaking" test information into training,
    a common beginner mistake.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


def prepare_data(csv_path="data/study_data.csv"):
    """
    Full preprocessing pipeline. Returns PyTorch tensors + the
    mean/std used for scaling (needed later to scale new student data
    the exact same way when making predictions in main.py).
    """
    df = load_data(csv_path)
    df = encode_columns(df)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["Priority_Level"].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test, mean, std = standardize(X_train, X_test)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    return X_train_t, X_test_t, y_train_t, y_test_t, mean, std


def scale_new_sample(feature_row, mean, std):
    """
    Scale a single new student/subject entry using the SAME mean/std
    that were computed from the training data. feature_row must already
    have Subject_Difficulty encoded as a number (0/1/2).
    """
    feature_row = np.array(feature_row, dtype=np.float32)
    scaled = (feature_row - mean) / std
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # add batch dim
