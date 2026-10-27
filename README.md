# StudySync

**A personalized study planner that uses a PyTorch neural network to predict subject priorities and build a daily study plan.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://studysync-bishaleyyy.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

A BSc. CSIT Neural Networks project. For each subject, a small Deep Neural Network predicts whether it needs **Low**, **Medium** or **High** priority. A simple planner then splits your available study hours accordingly.

## Live Demo

**[studysync-bishaleyyy.streamlit.app](https://studysync-bishaleyyy.streamlit.app/)**

The app runs on Streamlit's free tier, so it may be asleep if unused for a while. Click **Wake up** and wait a few seconds.

## Features

- Add as many subjects as you like, each with its own details
- DNN predicts a Low / Medium / High priority per subject, with a confidence value
- Generates a study plan (cards, bar chart and table) from your available hours
- Model is loaded once and never retrained while using the app

## How It Works

1. Enter marks, attendance, assignment score, difficulty and days left for each subject.
2. The inputs are scaled with the training data's mean and standard deviation.
3. The DNN predicts the priority class for each subject.
4. The planner (`planner.py`, not a neural network) splits your hours by weight: **High = 3, Medium = 2, Low = 1**.

**Model:** `Input (6) → Linear(6, 16) → ReLU → Linear(16, 8) → ReLU → Linear(8, 3)`, trained with cross-entropy loss and the Adam optimizer.

| Input | Range in training data |
|---|---|
| Study hours available | 1–6 |
| Previous marks | 20–99 |
| Attendance (%) | 40–99 |
| Assignment score (%) | 0–99 |
| Subject difficulty | Low / Medium / High |
| Days left | 1–29 |

## Results

On a held-out test set of 60 samples, the model reaches **80% accuracy** (48 of 60 correct).

| Training loss | Confusion matrix |
|---|---|
| ![Loss curve](outputs/loss_curve.png) | ![Confusion matrix](outputs/confusion_matrix.png) |

> **Note:** The dataset is **synthetic** (300 samples generated from a hand-made rule plus random noise). It is not real student data, so this accuracy only shows the model can learn that rule. It is a prototype for academic demonstration. The test set is also small, and retraining may give slightly different numbers.

## Tech Stack

Python, PyTorch, NumPy, Pandas, Matplotlib, Seaborn, Streamlit

## Run Locally

The trained model is included, so no training is needed.

```bash
git clone https://github.com/Bishaleyyy/StudySync.git
cd StudySync
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

To retrain the model (optional, overwrites `data/`, `model/` and `outputs/`):

```bash
python create_dataset.py
python train.py
```

## Project Structure

```
StudySync/
├── app.py               # Streamlit web app
├── create_dataset.py    # generates the synthetic dataset
├── preprocessing.py     # encoding, scaling, train/test split
├── model.py             # DNN architecture
├── train.py             # trains, evaluates and saves the model
├── planner.py           # allocates study hours by priority
├── main.py              # command-line demo
├── requirements.txt
├── data/                # study_data.csv
├── model/               # trained weights and scaler values
└── outputs/             # loss curve and confusion matrix
```

## Author

Built by [Bishal](https://github.com/Bishaleyyy)
