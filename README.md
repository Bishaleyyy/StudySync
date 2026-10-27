# StudySync – A Neural Network Based Personalized Study Planning System

A personalized study planner. A small PyTorch neural network predicts how much priority each subject needs (Low / Medium / High), and a simple planner turns that into hours per subject.

## Live Demo

**[studysync-bishaleyyy.streamlit.app](https://studysync-bishaleyyy.streamlit.app/)**

The app is hosted on Streamlit's free tier, so it may be asleep if unused for a while. Click "Wake up" and wait a few seconds.

## How It Works

1. Enter your subjects with marks, attendance, assignment score, difficulty and days left.
2. The neural network predicts a priority for each subject.
3. The planner splits your daily study hours by priority (High = 3, Medium = 2, Low = 1).

## Tech Stack

Python, PyTorch, NumPy, Pandas, Matplotlib, Seaborn, Streamlit

## Run Locally

```bash
git clone https://github.com/Bishaleyyy/StudySync.git
cd StudySync
pip install -r requirements.txt
streamlit run app.py
```

The trained model is already included in `model/`, so the app runs right away.

To retrain the model:

```bash
python create_dataset.py
python train.py
```

## Project Files

- `app.py` - Streamlit web app
- `model.py` - neural network (6 → 16 → 8 → 3)
- `preprocessing.py` - encoding, scaling, train/test split
- `train.py` - trains and saves the model
- `planner.py` - allocates study hours by priority
- `create_dataset.py` - generates the synthetic dataset
- `main.py` - command-line demo

## Note

The dataset is synthetic (rule-based with random noise), made for academic demonstration. It is not real student data. The model reaches about 80% accuracy on a 60-sample test set.
