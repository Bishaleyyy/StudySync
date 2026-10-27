"""
app.py - Streamlit interface for StudySync.
Reuses model.py, preprocessing.py and planner.py (no ML logic here).
Run:  streamlit run app.py   (after: python create_dataset.py, python train.py)
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import StudySyncDNN
from planner import generate_multi_day_plan, total_hours_per_subject
from preprocessing import DIFFICULTY_MAP, PRIORITY_MAP_INV, scale_new_sample

BASE = Path(__file__).resolve().parent
MODEL_PATH, MEAN_PATH, STD_PATH = (BASE / "model" / n for n in
                                   ("studysync_model.pth", "scaler_mean.npy", "scaler_std.npy"))

COLORS = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#34d399"}
ORDER = ["High", "Medium", "Low"]
# Default form values (name, marks, attendance, assignment, difficulty, days_left)
DEFAULTS = [("DAA", 45, 70, 55, "High", 10), ("DBMS", 80, 90, 85, "Low", 10),
            ("OS", 60, 75, 65, "Medium", 10)]

st.set_page_config(page_title="StudySync", page_icon="📘", layout="wide")

st.markdown("""
<style>
.stApp { background:#111318; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], .stAppDeployButton { display:none; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2.5rem; max-width:1150px; }
.logo { display:flex; align-items:center; gap:10px; }
.logo-mark { width:36px; height:36px; border-radius:10px; background:#7c5cf0; display:flex;
  align-items:center; justify-content:center; font-size:18px; }
.logo-text { font-size:1.2rem; font-weight:700; color:#fff; line-height:1.1; }
.logo-sub { font-size:.72rem; color:#8b90a3; }
.page-title h1 { margin:0; font-size:1.8rem; color:#fff; }
.page-title p { margin:4px 0 20px 0; color:#9aa0b4; }
.card { background:#1a1d27; border:1px solid #2a2e3d; border-radius:12px; padding:16px 18px; height:100%; }
.card .note { color:#9aa0b4; font-size:.85rem; margin-top:4px; }
.badge { display:inline-block; padding:2px 11px; border-radius:999px; font-size:.78rem;
  font-weight:600; border:1px solid; }
.subject-name { font-size:1.1rem; font-weight:700; color:#fff; }
.hours { font-size:1.5rem; font-weight:700; color:#b9a7ff; }
div[data-testid="stFormSubmitButton"] > button {
  background:#7c5cf0; color:#fff; border:none; border-radius:10px; padding:.55rem 1.3rem; font-weight:600; }
div[data-testid="stFormSubmitButton"] > button:hover { background:#6a49e0; color:#fff; }
div[data-testid="stExpander"] { border:1px solid #2a2e3d; border-radius:12px; background:#1a1d27; }
</style>
""", unsafe_allow_html=True)


# ---------------- Cached loader (nothing is retrained here) ----------------
@st.cache_resource
def load_model():
    """Load weights + scaler values saved by train.py (None if missing)."""
    if not all(p.exists() for p in (MODEL_PATH, MEAN_PATH, STD_PATH)):
        return None
    net = StudySyncDNN(input_size=6)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return net, np.load(MEAN_PATH), np.load(STD_PATH)


def predict(net, mean, std, row):
    """Same steps as main.py: scale -> model -> argmax (+ softmax confidence)."""
    with torch.no_grad():
        probs = torch.softmax(net(scale_new_sample(row, mean, std)), dim=1)[0]
    idx = int(torch.argmax(probs))
    return PRIORITY_MAP_INV[idx], float(probs[idx])


def badge(p):
    c = COLORS[p]
    return f'<span class="badge" style="color:{c};border-color:{c};background:{c}22;">{p} priority</span>'


def plan_chart(df, chart_title):
    """Dark-styled bar chart of allocated hours, coloured by priority."""
    scale = alt.Scale(domain=ORDER, range=[COLORS[p] for p in ORDER])
    chart = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
             .encode(x=alt.X("Subject:N", sort=None, axis=alt.Axis(labelAngle=0)),
                     y=alt.Y("Hours:Q", title="Allocated hours"),
                     color=alt.Color("Priority:N", scale=scale),
                     tooltip=list(df.columns))
             .properties(height=300, title=chart_title))
    st.altair_chart(chart.configure(background="transparent").configure_view(strokeWidth=0)
                    .configure_axis(labelColor="#b7bde0", titleColor="#b7bde0", gridColor="#ffffff14")
                    .configure_title(color="#e6e9f5").configure_legend(labelColor="#b7bde0", titleColor="#b7bde0"))


# ---------------- Header ----------------
st.markdown('<div class="logo"><div class="logo-mark">📘</div><div><div class="logo-text">StudySync</div>'
            '<div class="logo-sub">Neural-network study planner</div></div></div>', unsafe_allow_html=True)
st.divider()

loaded = load_model()
if loaded is None:
    st.error("Trained model files not found in `model/`.\n\nRun once: `python create_dataset.py` "
             "then `python train.py`, and refresh this page.")
    st.stop()
net, mean, std = loaded

# ---------------- Input form ----------------
st.markdown('<div class="page-title"><h1>Plan Generator</h1><p>Enter your subjects, then let the model '
            'decide where your time matters most.</p></div>', unsafe_allow_html=True)
n = int(st.number_input("How many subjects?", min_value=1, value=3, step=1))

with st.form("student_form"):
    total_hours = st.slider("Study hours available per day", 1, 6, 5,
                            help="Hours you can study EACH day. Used as the Study_Hours_Available feature and split across subjects every day until their exams.")
    subjects = []
    for i in range(n):
        d = DEFAULTS[i] if i < len(DEFAULTS) else (f"Subject {i + 1}", 60, 75, 65, "Medium", 10)
        with st.expander(f"Subject {i + 1}", expanded=True):
            name = st.text_input("Subject name", d[0], key=f"name_{i}")
            a, b, c = st.columns(3)
            marks = a.slider("Previous marks", 20, 99, d[1], key=f"marks_{i}")
            att = b.slider("Attendance (%)", 40, 99, d[2], key=f"att_{i}")
            asg = c.slider("Assignment score (%)", 0, 99, d[3], key=f"asg_{i}")
            e, f = st.columns(2)
            diff = e.selectbox("Subject difficulty", list(DIFFICULTY_MAP),
                               index=list(DIFFICULTY_MAP).index(d[4]), key=f"diff_{i}")
            days = f.slider("Days left before exam", 1, 29, d[5], key=f"days_{i}")
            subjects.append((name.strip() or f"Subject {i + 1}", marks, att, asg, diff, days))
    submitted = st.form_submit_button("Generate Study Plan")

# ---------------- Prediction + planner ----------------
if submitted:
    names = [s[0] for s in subjects]
    if len(set(names)) != len(names):
        st.warning("Please give each subject a different name.")
        st.stop()
    priorities, confidences = {}, {}
    for name, marks, att, asg, diff, days in subjects:
        # Feature order must match FEATURE_COLUMNS in preprocessing.py
        row = [total_hours, marks, att, asg, DIFFICULTY_MAP[diff], days]
        priorities[name], confidences[name] = predict(net, mean, std, row)
    days_map = {s[0]: s[5] for s in subjects}
    schedule = generate_multi_day_plan(priorities, days_map, total_hours)
    st.session_state["result"] = dict(priorities=priorities, confidences=confidences, total=total_hours,
                                      days=days_map, schedule=schedule, plan=schedule[0],
                                      totals=total_hours_per_subject(schedule))

# ---------------- Results ----------------
r = st.session_state.get("result")
if r:
    st.subheader("Predicted priorities & study plan")
    ordered = sorted(r["plan"].items(), key=lambda kv: kv[1], reverse=True)
    cols = st.columns(min(len(ordered), 4))
    for i, (subject, hrs) in enumerate(ordered):
        with cols[i % len(cols)]:
            st.markdown(f'<div class="card"><div class="subject-name">{subject}</div>'
                        f'<div style="margin:8px 0 10px 0;">{badge(r["priorities"][subject])}</div>'
                        f'<div class="hours">{hrs} hrs today</div>'
                        f'<div class="note">Total till exam: {r["totals"][subject]} hrs ({r["days"][subject]} days)</div>'
                        f'<div class="note">Model confidence: {r["confidences"][subject] * 100:.0f}%</div></div>',
                        unsafe_allow_html=True)
            st.write("")
    table = pd.DataFrame({"Subject": [s for s, _ in ordered], "Hours": [h for _, h in ordered],
                          "Priority": [r["priorities"][s] for s, _ in ordered]})
    plan_chart(table, f"Today's study plan ({r['total']} hrs of {r['total']} hrs/day)")
    detail = table.rename(columns={"Hours": "Today (hrs)"})
    detail["Days left"] = [r["days"][s] for s in detail["Subject"]]
    detail["Total till exam (hrs)"] = [r["totals"][s] for s in detail["Subject"]]
    st.dataframe(detail, hide_index=True)
    with st.expander(f"Day-by-day schedule ({len(r['schedule'])} days, {r['total']} hrs/day)"):
        sched_df = pd.DataFrame(r["schedule"]).fillna(0.0)
        sched_df.insert(0, "Day", [f"Day {i + 1}" for i in range(len(sched_df))])
        sched_df["Day total"] = sched_df.drop(columns="Day").sum(axis=1).round(2)
        st.dataframe(sched_df, hide_index=True)

st.caption("StudySync · BSc. CSIT Neural Networks project")
