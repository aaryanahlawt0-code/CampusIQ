"""Streamlit dashboard for campus crowd prediction."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "campus_crowd.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_comparison.csv"

st.set_page_config(page_title="Campus Crowd Predictor", page_icon="🏫", layout="wide")
st.title("🏫 Campus Crowd Prediction System")
st.caption("Statistical Machine Learning | Synthetic demonstration dataset")

if not MODEL_PATH.exists():
    st.error("Model not found. First run: python train.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
df = pd.read_csv(DATA, parse_dates=["date"])

def crowd_label(pct):
    if pct < 35:
        return "Low"
    if pct < 70:
        return "Moderate"
    return "High"

st.sidebar.header("Prediction settings")
location = st.sidebar.selectbox("Campus location", ["Library", "Cafeteria"])
date = st.sidebar.date_input("Prediction date", value=pd.Timestamp.today().date())
hour = st.sidebar.slider("Hour of day", 8, 22, 13)
exam = st.sidebar.checkbox("Exam period")
event = st.sidebar.checkbox("Campus event")
dow = pd.Timestamp(date).dayofweek
weekend = int(dow >= 5)

input_df = pd.DataFrame([{
    "hour_sin": np.sin(2 * np.pi * hour / 24),
    "hour_cos": np.cos(2 * np.pi * hour / 24),
    "day_sin": np.sin(2 * np.pi * dow / 7),
    "day_cos": np.cos(2 * np.pi * dow / 7),
    "is_weekend": weekend,
    "exam_period": int(exam),
    "campus_event": int(event),
    "location": location,
}])

prediction = max(0, float(model.predict(input_df)[0]))
capacity = 300 if location == "Library" else 180
prediction = min(prediction, capacity)
pct = 100 * prediction / capacity
label = crowd_label(pct)

left, right, third = st.columns(3)
left.metric("Predicted people", f"{prediction:.0f}")
right.metric("Estimated capacity used", f"{pct:.1f}%")
third.metric("Crowd level", label)

st.progress(int(round(pct)), text=f"{location}: approximately {prediction:.0f} of {capacity} places occupied")

st.info("This is a synthetic-data academic prototype. Predictions are not live measurements of a real campus.")

st.subheader("Historical crowd patterns")
loc_df = df[df["location"] == location]
hourly = loc_df.groupby("hour")["occupancy_count"].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly.index, hourly.values, marker="o")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Average people")
ax.set_title(f"Average hourly occupancy — {location}")
ax.set_xticks(range(8, 23))
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

if METRICS_PATH.exists():
    st.subheader("Model comparison")
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption(f"Selected prediction model: {bundle['model_name']}")
else:
    st.warning("Model comparison report not found. Run python train.py.")
