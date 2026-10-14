
"""Interactive dashboard for the predictive-maintenance project."""

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data import PROJECT_ROOT
from src.simulator import generate_virtual_stream
from src.features import build_features

MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.joblib"

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ AI-Powered Predictive Maintenance")
st.caption(
    "Virtual sensor simulation and machine-failure risk estimation"
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python -m src.train` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Simulation settings")

scenario = st.sidebar.selectbox(
    "Choose a virtual scenario",
    [
        "Healthy operation",
        "Gradual degradation",
        "Sudden stress",
    ],
)

steps = st.sidebar.slider(
    "Number of time steps",
    min_value=20,
    max_value=150,
    value=60,
    step=10,
)

if st.sidebar.button("Generate virtual readings"):
    st.session_state["simulation_seed"] = (
        st.session_state.get("simulation_seed", 42) + 1
    )

seed = st.session_state.get("simulation_seed", 42)

# Generate the virtual stream and score it with the trained model.
stream = generate_virtual_stream(steps_per_scenario=steps, seed=seed)
stream = stream[stream["scenario"] == scenario].copy()

X = build_features(stream)
stream["predicted_failure"] = model.predict(X).astype(int)
stream["failure_probability"] = model.predict_proba(X)[:, 1]
stream["predicted_status"] = stream["predicted_failure"].map({
    0: "No failure predicted",
    1: "Failure predicted",
})

# Summary metrics.
latest = stream.iloc[-1]
mean_probability = stream["failure_probability"].mean()
maximum_probability = stream["failure_probability"].max()
predicted_failures = int(stream["predicted_failure"].sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected scenario", scenario)
col2.metric("Latest failure probability", f"{latest['failure_probability']:.1%}")
col3.metric("Average probability", f"{mean_probability:.1%}")
col4.metric("Records predicted as failure", f"{predicted_failures} / {len(stream)}")

st.info(
    "These are model estimates from artificial sensor readings, "
    "not confirmed real-world machine failures."
)

st.subheader("Failure probability over simulated time")

probability_chart = px.line(
    stream,
    x="step",
    y="failure_probability",
    title="Estimated failure probability",
    labels={
        "step": "Simulation step",
        "failure_probability": "Estimated probability",
    },
)
probability_chart.update_yaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(probability_chart, width="stretch")

st.subheader("Virtual sensor trends")

sensor = st.selectbox(
    "Choose a sensor measurement",
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ],
)

sensor_chart = px.line(
    stream,
    x="step",
    y=sensor,
    title=f"{sensor} over simulated time",
    labels={"step": "Simulation step", sensor: sensor},
)
st.plotly_chart(sensor_chart, width="stretch")

st.subheader("Latest virtual sensor readings")

display_columns = [
    "step",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "predicted_status",
    "failure_probability",
]

st.dataframe(
    stream[display_columns].tail(10).style.format({
        "failure_probability": "{:.2%}",
    }),
    width="stretch",
)

st.download_button(
    label="Download scenario predictions (CSV)",
    data=stream.to_csv(index=False).encode("utf-8"),
    file_name="virtual_maintenance_predictions.csv",
    mime="text/csv",
)
