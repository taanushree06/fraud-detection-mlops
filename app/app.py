"""Streamlit app: Credit Card Fraud Detection."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_validation import FEATURES  # noqa: E402
from src.monitoring import clear_log, log_prediction, read_log, summarize  # noqa: E402
from src.predict import load_artifact, predict_one  # noqa: E402

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def get_artifact():
    return load_artifact()


@st.cache_data
def get_samples():
    path = ROOT / "models" / "sample_inputs.json"
    return json.loads(path.read_text()) if path.exists() else {}


def key(f):
    return f"in_{f}"


# Initialise the input fields once
for f in FEATURES:
    if key(f) not in st.session_state:
        st.session_state[key(f)] = 0.0


def load_example(name):
    for f in FEATURES:
        st.session_state[key(f)] = float(get_samples()[name][f])
    st.session_state.pop("result", None)


def reset_inputs():
    for f in FEATURES:
        st.session_state[key(f)] = 0.0
    st.session_state.pop("result", None)


try:
    artifact = get_artifact()
except Exception as exc:  # noqa: BLE001
    st.error("Model file not found. Run: python src\\register_model.py")
    st.caption(str(exc))
    st.stop()

threshold = float(artifact["threshold"])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Model info")
    st.write(f"**Model:** {artifact['model_name']}")
    st.write(f"**Decision threshold:** {threshold}")
    st.write(f"**Trained at:** {artifact['trained_at']}")
    st.subheader("Test-set performance")
    for k, v in artifact["test_metrics"].items():
        st.write(f"{k.replace('_', ' ').upper()}: {v}")
    st.caption("Preprocessing + model + threshold are saved together in one file.")

st.title("Credit Card Fraud Detection")
tab_predict, tab_monitor = st.tabs(["Predict", "Monitoring"])

# ---------------- Predict tab ----------------
with tab_predict:
    st.write("Enter the transaction features, or load a real example from the test set.")
    samples = get_samples()
    b1, b2, b3, _ = st.columns([1.3, 1.5, 1, 4])
    if samples:
        b1.button("Load fraud example", on_click=load_example, args=("fraud_example",))
        b2.button("Load legitimate example", on_click=load_example, args=("legit_example",))
    b3.button("Reset", on_click=reset_inputs)

    st.subheader("Transaction features")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.number_input("Time (seconds since first transaction)", key=key("Time"),
                    min_value=0.0, step=1.0, format="%.2f")
    c2.number_input("Amount", key=key("Amount"), min_value=0.0, step=0.01, format="%.2f")

    st.caption("V1 to V28 are anonymized PCA components from the original dataset.")
    v_features = [f for f in FEATURES if f.startswith("V")]
    cols = st.columns(4)
    for i, f in enumerate(v_features):
        cols[i % 4].number_input(f, key=key(f), step=0.01, format="%.6f")

    if st.button("PREDICT", type="primary"):
        values = {f: st.session_state[key(f)] for f in FEATURES}
        try:
            result = predict_one(artifact, values)
            log_prediction(result)
            st.session_state["result"] = result
        except ValueError as exc:
            st.session_state.pop("result", None)
            st.error(f"Invalid input: {exc}")

    result = st.session_state.get("result")
    if result:
        st.divider()
        if result["is_fraud"]:
            st.error(f"Prediction: {result['prediction']}")
        else:
            st.success(f"Prediction: {result['prediction']}")
        m1, m2 = st.columns(2)
        m1.metric("Fraud Probability", f"{result['fraud_probability'] * 100:.1f}%")
        m2.metric("Decision Threshold", f"{result['threshold']}")
        st.progress(min(max(result["fraud_probability"], 0.0), 1.0))
        st.caption(f"Flagged as fraud when probability >= {result['threshold']}")

# ---------------- Monitoring tab ----------------
with tab_monitor:
    st.subheader("Prediction log")
    log = read_log()
    s = summarize(log)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total predictions", s["total"])
    k2.metric("Fraud predictions", s["fraud"])
    k3.metric("Legitimate predictions", s["legit"])
    k4.metric("Avg fraud probability", f"{s['avg_probability'] * 100:.1f}%")

    if s["total"]:
        st.write("Fraud probability of each prediction (in order)")
        st.line_chart(log["fraud_probability"].reset_index(drop=True))
        st.write("Latest predictions")
        st.dataframe(log.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
        st.caption("Stored in logs/predictions.csv")
        if st.button("Clear log"):
            clear_log()
            st.rerun()
    else:
        st.info("No predictions yet. Make one in the Predict tab, then refresh this tab.")
