"""Streamlit app - predictive maintenance triage for engines (decision support, not an autonomous verdict)."""
import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DEPLOY_DIR = Path(__file__).resolve().parent
MODEL_PATH = DEPLOY_DIR / "best_model.joblib"      # committed to the repository by the GitHub Actions workflow
CARD_PATH = DEPLOY_DIR / "model_card.json"
STATS_PATH = DEPLOY_DIR / "feature_stats.json"

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🔧", layout="wide")


def snake_case(name):
    """Same column standardisation as the training pipeline (e.g. 'Engine rpm' -> 'engine_rpm')."""
    return re.sub(r"\W+", "_", name.strip().lower()).strip("_")


@st.cache_resource
def load_assets():
    bundle = joblib.load(MODEL_PATH)
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}
    return bundle, card, stats


def triage(probability, balanced, high_recall):
    """Three-level decision guidance from the two saved operating points."""
    if probability >= balanced:
        return "HIGH RISK", "Prioritise this engine for inspection or maintenance review."
    if probability >= high_recall:
        return "ELEVATED RISK", "Schedule an inspection if capacity allows; keep monitoring."
    return "LOWER RISK", "Continue routine monitoring."


st.title("🔧 Predictive Maintenance for Engine Systems")
st.caption("Sensor-based engine triage. The model ranks engines by fault risk; every alert is a prompt for inspection, not a diagnosis.")

if not MODEL_PATH.exists() or not CARD_PATH.exists():
    st.error("Model files not found. Run the GitHub Actions workflow (or the notebook) to create deployment/best_model.joblib.")
    st.stop()

bundle, card, stats = load_assets()
model, features = bundle["model"], list(bundle["feature_columns"])
balanced = float(bundle["threshold"])
high_recall = float(card.get("high_recall_threshold", balanced))
test_metrics = card.get("test_metrics", {})

with st.sidebar:
    st.header("Model information")
    st.write(f"**Model:** {bundle.get('model_type', 'saved model')}")
    st.write(f"**Balanced threshold:** {balanced:.2f}")
    st.write(f"**High-recall threshold:** {high_recall:.2f}")
    if test_metrics:
        st.write(f"**Test ROC-AUC:** {test_metrics.get('ROC-AUC', float('nan')):.2f}")
        st.write(f"**Test balanced accuracy:** {test_metrics.get('Balanced Acc', float('nan')):.2f}")
    st.info("Moderate discrimination: use the score to prioritise inspections. Keep hard-limit alarms (for example coolant over-temperature) in place.")
    with st.expander("Model limitations"):
        for item in card.get("limitations", []):
            st.write("- " + item)

single_tab, batch_tab = st.tabs(["Single engine", "Batch CSV"])

with single_tab:
    st.subheader("Enter the current sensor readings")
    with st.form("engine_form"):
        columns = st.columns(3)
        inputs = {}
        for i, feature in enumerate(features):
            s = stats.get(feature, {})
            default = float(s.get("median", 1.0))
            span = float(s.get("max", default * 2) - s.get("min", 0.0))
            step = max(round(span / 200, 2), 0.01)
            with columns[i % 3]:
                inputs[feature] = st.number_input(feature.replace("_", " ").capitalize(), min_value=0.0, value=default,
                                                  step=step, format="%.2f", key=feature)
        submitted = st.form_submit_button("Assess maintenance risk", type="primary")

    if submitted:
        row = pd.DataFrame([inputs], columns=features)           # the inputs saved into a DataFrame
        probability = float(model.predict_proba(row)[:, 1][0])
        band, advice = triage(probability, balanced, high_recall)
        out_of_range = [f for f in features if stats.get(f) and not stats[f]["min"] <= row.at[0, f] <= stats[f]["max"]]
        st.markdown("---")
        left, right = st.columns([1, 2])
        left.metric("Predicted fault probability", f"{probability:.1%}", f"{probability - balanced:+.1%} vs balanced threshold")
        message = f"**{band}** - {advice}"
        (right.error if band == "HIGH RISK" else right.warning if band == "ELEVATED RISK" else right.success)(message)
        if out_of_range:
            st.warning("Outside the range seen in training (prediction less reliable): " + ", ".join(out_of_range))
        st.write("Submitted sensor row (DataFrame passed to the model):")
        st.dataframe(row)

with batch_tab:
    st.subheader("Score a CSV file of engine readings")
    st.caption("Columns: " + ", ".join(features) + " (original headers such as 'Engine rpm' are accepted).")
    uploaded = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        data.columns = [snake_case(c) for c in data.columns]
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scored = data.copy()
            scored["fault_probability"] = model.predict_proba(data[features])[:, 1]
            scored["risk_band"] = [triage(p, balanced, high_recall)[0] for p in scored["fault_probability"]]
            st.write(f"Scored {len(scored):,} engines.")
            st.bar_chart(scored["risk_band"].value_counts())
            st.dataframe(scored.sort_values("fault_probability", ascending=False).head(200))
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"),
                               file_name="scored_engines.csv", mime="text/csv")
