import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page settings
st.set_page_config(page_title="💳 Fraud Detection", layout="wide")

# Load model, scaler, and example data
model = joblib.load("random_forest_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")
frauds, legits = joblib.load("examples.pkl")

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This smart ML app predicts whether a credit card transaction is **fraudulent or legitimate**,  
using 30 anonymized features (`V0–V28`, `Time`, `Amount`).

🧪 **Select an example or enter your own values.**  
📊 **Get a prediction with fraud probability and actual label for validation.**
""")

# Session state
if "input_values" not in st.session_state:
    st.session_state.input_values = [0.0] * 30
if "actual_label" not in st.session_state:
    st.session_state.actual_label = None

# Sidebar
st.sidebar.title("🔍 Select Input Data Type")
example_type = st.sidebar.radio(
    "Choose input type:",
    ["Custom Input", "Random Fraud Example", "Random Legit Example"]
)

# Handle example loading
if example_type == "Random Fraud Example":
    row = frauds.sample(1).iloc[0]
    st.session_state.input_values = row.values.tolist()
    st.session_state.actual_label = 1

elif example_type == "Random Legit Example":
    row = legits.sample(1).iloc[0]
    st.session_state.input_values = row.values.tolist()
    st.session_state.actual_label = 0

elif example_type == "Custom Input":
    st.session_state.actual_label = None

# Input fields (30 features)
cols = st.columns(3)
input_values = []
for i in range(30):
    label = f"V{i}" if i != 28 else "Amount"
    with cols[i % 3]:
        val = st.number_input(
            label,
            value=float(st.session_state.input_values[i]),
            step=0.01,
            format="%.6f",
            key=f"input_{i}"
        )
        input_values.append(val)

# Convert to array
input_array = np.array(input_values).reshape(1, -1)

# Predict button
if st.button("🚀 Predict"):
    scaled = scaler.transform(input_array)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    st.subheader("🧾 Prediction Result")
    if prediction == 1:
        st.error("⚠️ **Fraud Detected**")
    else:
        st.success("✅ **Legitimate Transaction**")

    st.info(f"📈 **Probability of Fraud:** {probability:.4f}")

    # Show actual label
    if st.session_state.actual_label is not None:
        label_text = "Fraud" if st.session_state.actual_label == 1 else "Legit"
        st.markdown(f"🧷 **Actual Label from Dataset:** `{label_text}`")
        if prediction != st.session_state.actual_label:
            st.warning("⚠️ Prediction does not match actual label (normal in real-world ML).")
