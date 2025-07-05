import streamlit as st
import numpy as np
import joblib

# Set page settings
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# Load model and scaler
model = joblib.load("random_forest_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This smart ML app predicts whether a credit card transaction is **fraudulent or legitimate**
using 30 anonymized features (`V0–V28`, `Time`, `Amount`).

Enter your own values below to get predictions.
""")

# Input fields (30 total in 3 columns)
cols = st.columns(3)
input_values = []

for i in range(30):
    label = f"V{i}" if i != 28 else "Amount"
    with cols[i % 3]:
        val = st.number_input(
            label,
            value=0.0,
            step=0.01,
            format="%.6f",
            key=f"input_{i}"
        )
        input_values.append(val)

# Convert to array
input_array = np.array(input_values).reshape(1, -1)

# Predict
if st.button("🚀 Predict"):
    try:
        scaled = scaler.transform(input_array)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        st.subheader("🧾 Prediction Result")
        if prediction == 1:
            st.error("⚠️ **Fraud Detected**")
        else:
            st.success("✅ **Legitimate Transaction**")

        st.info(f"📈 **Probability of Fraud:** {probability:.4f}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
