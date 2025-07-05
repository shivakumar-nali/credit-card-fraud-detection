import streamlit as st
import numpy as np
import joblib

# Set page settings
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")

# Load model, scaler, and example inputs
model = joblib.load("random_forest_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")
examples = joblib.load("examples.pkl")

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This smart ML app predicts whether a credit card transaction is **fraudulent or legitimate**  
using 30 anonymized features (`V0–V28`, `Time`, `Amount`).

🧪 Select an example or enter your own values.  
📊 Get a prediction with **fraud probability** and **actual label** for validation.
""")

# Session state to hold input values and label
if "input_values" not in st.session_state:
    st.session_state.input_values = [0.0] * 30
if "actual_label" not in st.session_state:
    st.session_state.actual_label = None

# Sidebar for example type
st.sidebar.title("🔍 Select Input Data Type")
example_type = st.sidebar.radio("Choose input type:", ["Custom Input", "Random Fraud Example", "Random Legit Example"])

# Load pre-saved example based on choice
if example_type == "Random Fraud Example":
    st.session_state.input_values = examples["fraud"].tolist()
    st.session_state.actual_label = 1

elif example_type == "Random Legit Example":
    st.session_state.input_values = examples["legit"].tolist()
    st.session_state.actual_label = 0

elif example_type == "Custom Input":
    st.session_state.actual_label = None

# Input fields (30 total in 3 columns)
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

        # Show actual label if example was used
        if st.session_state.actual_label is not None:
            label = st.session_state.actual_label
            label_text = "Fraud" if label == 1 else "Legit"
            st.markdown(f"🧷 **Actual Label from Example:** `{label_text}`")

            if prediction != label:
                st.warning("⚠️ Prediction does not match actual label — this can happen in real-world ML.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
