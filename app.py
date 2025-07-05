# Load model and scaler
model = joblib.load("random_forest_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the example arrays
examples = joblib.load("examples.pkl")

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This smart ML app predicts whether a credit card transaction is **fraudulent or legitimate**
using 30 anonymized features (`V0–V28`, `Time`, `Amount`).

🧪 Select an example or enter your own values.
📊 Get a prediction with **fraud probability**.
""")

# Session state to hold input values
if "input_values" not in st.session_state:
    st.session_state.input_values = [0.0] * 30

# Sidebar for example type
st.sidebar.title("🔍 Select Input Data Type")
example_type = st.sidebar.radio("Choose input type:", ["Custom Input", "Example Fraud", "Example Legit"])

# Load example if selected
if example_type == "Example Fraud":
    st.session_state.input_values = examples["fraud"].tolist()

elif example_type == "Example Legit":
    st.session_state.input_values = examples["legit"].tolist()

# Input fields
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

    except Exception as e:
        st.error(f"Something went wrong: {e}")
