import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(
    page_title="Insurance Premium Analyzer 💰",
    page_icon="💸",
    layout="centered"
)

# Load model
with open("best_model (4).pkl", "rb") as f:
    model = pickle.load(f)

# --- HEADER SECTION ---
st.markdown(
    """
    <div style="text-align:center; padding:10px 0;">
        <h1 style="color:#2E8B57;">💰 Insurance Premium Analyzer</h1>
        <p style="font-size:18px;">Predict your estimated medical insurance charges instantly!</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- INPUT FORM ---
st.markdown("### 🧾 Enter Your Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=25)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0)
with col2:
    sex = st.selectbox("🚻 Sex", ["male", "female"])
    smoker = st.selectbox("🚬 Smoker", ["yes", "no"])
    region = st.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])

st.divider()

# --- DATA PREPARATION ---
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

input_encoded = pd.get_dummies(input_df, drop_first=True)

model_features = getattr(model, "feature_names_in_", None)
if model_features is not None:
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

# --- PREDICTION BUTTON ---
if st.button("🔮 Predict My Insurance Cost"):
    try:
        prediction = model.predict(input_encoded)[0]
        st.success("✅ Prediction Successful!")
        
        st.markdown(
            f"""
            <div style="background-color:#F0FFF0; padding:20px; border-radius:15px; text-align:center;">
                <h2 style="color:#006400;">💸 Estimated Insurance Cost</h2>
                <h1 style="color:#2E8B57;">₹{prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            ---
            🧠 **Tip:** Your premium depends heavily on BMI and smoking habits.
            Lowering BMI or quitting smoking can significantly reduce insurance costs.
            """
        )

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

# --- FOOTER ---
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px; color:gray;">
    Built with ❤️ using Streamlit | Rohan Jangatale © 2025
    </p>
    """,
    unsafe_allow_html=True
)
