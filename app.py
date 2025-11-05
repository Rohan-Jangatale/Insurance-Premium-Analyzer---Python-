
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("best_model (4).pkl", "rb") as f:
    model = pickle.load(f)

st.title("💰 Insurance Premium Analyzer")
st.write("Predict your insurance charges based on your details.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Do you Smoke?", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# 🔧 Encode categorical variables like during training
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Ensure columns match the model
# Create empty columns for any missing ones (to match model input)
model_features = getattr(model, "feature_names_in_", None)
if model_features is not None:
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"💸 Estimated Insurance Cost: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
