import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("best_model (4).pkl", "rb") as f:
    model = pickle.load(f)

st.title("💰 Insurance Cost Prediction App")
st.write("Predict your medical insurance charges based on personal details.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Convert to DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💸 Estimated Insurance Cost: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

