# app.py

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('water_model.pkl')

st.title("💧 Water Potability Prediction App")

st.markdown("Enter the water quality values below:")

# 9 input sliders/fields
ph = st.slider("pH Level", 0.0, 14.0, 6.5)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=150.0)
solids = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, max_value=50000.0, value=10000.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, max_value=15.0, value=7.0)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=250.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, max_value=1000.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, max_value=50.0, value=15.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, max_value=150.0, value=80.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=3.0)

# Combine inputs into a single array
input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                        conductivity, organic_carbon, trihalomethanes, turbidity]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ The water is safe for drinking.")
    else:
        st.error("❌ The water is not safe for drinking.")
