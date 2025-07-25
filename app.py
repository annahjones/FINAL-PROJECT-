
import streamlit as st
import numpy as np
import pickle
import os

# Title
st.set_page_config(page_title="Water Quality Predictor", page_icon="💧")
st.title("💧 Water Potability Predictor")
st.markdown("Enter water chemical parameters to determine if it's safe for drinking.")

# Input fields
ph = st.slider("pH Level", 0.0, 14.0, 6.5)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=150.0)
solids = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, max_value=50000.0, value=10000.0)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, max_value=500.0, value=250.0)

# Prediction section
if st.button("Predict"):
    try:
        # Check if model exists
        if not os.path.exists("model.pkl"):
            st.error("Model file (model.pkl) not found. Please upload the trained model.")
        else:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)

            features = np.array([[ph, hardness, solids, sulfate]])
            prediction = model.predict(features)
            if prediction[0] == 1:
                st.success("✅ Water is SAFE for drinking.")
            else:
                st.error("❌ Water is NOT safe for drinking.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
