import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer   # ✅ Added this line

# Load model and scaler
model = joblib.load(r"streamlit_and_deployment/best_model.pkl")
scaler = joblib.load(r"streamlit_and_deployment/scaler.pkl")


# ✅ Load dataset to get feature names
data = load_breast_cancer()

st.title('Breast Cancer Prediction App')

st.write("""
Predict breast cancer based on the features of the cell nuclei.
Please enter the values for each feature below:
""")

# Create input fields for each feature
input_data = {}
for feature_name in data.feature_names:
    input_data[feature_name] = st.number_input(f'Enter {feature_name}:', value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Scale the input data
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction Result:')
    if prediction[0] == 0:
        st.write('The model predicts: Malignant (Cancerous)')
        st.write(f'Probability of being Malignant: {prediction_proba[0][0]:.2f}')
        st.write(f'Probability of being Benign: {prediction_proba[0][1]:.2f}')
    else:
        st.write('The model predicts: Benign (Non-cancerous)')
        st.write(f'Probability of being Benign: {prediction_proba[0][1]:.2f}')
        st.write(f'Probability of being Malignant: {prediction_proba[0][0]:.2f}')
