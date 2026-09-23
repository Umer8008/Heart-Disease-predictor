import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_features = joblib.load('features.pkl')
st.title("Heart Disease Prediction")
st.write("Please enter the following details:")
age=st.slider("Age", 20, 100, 30)
sex=st.selectbox("Sex", ["Male", "Female"])
chest_pain_type=st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_blood_pressure=st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol=st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar=st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg=st.slider("Resting Electrocardiographic Results", 0, 2, 0, format=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
max_heart_rate=st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_induced_angina=st.selectbox("Exercise Induced Angina", ["Yes", "No"])
depression=st.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 1.0)
slope=st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])


if st.button("Predict"):
    raw_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "chest_pain_type": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_type),
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": 1 if fasting_blood_sugar == "Yes" else 0,
        "rest_ecg": rest_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina": 1 if exercise_induced_angina == "Yes" else 0,
        "depression": depression,
        "slope": ["Upsloping", "Flat", "Downsloping"].index(slope)
    }
    input_data = pd.DataFrame([raw_data], columns=expected_features)
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_features]
    prediction=model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("The model predicts that you may have heart disease. Please consult a doctor for further evaluation.") 
    else:
        st.success("The model predicts that you are unlikely to have heart disease. However, please consult a doctor for a comprehensive evaluation.")  
               