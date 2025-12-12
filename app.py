import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_heart_disease

st.title("❤️ Heart Disease Prediction System")
st.markdown("This app predicts whether a person is likely to have heart disease based on health parameters.")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

def user_input_features():
    age = st.sidebar.number_input("Age", 1, 120, 30)
    sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("🔍 Input Data Preview")
st.write(input_df)

# Prediction button
if st.button("Predict Heart Disease"):
    # Convert DataFrame to list for prediction
    input_list = input_df.iloc[0].tolist()
    prediction = predict_heart_disease(input_list)
    result = "🩺 The patient **has heart disease.**" if prediction == 1 else "💚 The patient **does not have heart disease.**"
    st.success(result)
   
