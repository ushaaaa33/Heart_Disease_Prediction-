import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_heart_disease(input_data):
    """
    Predict heart disease based on input features.

    Args:
    input_data (list or array): List of 13 features in order:
        [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    Returns:
    int: 1 if heart disease is predicted, 0 otherwise.
    """
    # Convert to DataFrame with feature names to avoid warnings
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_df)

    return prediction[0]
