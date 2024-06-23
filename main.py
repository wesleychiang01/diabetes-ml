import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

# Load the PCA model
pca = joblib.load('pca_model.pkl')

# Define the input fields for the user
st.title('Diabetes Prediction App')
st.write('Please enter the following details:')

# Input fields
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.number_input('Age', min_value=0, max_value=120, value=0)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
smoking_history = st.selectbox('Smoking History', ['never', 'no info', 'current', 'ever', 'former', 'not current'])
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0, format="%.1f")
HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, value=0.0, format="%.1f")
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500, value=0)

# Mean and Standard Deviation applied by StandardScaler
mean_values = [4.16065151e-01, 4.17943257e+01, 7.76007322e-02, 4.08025295e-02,
               2.23115886e+00, 2.73214611e+01, 5.53260874e+00, 1.38218231e+02]
std_deviation = [0.49328427, 22.46283076, 0.26754226, 0.19783246,
                 1.87995317, 6.76768037, 1.07322644, 40.90955861]


# Function to preprocess input data
def preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    # Encode gender (assuming Female=0, Male=1)
    gender_encoded = 0 if gender == 'Female' else 1

    # Encode hypertension and heart_disease (assuming Yes=1, No=0)
    hypertension_encoded = 1 if hypertension == 'Yes' else 0
    heart_disease_encoded = 1 if heart_disease == 'Yes' else 0

    # Encode smoking_history based on provided mapping
    smoking_map = {'never': 4, 'no info': 0, 'current': 1, 'ever': 2, 'former': 3, 'not current': 5}
    smoking_history_encoded = smoking_map.get(smoking_history, 0)

    # Create a Pandas DataFrame with the processed input
    data = pd.DataFrame({
        'gender': [gender_encoded],
        'age': [age],
        'hypertension': [hypertension_encoded],
        'heart_disease': [heart_disease_encoded],
        'smoking_history': [smoking_history_encoded],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Apply standardization using mean and standard deviation
    for i, col in enumerate(data.columns):
        data[col] = (data[col] - mean_values[i]) / std_deviation[i]

    return data


# Predict diabetes status
if st.button('Predict'):
    input_data = preprocess_input(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level,
                                  blood_glucose_level)
    # st.info('Scaled Data:\n' + str(input_data))

    input_array = input_data.to_numpy()

    # Apply PCA transformation
    X_pca = pca.transform(input_array)
    # st.info('X_pca:\n' + str(X_pca))

    # Make prediction using your TensorFlow/Keras model
    prediction = model.predict(X_pca)
    prediction_class = np.argmax(prediction, axis=1)

    if prediction_class[0] == 1:
        st.error('The model predicts that you have diabetes.')
    else:
        st.success('The model predicts that you do not have diabetes.')
