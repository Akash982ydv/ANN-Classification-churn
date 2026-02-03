import streamlit as st
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their profile data.")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_objects():
    model = load_model('model.h5')
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    return model, scaler, onehot_encoder_geo, label_encoder_gender

loaded_model, scaler, loaded_onehot_encoder_geo, loaded_label_encoder_gender = load_model_and_objects()
# Define input fields
st.header("Enter Customer Profile Data")
input_data = {
    'CreditScore': st.number_input("Credit Score", min_value=300, max_value=850, value=600),
    'Geography': st.selectbox("Geography", options=['France', 'Spain', 'Germany']),
    'Gender': st.selectbox("Gender", options=['Male', 'Female']),
    'Age': st.number_input("Age", min_value=18, max_value=100, value=30),
    'Tenure': st.number_input("Tenure (years)", min_value=0, max_value=10, value=3),
    'Balance': st.number_input("Balance", min_value=0.0, value=50000.0),
    'NumOfProducts': st.number_input("Number of Products", min_value=1, max_value=5, value=2),
    'HasCrCard': st.selectbox("Has Credit Card", options=[0, 1]), 
    'IsActiveMember': st.selectbox("Is Active Member", options=[0, 1]),
    'EstimatedSalary': st.number_input("Estimated Salary", min_value=0.0, value=60000.0)
}
if st.button("Predict Churn"):
    ## One-hot encode 'Geography'
    geo_encoded = loaded_onehot_encoder_geo.transform([[input_data['Geography']]])
    geo_df = pd.DataFrame(geo_encoded, columns=loaded_onehot_encoder_geo.get_feature_names_out(['Geography']))
    ## encode categorical variables
    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = loaded_label_encoder_gender.transform(input_df['Gender'])
    ## concatenation
    input_df_final = pd.concat([input_df.drop('Geography', axis=1), geo_df], axis=1)
    ##scale the input data
    input_scaled = scaler.transform(input_df_final)
    ## predict using the loaded model
    prediction = loaded_model.predict(input_scaled)
    prediction_class = (prediction > 0.5).astype(int)
    st.write(f"Predicted class: {prediction_class[0][0]}, Probability of Churn: {prediction[0][0]:.4f}")
    if prediction_class[0][0] >0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")
