import streamlit as st
import joblib
import numpy as np

# Load the pre-trained KNN model and scaler
model = joblib.load(r'C:\Users\user\Documents\Car_Price_Prediction\Car_Price_Prediction\car_purchase_predictor.pkl')
scaler = joblib.load(r'C:\Users\user\Documents\Car_Price_Prediction\Car_Price_Prediction\scaler.pkl')

st.title('Car Purchase Prediction')

# Input fields
age = st.number_input('Enter Age:', min_value=18, max_value=100, value=25)
salary = st.number_input('Enter Salary:', min_value=0, value=50000)

if st.button('Predict'):
    # Prepare the data (scaling using the loaded scaler)
    input_data = np.array([[age, salary]])
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success('Customer is interested to purchase the car 😊')
    else:
        st.warning('Customer is not interested to purchase a car 😔')
