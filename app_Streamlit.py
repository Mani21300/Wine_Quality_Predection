import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('linear_wine_model.pkl', 'rb'))

# Title of the web app
st.title('Wine Quality Prediction')

# Add a brief description
st.write("""
### This is a web app to predict wine quality based on various features like acidity, alcohol content, etc.
Please provide the required details to predict the quality of wine.
""")

# Input fields for user to input features
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=15.0, value=7.4)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.7)
citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=1.0, value=0.0)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=15.0, value=1.9)
chlorides = st.number_input('Chlorides', min_value=0.0, max_value=0.2, value=0.076)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0, max_value=100, value=11)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0, max_value=200, value=34)
density = st.number_input('Density', min_value=0.990, max_value=1.010, value=0.9978)
pH = st.number_input('pH', min_value=2.0, max_value=4.0, value=3.51)
sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.number_input('Alcohol', min_value=8.0, max_value=15.0, value=9.4)

# Create a list of input values
input_features = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                   free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]

# When the button is clicked, make the prediction
if st.button('Predict Quality'):
    prediction = model.predict(input_features)
    st.write(f"The predicted quality of the wine is: {prediction[0]:.2f}")
