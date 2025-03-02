import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction App")

# Collect user input
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Class: {prediction}")
