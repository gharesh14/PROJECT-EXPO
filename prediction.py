import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Real-Time Data Prediction")

# File uploader for the model
uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

if uploaded_model is not None:
    # Load the model and feature names from the uploaded file
    with uploaded_model as f:
        model, feature_names = pickle.load(f)

    st.write(f"Model loaded successfully with feature names: {feature_names}")

    # Create input fields for the features
    input_data = {}
    for feature in feature_names:
        input_value = st.number_input(f"Enter value for {feature}", value=0.0, format="%.2f")
        input_data[feature] = input_value

    # Predict button
    if st.button("Make Prediction"):
        # Convert input data to a DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Check if the model is compatible with the input features
        if set(feature_names) != set(input_df.columns):
            st.error("The feature names in the model and the input data do not match.")
        else:
            # Make prediction
            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")
else:
    st.info("Please upload a trained model to start making predictions.")
