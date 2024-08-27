import streamlit as st
import pandas as pd
from ml_utility1 import read_data, preprocess_data

st.title("No Code Machine Learning")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = read_data(uploaded_file)
    st.dataframe(df.head())

    # Display initial missing values by column
    st.subheader("Initial Missing Values by Column")
    initial_missing_values = df.isnull().sum()
    st.write(initial_missing_values)

    # Model and preprocessing options
    target_column = st.selectbox("Select the Target Column", list(df.columns))
    scaler_type = st.selectbox("Select a scaler", ["standard", "minmax"])
    
    if st.button("Preprocess Data"):
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
        
        # Store in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("Data preprocessed and ready for model comparison!")

        # Show missing values after preprocessing
        st.subheader("Missing Values by Column After Preprocessing")
        df_preprocessed = pd.concat([X_train, y_train], axis=1)  # Combine features and target to check missing values
        missing_values_after_preprocess = df_preprocessed.isnull().sum()
        st.write(missing_values_after_preprocess)

        # Add "Next" button to set the page state for visualization
        if st.button("Next"):
            st.session_state.page = "visualization"

else:
    st.info("Please upload a dataset to begin.")

# Logic for page navigation
if "page" in st.session_state and st.session_state.page == "visualization":
    # To execute `visualization.py`, read its contents and execute them.
    with open("visualization.py") as f:
        exec(f.read())
