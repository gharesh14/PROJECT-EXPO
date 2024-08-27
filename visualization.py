import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle
import os
import altair as alt

# Define function to save model and feature names
def save_model_with_features(model, feature_names, file_path):
    with open(file_path, "wb") as f:
        pickle.dump((model, feature_names), f)

st.title("Model Comparison Results")

# Check if data is available
if "X_train" in st.session_state and "X_test" in st.session_state:
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    feature_names = X_train.columns.tolist()  # Get feature names from X_train

    # Separate models into classification and regression
    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
    }

    results = []

    # Evaluate classification models
    for model_name, model in classification_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({"Model": model_name, "Metric": "Accuracy", "Score": accuracy})

        # Save the model with feature names
        model_path = f"trained_model/{model_name}.pkl"
        if not os.path.exists("trained_model"):
            os.makedirs("trained_model")
        save_model_with_features(model, feature_names, model_path)

    # Evaluate regression models
    for model_name, model in regression_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": model_name, "Metric": "MSE", "Score": mse})
        results.append({"Model": model_name, "Metric": "R²", "Score": r2})

        # Save the model with feature names
        model_path = f"trained_model/{model_name}.pkl"
        if not os.path.exists("trained_model"):
            os.makedirs("trained_model")
        save_model_with_features(model, feature_names, model_path)

    # Display the results
    results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    st.write("Model Performance Comparison")

    # Add a text box with additional information
    st.markdown("""
    ### Model Performance Summary
    - **Classification Models**: These models are evaluated based on accuracy. The accuracy score represents the proportion of correctly predicted instances.
    - **Regression Models**: These models are evaluated using Mean Squared Error (MSE) and R² score. MSE measures the average squared difference between predicted and actual values, while R² score indicates the proportion of variance explained by the model.

    Below, you can see a bar chart comparing the performance of different models. The chart includes accuracy scores for classification models and MSE/R² for regression models.
    """)

    # Create Altair bar chart
    chart = alt.Chart(results_df).mark_bar().encode(
        x=alt.X('Model:N', title="Model"),
        y=alt.Y('Score:Q', title="Score"),
        color='Metric:N',
        tooltip=['Model', 'Metric', 'Score']
    ).properties(
        width=700,
        height=400
    )

    st.altair_chart(chart)

    # Allow user to input filename and download the model
    st.subheader("Download a Model")
    model_names = list(classification_models.keys()) + list(regression_models.keys())
    selected_model = st.selectbox("Select Model to Download", model_names)
    file_name = st.text_input("Enter filename for the .pkl file", value=f"{selected_model}.pkl")

    if st.button("Generate and Download Model"):
        model_path = f"trained_model/{selected_model}.pkl"
        if os.path.exists(model_path):
            # Read the model file
            with open(model_path, "rb") as f:
                model_data = f.read()

            # Create a downloadable link with the specified filename
            st.download_button(
                label="Download Model",
                data=model_data,
                file_name=file_name,
                mime="application/octet-stream"
            )
        else:
            st.error("The selected model file does not exist. Please ensure that the model has been trained and saved.")
else:
    st.write("Please preprocess the data on the 'ML' page before comparing models.")
