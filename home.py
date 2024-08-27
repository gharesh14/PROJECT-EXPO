import streamlit as st

# Page Header and Introduction
st.header("Welcome to Your No-Code Machine Learning Platform")
st.title("Harness the Power of Machine Learning with Ease")
st.write("""
Machine learning has never been more accessible. Whether you’re a beginner or an experienced data scientist, our platform enables you to build, compare, and deploy powerful machine learning models without writing a single line of code. Transform your data into actionable insights and predictions with just a few clicks.
""")

st.title("Empower Your Data Journey")
st.header("From Data to Decisions in a Few Simple Steps")
st.write("""
Our platform is designed to make machine learning effortless. Upload your dataset, preprocess your data, build and compare models, and make real-time predictions—all within an intuitive interface. No prior experience is needed; we guide you through every step of the process.
""")

# How It Works Section
st.header("How It Works:")
st.write("""
1. **Upload Your Dataset:** Start by uploading your data in CSV or Excel format. Our platform supports various data types and structures.
2. **Preprocess Your Data:** Select your target variable and scaling method. We automatically handle missing values and prepare your data for modeling.
3. **Compare Models:** Our platform trains multiple machine learning models, evaluates their performance, and provides visual comparisons. Choose the best model based on accuracy, MSE, or R² scores.
4. **Make Predictions:** Use the best-performing model to make real-time predictions on new data. Simply input the features, and the platform generates predictions instantly.
5. **Download Your Model:** Save your trained model for future use, complete with feature names and configurations, ready for deployment.
""")

# Why Choose Us Section
st.markdown("### Why Choose Our Platform?")
st.markdown("""
- **No Coding Required:** Our user-friendly interface makes machine learning accessible to everyone, regardless of technical background.
- **Comprehensive Model Comparison:** Quickly identify the best model for your data with our automated training and evaluation process.
- **Real-Time Predictions:** Apply your trained models to new data and get instant predictions, perfect for business and research applications.
- **Flexible and Scalable:** Suitable for projects of any size, from small datasets to large-scale analyses.
- **Download and Deploy:** Save your models in a portable format, ready for immediate deployment in any environment.
""")

# Get Started Section
st.markdown("### Start Your Machine Learning Journey Today")
st.markdown("""
Whether you're looking to optimize business processes, explore new research avenues, or simply learn more about machine learning, our platform is here to help. Begin your journey towards smarter, data-driven decisions today.
""")

# Button to navigate to the main page
if st.button('Get Started'):
    st.query_params = {"page": "main1"}

