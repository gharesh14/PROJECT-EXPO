import streamlit as st
from streamlit_option_menu import option_menu

# Set the page configuration
st.set_page_config(
    page_title="No Code ML",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for the selected option
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = "Home"  # Default option

# Define the options with FontAwesome icons
options = ["Home", "ML", "Visualization", "Real-Time Analysis", "About"]
icons = ["house", "gear", "robot", "search", "info-circle"]  # FontAwesome icons

# Place the option menu in the sidebar, making it vertical
with st.sidebar:
    selected = option_menu(
        menu_title="No Code ML",
        options=options,
        icons=icons,  # Set FontAwesome icons for each option
        menu_icon="cast",  # Icon for the menu itself
        default_index=options.index(st.session_state.selected_option),
        orientation="vertical",  # Set orientation to vertical
        key="main_option_menu"  # Add a unique key
    )

# Update session state based on menu selection
st.session_state.selected_option = selected

# Link the selected option to the respective Python file
if st.session_state.selected_option == "Home":
    with open("home.py") as f:
        exec(f.read())
elif st.session_state.selected_option == "Real-Time Analysis":
    with open("prediction.py") as f:
        exec(f.read())
elif st.session_state.selected_option == "ML":
    with open("main1.py") as f:
        exec(f.read())
elif st.session_state.selected_option == "Visualization":
    with open("visualization.py") as f:
        exec(f.read())
elif st.session_state.selected_option == "About":
    with open("About.py") as f:
        exec(f.read())
