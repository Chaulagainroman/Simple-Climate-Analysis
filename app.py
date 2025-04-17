import streamlit as st
from data_utils import load_data
import sys

sys.path.append(r"D:\PROJECTS\Climate Project\Pages")
from Pages import data_exploration, model_training, prediction_page



# Set the page Configuration

st.set_page_config(
    page_title="Climate Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Give the Title
st.title("Climate Trend Analysis and Prediction")
st.markdown(
    """
    This is a simple climate trend analysis and prediction app. 
    It uses a synthetic dataset to demonstrate the functionality.
    """
)
st.sidebar.header("Climate Analysis")

df = load_data()

# Give the sidebar for the app navigation

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ("Data Exploration", "Model Training", "Prediction"),key="navigation_radio"
)


# Display the Selected Page 

if page == "Data Exploration":
    data_exploration.show(df)
elif page == "Model Training":
    model_training.show(df)
elif page == "Prediction":
    prediction_page.show(df)
else:
    st.write("Please select a valid page from the sidebar.")
