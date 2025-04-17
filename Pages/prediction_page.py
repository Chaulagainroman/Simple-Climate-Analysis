import streamlit as st
import pandas as pd
import numpy as np
from models import load_model
from visualizations import plot_prediction_context


def show(df):
    """
    Display the Prediction page for the climate analysis app.
    
    Args:
        df (pd.DataFrame): The dataset containing historical climate data.
    """
    st.header("Temperature Prediction")
    
    
    # Load the trained model
    model = load_model(filename="climate_model.pkl")
    
    if model is None:
        st.error("No trained model found. Please go to the Model Training page to train and save a model.")
        return

    # Create input widgets for year and month
    col1, col2 = st.columns(2)
    with col1:
        pred_year = st.number_input(
            "Select Year",
            min_value=int(df["year"].min()),
            max_value=2100,
            value=2024,
            step=1
        )
    with col2:
        pred_month = st.selectbox(
            "Select Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.to_datetime(f"2023-{x:02d}-01").strftime("%B")
        )

    # Button to trigger prediction
    if st.button("Predict Temperature"):
        # Prepare the input features
        input_data = np.array([[pred_year, pred_month]])
        
        # Make the prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Display the prediction
            st.success(f"Predicted Temperature for {pd.to_datetime(f'{pred_year}-{pred_month:02d}-01').strftime('%B %Y')}: {prediction:.2f}°C")
            
            # Prepare historical data for the same month
            hist_data = df[df["month"] == pred_month][["year", "temperatures"]]
            hist_temps = list(zip(hist_data["year"], hist_data["temperatures"]))
            
            # Plot the prediction in context
            fig = plot_prediction_context(hist_temps, pred_year, pred_month, prediction)
            st.pyplot(fig)
            
        except Exception as e:
            # st.error(f"Error making prediction: {str(e)}")
            pass

    # Optional: Display a note about the model
    st.markdown(
        """
        **Note**: Ensure a model has been trained and saved on the Model Training page.
        The prediction uses the most recently saved model (`climate_model.pkl`).
        """
    )