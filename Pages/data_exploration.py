import streamlit as st
from visualizations import plot_time_series, plot_seasonal_pattern, plot_yearly_trends, plot_actual_vs_predicted

def show(df): 
    """
    Function to display the data exploration page.
    """
    
    st.title("Data Exploration")
    
    # Display the dataset
    st.subheader("Raw Tempreature Data")
    st.dataframe(df.head(10))
    
    # Display the dataset statistics
    st.subheader("Statistical Summary")
    st.write(df['temperatures'].describe())
    
    # Plot the time series
    st.subheader("Temperature Over Time")
    fig = plot_time_series(df)
    st.pyplot(fig)
    
    # Plot the seasonal pattern
    st.subheader("Seasonal Temperature Patteren")
    fig = plot_seasonal_pattern(df)
    st.pyplot(fig)
    
    # Plot the yearly trends
    st.subheader("Yearly Average Temperature")
    fig = plot_yearly_trends(df)
    st.pyplot(fig)