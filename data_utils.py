import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def load_data():
    """
    Generate or load the dataset(Climate).
    In real world you will load your csv data or read the api data.
    """
    
    # Generate a random dataset for demonstration purposes
    dates = pd.date_range(start = "2010-01-01", end = "2023-12-31", freq="ME")
    
    # Generate the synthetic data
    
    temps = []
    for i in range(len(dates)):
        # Based my temperature with seasonal patterns
        
        seasonal = 15+10 * np.sin(2 * np.pi * i / 12)
        
        # Add and uppward trend 
        
        trend = 0.03 * 1
        
        # Add some noise
        
        noise = np.random.normal(0, 1.5)
        
        temps.append(seasonal + trend + noise)
        
    # Create a DataFrame
    
    df = pd.DataFrame({
        "date": dates,
        "temperatures": temps
    })
        
    # Extract the features 
    
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    
    return df


# Function to prepare features
def prepare_features(df):
    
    """
    prepare the features for the model.
    """
    X = df[["year", "month", ]].values
    y = df["temperatures"].values
    return X, y