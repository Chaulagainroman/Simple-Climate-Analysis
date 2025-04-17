import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_time_series(df):
    """
    Plot the temperature over time.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], df["temperatures"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature in  Celsius")
    ax.set_title("Temperature Over Time")
    ax.grid(True)
    
    return fig

def plot_seasonal_pattern(df):
    
    """
    Plot the monthly temperature distribution.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="month", y="temperatures", data=df, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature in Celsius")
    ax.set_title("Monthly Temperature Distribution")
    
    return fig

def plot_yearly_trends(df):
    
    """
    Plot the yearly average temperature.
    """
    
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_avg = df.groupby("year")["temperatures"].mean().reset_index()
    ax.plot(yearly_avg["year"], yearly_avg["temperatures"], marker = "o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Temperature in Celsius")
    ax.set_title("Yearly Average Temperature")
    ax.grid(True)
    
    return fig
    
def plot_actual_vs_predicted(y_test, y_pred):
        
    """
    Plot actual vs predicted values.
    """
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha = 0.7)
    # ax.plot(y_pred, label="Predicted", color="red")
    ax.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], "r--")
    ax.set_xlabel("Actual Temperature in Celsius")
    ax.set_ylabel("Predicted Temperature in Celsius")
    ax.set_title("Actual vs Predicted Temperatures")
    ax.legend()
    
    return fig
    
def plot_prediction_context(hist_temps, pred_year, pred_month, prediction):
    """
    Plot the historical temperatures and the prediction for a specific month.
    """
    
    year_hist, temp_hist = zip(*hist_temps)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot historical Data for the same month
    
    ax.scatter(year_hist, temp_hist, label = f"Historical Data for {pred_month}", color = "blue", alpha = 0.7)
    ax.plot(year_hist, temp_hist, 'b--', alpha = 0.6)
    
    # Plot the prediction
    
    ax.scatter([pred_year], [prediction], label = "Prediction", color = "red", s=100, marker = "o")
    
    # Add a trend line
    
    z = np.polyfit(year_hist, temp_hist, 1)
    p = np.poly1d(z)
    
    ax.plot(range(2010, pred_year + 1), p(range(2010, pred_year +1)), "g--", label = "Trend Line")
    
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Temperature for month {pred_month}")
    ax.title(f"Historical and Predicted Temperatures for the month of {pred_month}")
    ax.legend()
    ax.grid(True)
    
    return fig
    
    
    
    
    
    
    
    