import streamlit as st
from data_utils import  prepare_features
from models import split_data, train_model, evaluate_model, save_model, load_model
from visualizations import plot_actual_vs_predicted


def show(df):
    """
    Function to display the model training page.
    """
    
    st.title("Model Training")
    
    st.subheader("Train a Model to Predict Temperature")
    
    # Prepare the features and target variable
    
    X, y = prepare_features(df)
    
    # Split the data into training and testing sets
    
    tes_size = st.slider("Select the test size", 0.1, 0.4, 0.2)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=tes_size)
    
    st.write(f"Training set size: {len(X_train)} Samples")
    st.write(f"Testing set size: {len(X_test)} Samples")
    
    # Select the model type
    
    model_type = st.selectbox("Select the model type", ("LinearRegression", "RandomForestRegressor"))
    st.write(f"Selected model type: {model_type}")
    
    # Train the model
    
    if st.button("Train Model"):
        with st.spinner("Training in progress ..."):
            # Train the model
            model = train_model(X_train, y_train, model_type)
            st.success("Model trained successfully!")
            
            # Evaluate the model
            metrices = evaluate_model(X_train, y_train, X_test, y_test, model)
            st.write("Model Performance Metrics:")
            
            # Display the performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Train Performance")
                
                col1.metric(label="Train RMSE", value=f"{metrices['train_rmse']:.2f} °C")
                col1.metric(label="Train R2", value=f"{metrices['train_r2']:.4f}")
            with col2:
                st.subheader("Test Performance")
               
                col2.metric(label="Test RMSE", value=f"{metrices['test_rmse']:.2f} °C")
                col2.metric(label="Test R2", value=f"{metrices['test_r2']:.4f}")
           
            # Plot the actual vs predicted values
            
            st.subheader("Actual vs Predicted Values")
            fig = plot_actual_vs_predicted(metrices["y_test"], metrices["y_pred_test"])
            st.pyplot(fig)
            
            # Save the model
            save_model(model)
            
            st.success("Model saved successfully!")
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type 
            