import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle # for saving the model



def split_data(X, y , test_size = 0.2):
    """
    Split the data into training and testing sets.
    """
     
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_model(X_train, y_train, model_type = "LinerarRegression"):
    """
    Train the model based on the selected model type.
    """
    
    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "RandomForestRegressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'LinearRegression' or 'RandomForestRegressor'.")
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(X_train, y_train,X_test, y_test, model):
    """
    Evaluate the model and return the performance metrics.
    """
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate the performance metrics
    
    metrics ={
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "test_r2": r2_score(y_test, y_pred_test),
        "y_test": y_test,
        "y_pred_test": y_pred_test
        
    }
    
    return metrics


def save_model(model, filename="climate_model.pkl"):
    """
    Save the trained model to a file.
    """
        
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        
def load_model(filename="climate_model.pkl"):
    """
    Load the trained model from a file.
    """
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Model file '{filename}' not found.")
        return None
    
