import numpy as np

def calculate_rmse(targets, predictions):
    """
    Calculate the Root Mean Square Error (RMSE) between targets and predictions.
    """
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    return rmse


def calculate_mae(targets, predictions):
    """
    Calculate the Mean Absolute Error (MAE) between targets and predictions.
    """
    mae = np.mean(np.abs(predictions - targets))
    return mae


def calculate_average_scores(targets, predictions):
    """
    Calculate the average RMSE and MAE scores.
    
    Args:
        targets (ndarray): Array of true target values.
        predictions (ndarray): Array of predicted values.
    
    Returns:
        tuple: Average RMSE and MAE scores.
    """
    rmse = calculate_rmse(targets, predictions)
    mae = calculate_mae(targets, predictions)
    
    return rmse, mae
