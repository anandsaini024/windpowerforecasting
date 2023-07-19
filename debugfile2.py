import utils
import evaluation
import torch
import torch.nn as nn
import torch.optim as optim

data = utils.data_loader('C:/Users/shash/Desktop/Courses2022_23/Data Mining/Project/Project_data/wtbdata_245days.csv')
data[['Wspd','Prtv', 'Patv']].describe()

data = utils.handle_data_caveats(data)
data[['Wspd','Prtv', 'Patv']].describe()

data_turbine_1 = utils.single_turbine_plots(data)

power_z_score = utils.calculate_zscore(data_turbine_1)

columns_to_standardize= ['Wspd', 'Etmp', 'Itmp', 'Pab1', 'Pab2', 'Pab3', 'Prtv']
standard_data_turbine_1 = utils.standardize_columns(data_turbine_1, columns_to_standardize)

correlation = utils.calculate_correlation(standard_data_turbine_1)

# Baseline Model

from baseline_model import MovingAverage

MA= MovingAverage(window_size=3)

test_patv_values, test_moving_average = MA.calculate_moving_average(data, 200, 45)

rmse, mae = evaluation.calculate_average_scores(test_patv_values, test_moving_average)
print("RMSE Value of the Baseline Model: ", rmse)
print("MAE Value of the Baseline Model: ", mae)

# Linear Regresion
from linear_regression_model import LinearRegression
import numpy as np

LR_model = LinearRegression(learning_rate=0.01, num_iterations=1000)

train_x,train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(standard_data_turbine_1, 170,30,45)

X = np.array(train_x[['Wspd', 'Prtv']])
y = np.array(train_y)
val_x = np.array(val_x[['Wspd', 'Prtv']])
valy = np.array(val_y)
test_x = np.array(test_x[['Wspd', 'Prtv']])
test_y = np.array(test_y)

# Train the model
LR_model.fit(X, y)

# Make predictions on the validation set
val_pred_y_lr = LR_model.predict(val_x)

# Evaluate the model
LR_rmse, LR_mae = evaluation.calculate_average_scores (val_y, val_pred_y_lr)

print("RMSE Value of the LinearRegression Model: ", LR_rmse)
print("MAE Value of the LinearRegression: ", LR_mae)

# Perceptron

from perceptron import Perceptron

perceptron = Perceptron(num_inputs=2, learning_rate=0.01)
perceptron.train(X,y, num_epochs=100)
y_pred_val_perceptron = perceptron.predict(val_x)
# Evaluate the model
per_rmse, per_mae = evaluation.calculate_average_scores (val_y, y_pred_val_perceptron)
print("RMSE Value of the Perceptron: ", per_rmse)
print("MAE Value of the Perceptron: ", per_mae)


# Improvement attempt

# Lasso - l1

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

lasso = Lasso(alpha=10)
lasso.fit(X, y)
y_val_pred_lasso = lasso.predict(val_x)

# Evaluate the model
lasso_rmse, lasso_mae = evaluation.calculate_average_scores (val_y, y_val_pred_lasso)
print("RMSE Value of the Lasso: ", lasso_rmse)
print("MAE Value of the Lasso: ", lasso_mae)

# Ridge -l2

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ridge = Ridge(alpha=10)
ridge.fit(X, y)
y_val_pred_ridge = ridge.predict(val_x)

# Evaluate the model
ridge_rmse, ridge_mae = evaluation.calculate_average_scores (val_y, y_val_pred_ridge)
print("RMSE Value of the Ridge: ", ridge_rmse)
print("MAE Value of the Ridge: ", ridge_mae)


# # MLP
# from mlp import MLP

# # Set the hyperparameters
# input_size = 2
# hidden_size = 16
# output_size = 1
# learning_rate = 0.1
# num_epochs = 500

# # Create an instance of the MLP model
# model = MLP(input_size, hidden_size, output_size)

# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Convert the training data to tensors
# X_train = torch.tensor(X, dtype=torch.float32)
# y_train = torch.tensor(y, dtype=torch.float32)

# # Training loop
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Print training loss
#     if (epoch+1) % 100 == 0:
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# # Make predictions on the validation set
# X_val = torch.tensor(val_x, dtype=torch.float32)
# y_val = model(X_val)

# # Evaluate the model
# mlp_rmse, mlp_mae = evaluation.calculate_average_scores (val_y,)
# print("RMSE Value of the MLP: ", ridge_rmse)
# print("MAE Value of the MLP: ", ridge_mae)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = utils.data_loader('C:/Users/shash/Desktop/Courses2022_23/Data Mining/Project/Project_data/wtbdata_245days.csv')

# Initialize a dictionary to store the results for each turbine
RMSE = {}
MAE = {}
# Loop through each unique turbine ID
for turb_id in data['TurbID'].unique():
    # Filter the data for the current turbine ID
    turbine_data = data[data['TurbID'] == turb_id]
    turbine_data = utils.handle_data_caveats(turbine_data)
    train_x,train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(turbine_data, 170,30,45)

    # Split the data into features (X) and target (y)
    X = np.array(test_x[['Wspd', 'Prtv']])
    y= np.array(test_y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ridge = Ridge(alpha=0.1)
    
    ridge.fit(X_scaled, y)
    

    y_pred = ridge.predict(X_scaled)
    
    # Calculate the MSE loss
    rmse, mae = evaluation.calculate_average_scores (y, y_pred)
    
    # Store the results for the current turbine ID
    RMSE[turb_id] = rmse
    MAE[turb_id] = mae
# Calculate the average MSE across all turbines
average_rmse = sum(RMSE.values()) / len(RMSE)
average_mae =  sum(MAE.values()) / len(MAE)

# Print the RMSE for each turbine and the average RMSE
for turb_id, rmse in RMSE.items():
    print(f"Turbine ID: {turb_id}, RMSE: {rmse}")
    
print(f"Average MSE: {average_rmse}")

# Print the MAE for each turbine and the average MAE
for turb_id, mae in MAE.items():
    print(f"Turbine ID: {turb_id}, MAE: {mae}")
    
print(f"Average MSE: {average_rmse}")    
print(f"Average MaE: {average_mae}")