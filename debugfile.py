import utils

data = utils.data_loader('C:/Users/shash/Desktop/Courses2022_23/Data Mining/Project/Project_data/wtbdata_245days.csv')
column = ", ".join(data.columns)
print('Columns Name:', column)
data = utils.handle_data_caveats(data)
# Preprocessing

power_z_score = utils.calculate_zscore(data)
columns_to_standardize= ['Wspd', 'Etmp', 'Itmp', 'Pab1', 'Pab2', 'Pab3', 'Prtv']
data = utils.standardize_columns(data, columns_to_standardize)

# feature Selection
correlation = utils.calculate_correlation(data)

## For One Turbine
data_turbine1 = utils.single_turbine_plots(data)

### Baseline Model

from baseline_model import MovingAverage

MA= MovingAverage(window_size=12)

test_patv_values, test_moving_average = MA.calculate_moving_average(data_turbine1, 200, 45)

import evaluation
rmse, mae = evaluation.calculate_average_scores(test_patv_values, test_moving_average)
print("RMSE Value of the Baseline Model: ", rmse)
print("MAE Value of the Baseline Model: ", mae)

### Linear Regression Model
from linear_regression_model import LinearRegression
# Create an instance of LinearRegression
LR_model = LinearRegression(learning_rate=0.01, num_iterations=1000)

# Split the data into train and test sets
train_x,train_y, val_x, val_y, test_x, test_y = LR_model.train_val_test_split(data_turbine1, 170,30,45)

# # Train the model
# LR_model.fit(train_x[['Wspd','Prtv']], train_y)

# # Make predictions on the test set
# val_pred_y = LR_model.predict(val_x[['Wspd','Prtv']])

# # Evaluate the model
# LR_rmse, LR_mae = evaluation.calculate_average_scores (val_y, val_pred_y)
# print("RMSE Value of the LR Model: ", LR_rmse)
# print("MAE Value of the LR Model: ", LR_mae)

# # l1 regularization

# LR_model_l1 = LinearRegression(learning_rate=0.01, num_iterations=1000,regularization='l1', lambda_value=0.01)

# # Split the data into train and test sets
# train_x,train_y, val_x, val_y, test_x, test_y = LR_model_l1.train_val_test_split(data_turbine1, 170,30,45)

# # Train the model
# LR_model_l1.fit(train_x[['Wspd','Prtv']], train_y)

# # Make predictions on the test set
# val_pred_y_l1 = LR_model_l1.predict(val_x[['Wspd','Prtv']])

# # Evaluate the model
# LR_rmse_l1, LR_mae_l1 = evaluation.calculate_average_scores (val_y, val_pred_y_l1)
# print("RMSE Value of the LR Model: ", LR_rmse_l1)
# print("MAE Value of the LR Model: ", LR_mae_l1)

# # l2 regularization

# LR_model_l2 = LinearRegression(learning_rate=0.01, num_iterations=1000,regularization='l2', lambda_value=0.1)

# # Split the data into train and test sets
# train_x,train_y, val_x, val_y, test_x, test_y = LR_model_l2.train_val_test_split(data_turbine1, 170,30,45)

# # Train the model
# LR_model_l2.fit(train_x[['Wspd','Prtv']], train_y)

# # Make predictions on the test set
# val_pred_y_l2 = LR_model_l2.predict(val_x[['Wspd','Prtv']])

# # Evaluate the model
# LR_rmse_l2, LR_mae_l2 = evaluation.calculate_average_scores (val_y, val_pred_y_l2)
# print("RMSE Value of the LR Model: ", LR_rmse_l2)
# print("MAE Value of the LR Model: ", LR_mae_l2)


# # Neural Network
# from neural_network import NeuralNetwork
# neural_net = NeuralNetwork(num_inputs=2, hidden_layers=[4], num_outputs=1)
# learning_rate = 0.01
# num_epochs = 1000
# neural_net.train(train_x[['Wspd','Prtv']], train_y, learning_rate, num_epochs)
# predictions = neural_net.predict(val_x[['Wspd','Prtv']])


## Perceptron

# from perceptron import Perceptron
# single_perceptron = Perceptron()
# single_perceptron.fit(train_x[['Wspd', 'Prtv']], train_y)
# prediction = single_perceptron.predict(val_x[['Wspd', 'Prtv']])
# # Evaluate the model
# per_rmse, per_mae = evaluation.calculate_average_scores (val_y, prediction)
# print("RMSE Value of the perceptron: ", per_rmse)
# print("MAE Value of the perceptron: ", per_mae)

####################################################
# ### MLP
# from mlp import MLP
# import numpy as np
# mlp = MLP(input_size=2, hidden_sizes=[16, 16], output_size=1, learning_rate=0.01)
# num_epochs=100
# mlp.train(train_x[['Wspd', 'Prtv']], train_y, num_epochs)
# val_y_pred_mlp = mlp.predict(np.array(val_x[['Wspd', 'Prtv']]))
######################################################

from perceptron2 import Perceptron2
import numpy as np
perc2 = Perceptron2(num_inputs=1, learning_rate=0.01)
perc2.train(np.array(train_x[['Wspd']]), np.array(train_y), num_epochs=100)
y_pred_val = perc2.predict(np.array(val_x[['Wspd']]))
# Evaluate the model
per2_rmse, per2_mae = evaluation.calculate_average_scores (val_y, y_pred_val)
print("RMSE Value of the perceptron2: ", per2_rmse)
print("MAE Value of the perceptron2: ", per2_mae)


######################################################################
# ## mlp2
# from mlp2 import MultiLayerPerceptron2
# mlp2 = MultiLayerPerceptron2(
#     input_size=2,
#     hidden_sizes=[16, 8],  # Example hidden layer sizes
#     output_size=1,         # Single output neuron for regression
#     learning_rate=0.01     # Example learning rate
# )

# # Train the MLP on your data
# mlp2.train(np.array(train_x[['Wspd', 'Prtv']]), np.array(train_y), num_epochs=1000)
# val_y_pred_mlp2 = mlp2.predict(np.array(val_x[['Wspd', 'Prtv']]))
######################################################################3
from mlp3 import train_mlp, predict_mlp

hidden_sizes = [16, 8]  # Example hidden layer sizes
learning_rate = 0.001
num_epochs = 2000
X = np.array(train_x[['Wspd', 'Prtv']])
y = np.array(train_y).reshape(-1,1)
weights, biases = train_mlp(X, y, hidden_sizes, learning_rate, num_epochs)
val_predictions = predict_mlp(np.array(val_x[['Wspd', 'Prtv']]), weights, biases)
import numpy as np
mlp2_rmse, mlp2_mae = evaluation.calculate_average_scores (np.array(val_y), val_predictions)
print("RMSE Value of the perceptron2: ", mlp2_rmse)
print("MAE Value of the perceptron2: ", mlp2_mae)