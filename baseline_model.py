import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size

    def calculate_moving_average(self, data, train_days, test_days):

        train_days = train_days
        test_days = test_days
        
        # get unique days in the dataset
        unique_days = data['Day'].unique()
        
        # Select the indices for train, and test sets
        train_indices = data[data['Day'].isin(unique_days[:train_days])].index
        test_indices = data[data['Day'].isin(unique_days[train_days:])].index

        # Split the data into train, and test sets based on indices
        train_data= data.loc[train_indices]
        test_data = data.loc[test_indices]
    
        test_data['MovingAverage'] = test_data['Patv'].rolling(window=self.window_size).mean()
        test_data['MovingAverage'] = test_data['MovingAverage'].fillna(test_data['Patv'].mean())
        
        # # Plot the test Patv values and the moving average
        # plt.figure(figsize=(10, 6))
        # plt.plot(test_data['Patv'], label='Patv')
        # plt.plot(test_data['MovingAverage'], label='Moving Average')
        # plt.xlabel('Index')
        # plt.ylabel('Patv')
        # plt.title('Test Patv Values and Moving Average')
        # plt.legend()
        # plt.show()
        
        return test_data['Patv'], test_data['MovingAverage']
