import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt, seaborn as sns
plt.rcParams['figure.figsize'] = (16, 9)

def data_loader(file_path):
    """Load the wind power data from a CSV file and  make a dataframe."""
    data = pd.read_csv(file_path)
    data = pd.DataFrame(data)
    return data


def handle_data_caveats(data):
    
    """Handles the caveat given in the research paper describing the
        data inputs"""
    
    # 1. Negative power values

    """There are some active power and reactive power which are smaller 
    than zeros. We simply treat all the values which are smaller than 0 
    as 0, i.e. if 𝑃𝑎𝑡𝑣 < 0, then 𝑃𝑎𝑡𝑣 = 0."""

    neg_prtv, neg_patv = (data['Prtv'] < 0).sum(), (data['Patv'] < 0).sum()

    data['Patv'] = np.where(data['Patv'] < 0, 0, data['Patv'])
    data['Prtv'] = np.where(data['Prtv'] < 0, 0, data['Prtv'])

    neg_prtv, neg_patv = (data['Prtv'] < 0).sum(), (data['Patv'] < 0).sum()

    # 2. Missing Values
    # Rows where all the column values are missing
    missing_patv_values =data['Patv'].isna().sum()
    data =  data.dropna(subset=['Patv'])
    missing_patv_values =data['Patv'].isna().sum()

    return data

def single_turbine_plots(data):
    data_turbine1 = data[data["TurbID"]==1]
    print("Number of records for Turbine-1: ", data_turbine1.shape[0])
    
    # Plot Active Power for turbine -1
    power_turbine1 = plt.plot(data_turbine1['Patv'])
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.title('Active Power (average) of Turbine')
    plt.show()

    average_power_per_day = data_turbine1.groupby('Day')['Patv'].mean()
    plot_avg_per_day = plt.plot(average_power_per_day, color ='red')
    plt.title('Average power per day - Turbine-1')
    plt.xlabel('Days')
    plt.ylabel('Power Average')
    plt.show()
    
    return data_turbine1

def calculate_zscore(data):
    # Calculate z-score
    z_score = (data['Patv'] - data['Patv'].mean()) / data['Patv'].std()

    # Visualize z-scores using a histogram
    plt.hist(z_score, bins=10, edgecolor='black')
    plt.xlabel('Z-Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Z-Scores for Patv')
    plt.show()
    
def standardize_columns(data, columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def calculate_correlation(data):

    correlation_matrix = data.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Features')
    plt.show()

def train_val_test_split(data, train_days, val_days, test_days):
	train_days = train_days
	validation_days = val_days
	test_days = test_days

	# get unique days in the dataset
	unique_days = data['Day'].unique()

	# Select the indices for train, validation, and test sets
	train_indices = data[data['Day'].isin(unique_days[:train_days])].index
	validation_indices = data[data['Day'].isin(unique_days[train_days:train_days+validation_days])].index
	test_indices = data[data['Day'].isin(unique_days[train_days+validation_days:])].index

	# Split the data into train, validation, and test sets based on indices
	turbine1_train = data.loc[train_indices]
	turbine1_validation = data.loc[validation_indices]
	turbine1_test = data.loc[test_indices]

	# Splitting the feature and target set
	train_x = turbine1_train[['Day', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']]
	train_y = turbine1_train['Patv']

	val_x = turbine1_validation[['Day', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']]
	val_y = turbine1_validation['Patv']

	# Splitting the feature and target set for test
	test_x = turbine1_test[['Day', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']]
	test_y = turbine1_test['Patv']

	return train_x, train_y, val_x, val_y, test_x, test_y