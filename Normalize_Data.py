import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data_to_analyze = pd.read_csv("LoanPrediction_TrainSet_FeatureEngineered.csv")

# Specify the columns to normalize
features_to_normalize = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome', 'InstalmentAmount', 'BalancedIncome', 'Dependents', 'LoanAmountTerm', 'PropertyArea']

# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Apply the scaler to the data
data_to_analyze[features_to_normalize] = scaler.fit_transform(data_to_analyze[features_to_normalize])

# Save the scaled data to a new CSV file
data_to_analyze.to_csv('LoanPrediction_TrainSet_Scaled.csv', index=False)
