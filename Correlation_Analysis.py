import pandas as pd
import numpy as np

# Load the dataset
train_data = pd.read_csv("LoanPrediction_TrainSet.csv")

# Skip the first two columns
data_to_analyze = train_data.iloc[:, 2:]

# Encoding binary categorical variables with NaN preservation
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
loan_status_map = {'Y': 1, 'N': 0}

# Apply mapping with NaN values preserved
data_to_analyze['Gender'] = data_to_analyze['Gender'].map(gender_map)
data_to_analyze['Married'] = data_to_analyze['Married'].map(married_map)
data_to_analyze['Education'] = data_to_analyze['Education'].map(education_map)
data_to_analyze['SelfEmployed'] = data_to_analyze['SelfEmployed'].map(self_employed_map)
data_to_analyze['LoanStatus'] = data_to_analyze['LoanStatus'].map(loan_status_map)

# Encoding 'PropertyArea' with custom mapping preserving NaN
property_area_mapping = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
data_to_analyze['PropertyArea'] = data_to_analyze['PropertyArea'].map(property_area_mapping)

# Handling 'Dependents' (consider '3+' as 3)
data_to_analyze['Dependents'] = data_to_analyze['Dependents'].replace({'3+': 3}).astype(float)

# Calculate the correlation matrix
correlation_matrix = data_to_analyze.corr()

# Get the correlation with 'LoanStatus'
loan_status_correlation = correlation_matrix['LoanStatus'].sort_values(ascending=False)

# Print the top 5 correlated variables with 'LoanStatus', excluding 'LoanStatus' itself
print("Top 5 highly correlated variables with 'LoanStatus':")
print(loan_status_correlation[1:6])  # Skip the first entry since it will be 'LoanStatus' itself

# Save the modified DataFrame to a new CSV file
data_to_analyze.to_csv("LoanPrediction_TrainSet_Modified.csv", index=False)
