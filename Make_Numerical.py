import pandas as pd
import numpy as np

# Load the dataset
train_data = pd.read_csv("LoanPrediction_TrainSet_Filled.csv")

# Encoding binary categorical variables with NaN preservation
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
loan_status_map = {'Y': 1, 'N': 0}

# Apply mapping with NaN values preserved
train_data['Gender'] = train_data['Gender'].map(gender_map)
train_data['Married'] = train_data['Married'].map(married_map)
train_data['Education'] = train_data['Education'].map(education_map)
train_data['SelfEmployed'] = train_data['SelfEmployed'].map(self_employed_map)
train_data['LoanStatus'] = train_data['LoanStatus'].map(loan_status_map)

# Encoding 'PropertyArea' with custom mapping preserving NaN
property_area_mapping = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
train_data['PropertyArea'] = train_data['PropertyArea'].map(property_area_mapping)

# Handling 'Dependents' (consider '3+' as 3)
train_data['Dependents'] = train_data['Dependents'].replace({'3+': 3}).astype(float)

# Save the modified DataFrame to a new CSV file
train_data.to_csv("LoanPrediction_TrainSet_Encoded.csv", index=False)  # Save the file without index column

print("Data encoded and saved to 'LoanPrediction_TrainSet_Encoded.csv'.")

