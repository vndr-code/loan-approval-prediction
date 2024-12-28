import pandas as pd
import numpy as np

# Load your dataset
train_data = pd.read_csv("LoanPrediction_TrainSet.csv")

# Convert specific numerical columns to categorical if needed
train_data['CreditHistory'] = train_data['CreditHistory'].astype('object')
train_data['LoanAmountTerm'] = train_data['LoanAmountTerm'].astype('object')

# Fill missing values for numeric columns with the mean
for column in train_data.select_dtypes(include=[np.number]).columns:
    train_data[column].fillna(train_data[column].mean(), inplace=True)

# Fill missing values for categorical columns with the mode
for column in train_data.select_dtypes(exclude=[np.number]).columns:
    train_data[column].fillna(train_data[column].mode()[0], inplace=True)

# Save the modified DataFrame to a new CSV file
train_data.to_csv("LoanPrediction_TrainSet_Filled.csv", index=False)  # Save the file without index column

print("Missing values filled and data saved to 'LoanPrediction_TrainSet_Filled.csv'.")



