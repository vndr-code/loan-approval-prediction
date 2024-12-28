import pandas as pd

# Load the dataset
train_data = pd.read_csv("LoanPrediction_TrainSet_SMOTE_Upsampled.csv")

# Calculate TotalIncome by adding ApplicantIncome and CoapplicantIncome
train_data['TotalIncome'] = train_data['ApplicantIncome'] + train_data['CoapplicantIncome']

# Calculate InstalmentAmount from LoanAmount and LoanAmountTerm
# Assuming LoanAmountTerm is in months and LoanAmount in thousands
train_data['InstalmentAmount'] = (train_data['LoanAmount'] * 1000) / train_data['LoanAmountTerm']

# Calculate BalancedIncome by subtracting InstalmentAmount from TotalIncome
train_data['BalancedIncome'] = train_data['TotalIncome'] - train_data['InstalmentAmount']

# Save the modified DataFrame to a new CSV file
train_data.to_csv("LoanPrediction_TrainSet_FeatureEngineered.csv", index=False)

# Display a snippet of the modified DataFrame to verify the new columns
print(train_data.head())
