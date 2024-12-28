import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the dataset
train_data = pd.read_csv("LoanPrediction_TrainSet_Encoded.csv")

# Skip the first two columns
data_to_analyze = train_data.iloc[:, 2:]

# Count the instances for each label
label_counts = data_to_analyze['LoanStatus'].value_counts()
print("Label counts before upsampling:", label_counts)

# Initialize the SMOTE sampler
smote = SMOTE(random_state=42)

# Resampling the dataset
X = data_to_analyze.drop('LoanStatus', axis=1)  # Features
y = data_to_analyze['LoanStatus']  # Target variable

X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and labels back into a single DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['LoanStatus'] = y_resampled

# Count the instances for each label after upsampling
new_label_counts = resampled_data['LoanStatus'].value_counts()
print("Label counts after upsampling:", new_label_counts)

# Save the resampled data to a new CSV file
resampled_data.to_csv("LoanPrediction_TrainSet_SMOTE_Upsampled.csv", index=False)
