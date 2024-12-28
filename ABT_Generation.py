import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv("LoanPrediction_TrainSet.csv") 


# Convert specific numerical columns to categorical
train_data['CreditHistory'] = train_data['CreditHistory'].astype('object')

# Skip the first two columns
data_to_analyze = train_data.iloc[:, 2:]

# Calculate basic statistics for numerical columns
numeric_cols = data_to_analyze.select_dtypes(include=[np.number])
numeric_stats = numeric_cols.describe().transpose()  
numeric_stats['Missing%'] = 100 * numeric_cols.isnull().sum() / len(numeric_cols)  

# Calculate statistics for categorical columns
categorical_cols = data_to_analyze.select_dtypes(exclude=[np.number])
categorical_describe = categorical_cols.describe(include='all').transpose()

# Add specific categorical statistics
categorical_stats = pd.DataFrame()
categorical_stats['Mode'] = categorical_describe['top']
categorical_stats['Mode Freq'] = categorical_describe['freq']
categorical_stats['Missing%'] = 100 * categorical_cols.isnull().sum() / len(categorical_cols)
categorical_stats['Cardinality'] = categorical_cols.nunique()

# Find the second mode
for col in categorical_cols:
    mode_counts = categorical_cols[col].value_counts()
    if len(mode_counts) > 1:
        second_mode = mode_counts.index[1]
        second_mode_freq = mode_counts.iloc[1]
    else:
        second_mode = np.nan
        second_mode_freq = np.nan
    categorical_stats.loc[col, '2nd Mode'] = second_mode
    categorical_stats.loc[col, '2nd Mode Freq'] = second_mode_freq


# Prepare to plot histograms in a single figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjust the size as needed
fig.suptitle('Histograms of Numeric Variables')

# List of numeric columns to plot
columns_to_plot = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm']
for ax, column in zip(axes.flatten(), columns_to_plot):
    ax.hist(train_data[column].dropna(), bins=20, color='blue', alpha=0.7)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.grid(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the subplots to provide space for the main title
plt.show()

# Print the statistics tables
print("Numerical Data Statistics:")
print(numeric_stats)

print("\nCategorical Data Statistics:")
print(categorical_stats)
