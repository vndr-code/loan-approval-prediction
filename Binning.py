import pandas as pd
import matplotlib.pyplot as plt

# Load your data
train_data = pd.read_csv("LoanPrediction_TrainSet_Modified.csv")


# Create a new dataframe for plotting
plot_data = train_data.groupby(['Dependents', 'LoanStatus']).size().unstack(fill_value=0)

# Calculate the approval percentage for each dependent group
plot_data['Approval Rate'] = (plot_data[1] / (plot_data[1] + plot_data[0])) * 100

# Plotting
ax = plot_data[[0, 1]].plot(kind='bar', stacked=False, figsize=(10, 6))
plot_data['Approval Rate'].plot(secondary_y=True, color='black', marker='o', ax=ax)

ax.set_title('Loan Approval Status by Number of Dependents')
ax.set_xlabel('Number of Dependents')
ax.set_ylabel('Frequency')
ax.right_ax.set_ylabel('Approval Rate %')
plt.xticks(rotation=0)  # Rotates labels to make them readable
ax.legend([0, 1], labels=['Not Approved', 'Approved'], title='Loan Status')
ax.right_ax.legend(['Approval Rate'], loc='upper right')
plt.show()






