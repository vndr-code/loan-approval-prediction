import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss, confusion_matrix, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
train_data = pd.read_csv("LoanPrediction_TrainSet_FeatureEngineered.csv")
test_data = pd.read_csv("LoanPrediction_TestSet_FeatureEngineered.csv")

# Prepare features and target variable for training data
X_train = train_data.drop(['LoanStatus'], axis=1)
y_train = train_data['LoanStatus']

# Prepare features and target variable for test data
X_test = test_data.drop(['LoanStatus'], axis=1)
y_test = test_data['LoanStatus']

# Initialize the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators is the number of trees

# Train the model on the entire training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)[:, 1]  # Probability estimates for ROC AUC

# Evaluate the model using test data
print("Classification Report on Test Data:")
print(classification_report(y_test, predictions))
print("Accuracy on Test Data:", accuracy_score(y_test, predictions))
print("Logarithmic Loss:", log_loss(y_test, probabilities))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("ROC AUC Score:", roc_auc_score(y_test, probabilities))
print("F1 Score:", f1_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# Feature importance visualization
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))  # Increase figure size
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), np.array(X_train.columns)[indices], rotation=45, ha='right')  # Adjust rotation and alignment
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()  # Adjust layout
plt.show()


