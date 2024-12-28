import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load the scaled training dataset
train_data = pd.read_csv("LoanPrediction_TrainSet_Scaled.csv")
test_data = pd.read_csv("LoanPrediction_TestSet_Scaled.csv")

# Separate features and target in training data
X_train = train_data.drop('LoanStatus', axis=1)
y_train = train_data['LoanStatus']

# Separate features and target in test data
X_test = test_data.drop('LoanStatus', axis=1)
y_test = test_data['LoanStatus']

# Create a KNN classifier with the optimal k found previously
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]  # probabilities needed for log loss and ROC AUC

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Classification Accuracy:", accuracy)
print("Logarithmic Loss:", logloss)
print("Confusion Matrix:\n", conf_matrix)
print("Area Under Curve:", roc_auc)
print("F1 Score:", f1)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
