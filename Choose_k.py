import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv("LoanPrediction_TrainSet_Scaled.csv")

# Separate features and target
X = data.drop('LoanStatus', axis=1)
y = data['LoanStatus']

# Normalize the data 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define a range of k to test
k_range = range(1, 26)

# Empty list to store scores
k_scores = []

# Perform cross-validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')  # 10-fold cross-validation
    k_scores.append(scores.mean())

# Sorting scores and keeping track of k
k_scores_with_k = list(zip(k_scores, k_range))
k_scores_with_k.sort(reverse=True, key=lambda x: x[0])

# Print the best three k values and their scores
print("Top 5 k values with their Cross-Validated Accuracies:")
for score, k in k_scores_with_k[:5]:
    print(f"k = {k} with Accuracy: {score}")

# Optionally, plot the results
import matplotlib.pyplot as plt
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
