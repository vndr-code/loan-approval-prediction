# Loan Approval Prediction

## **Project Overview**
This project explores predictive modeling for automating loan approvals at a mock financial institution. Using historical loan application data, the study implements K-Nearest Neighbors (KNN) and Random Forest algorithms to predict loan approval outcomes. The analysis highlights the effectiveness of Random Forest, which outperformed KNN in predictive accuracy.

## **Purpose**
With increasing loan application volumes, the project simulates how financial institutions can leverage machine learning to automate decision-making, improve efficiency, and reduce manual intervention.

---

## **Predictive Modeling Project**
This repository contains scripts for data preprocessing, analysis, and predictive modeling using Python. The steps include generating an Analytics Base Table (ABT), handling missing values, feature engineering, and applying machine learning algorithms such as KNN, Decision Trees, and Random Forests.

---

## **System Requirements**
- **Operating System:** Windows, Linux, or macOS
- **Python Version:** Python 3.8 or higher

---

## **Installation Guide**

### **1. Install Python and Required Libraries**
1. Install Python from the official website: [https://python.org](https://python.org). 
2. Open a terminal or command prompt and install the required libraries by running:
   ```bash
   pip install pandas matplotlib scikit-learn numpy imbalanced-learn

## **Running the Code**

### **General Instructions**
1. Navigate to the directory containing the project files.
2. Open your preferred Python development environment (e.g., Visual Studio Code, PyCharm, or a terminal).

### **Scripts Overview**
Below are the steps and corresponding scripts to execute:

1. **Analytics Base Table (ABT) Generation**  
   - Open and run `ABT Generation.py` to prepare the ABT and perform initial data exploration.

2. **Correlation Analysis**  
   - Run `Correlation Analysis.py` to calculate and display variable correlations.

3. **Binning Analysis**  
   - Execute `Binning.py` to analyze variable distributions and create bins.

4. **Handling Missing Values**  
   - Use `Missing Values.py` to replace missing values with the mean (numeric) or mode (categorical).

5. **Convert to Numerical**  
   - Run `Make Numerical.py` to encode categorical variables into numerical values.

6. **Upsample Minority Class**  
   - Apply `upsampled SMOTE.py` to handle class imbalance using the SMOTE technique.

7. **Feature Engineering**  
   - Run `Features.py` to create new features for analysis.

8. **Normalize Data**  
   - Use `Normalize Data.py` to normalize data for KNN analysis.

9. **Choose K for KNN**  
   - Run `Choose k.py` to determine the optimal number of neighbors.

10. **KNN Analysis**  
    - Apply `KNN Analysis.py` to perform KNN-based predictions.

11. **Decision Tree Analysis**  
    - Use `DecisionTree Analysis.py` to generate and evaluate a decision tree model.

12. **Random Forest Analysis**  
    - Run `Forest Analysis.py` to generate and evaluate a random forest model.

---

## **Reproducing Results**
To reproduce the results, execute the scripts in the order listed above. Follow the steps for both the training and test datasets.

---

## **Troubleshooting**
If issues arise:
- Ensure Python is updated and all required libraries are installed.
- Check that dataset file paths in the scripts are correctly specified.
- Consult the documentation for the respective libraries if errors persist.

---

## **Notes**
This project demonstrates key data preprocessing and machine learning workflows, which can be adapted for other datasets or applications.

