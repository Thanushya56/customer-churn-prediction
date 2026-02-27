# Customer Churn Prediction

This project builds a machine learning model to predict customer churn using the IBM Telco Customer Churn dataset. It compares multiple classifiers (Logistic Regression, Decision Tree, Random Forest, SVM) and saves the best model for future use.

## Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Handling class imbalance using SMOTE
- Model training and evaluation
- Feature importance analysis
- Model persistence with joblib

## Requirements
Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib