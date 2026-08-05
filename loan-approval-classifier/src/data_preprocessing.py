import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

loan_dataset = pd.read_csv('../data/loan_approval_dataset.csv')

print("First few rows of the dataset:")
print(loan_dataset.head())
print("______________________________________")
print("Number of missing values in each column:")
print(loan_dataset.isnull().sum())
print("______________________________________")
print("Dataset information:")
print(loan_dataset.info())
print("______________________________________")
print("Dataset description:")
print(loan_dataset.describe())
print("______________________________________")
print("Number of duplicate rows:")
print(loan_dataset.duplicated().sum())
print("______________________________________")
print("Column names:")
print(loan_dataset.columns)
print("______________________________________")
print("the shape of the dataset:")
print(loan_dataset.shape)
print("______________________________________")
#splitting the dataset into features and target variable
features = loan_dataset.iloc[:, :-1]
target=loan_dataset.iloc[:, -1]
print(features.head(5))
print(target.head(5))
print("______________________________________")
#the features are divided into numerical and categorical features
num_features = ['loan_id', ' no_of_dependents', ' loan_amount', ' cibil_score',' residential_assets_value', ' commercial_assets_value',
       ' luxury_assets_value', ' bank_asset_value']
cat_features = [' education',' self_employed']
print("Numerical features:", num_features)
print("Categorical features:", cat_features)
print("______________________________________")
#turning categorical features into numerical features using one hot encoding and scaling the numerical features
num_transformer=StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("________________________________________")
#splitting the dataset into training and testing sets
X=  features
y= loan_dataset[' loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
