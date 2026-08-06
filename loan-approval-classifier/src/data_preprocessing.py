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
def load_data():
   loan_dataset = pd.read_csv('../data/loan_approval_dataset.csv')
   loan_dataset.columns = loan_dataset.columns.str.strip()
   return loan_dataset
loan_dataset = load_data()
loan_dataset=loan_dataset.drop("loan_id", axis=1)

loan_dataset.insert(loc=6, column='loan_to_income', 
                     value=loan_dataset['loan_amount'] / loan_dataset['income_annum'])

loan_dataset=loan_dataset.drop(["loan_amount","income_annum"], axis=1)
#splitting the dataset into features and target variable
features = loan_dataset.iloc[:, :-1]
target=loan_dataset.iloc[:, -1]

#the features are divided into numerical and categorical features
num_features =  ["no_of_dependents","loan_term","cibil_score","residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value",'loan_to_income']
cat_features =['education','self_employed']
#turning categorical features into numerical features using one hot encoding and scaling the numerical features
num_transformer=StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])


#splitting the dataset into training and testing sets
X=  features
y= loan_dataset['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

