from data_preprocessing import *

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#random forest pipeline
pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
#fit the pipelines
pipeline_rf.fit(X_train, y_train)

#logistic regression pipeline
pipeline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        random_state=42
    ))
])

#fit the pipelines
pipeline_lr.fit(X_train, y_train)