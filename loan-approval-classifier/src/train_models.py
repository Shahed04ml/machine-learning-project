from data_preprocessing import *
import joblib
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
joblib.dump(pipeline_rf, "../model_rf.pkl")

#logistic regression pipeline
pipeline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        random_state=42
    ))
])

#fit the pipelines
pipeline_lr.fit(X_train, y_train)
joblib.dump(pipeline_lr, "../model_lr.pkl")