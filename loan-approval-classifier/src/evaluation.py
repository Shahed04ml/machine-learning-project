from train_models import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

y_pred_RF = pipeline_rf.predict(X_test)

print(accuracy_score(y_test, y_pred_RF))

print(classification_report(y_test, y_pred_RF))

y_pred_lr = pipeline_lr.predict(X_test)

print(accuracy_score(y_test, y_pred_lr))

print(classification_report(y_test, y_pred_lr))