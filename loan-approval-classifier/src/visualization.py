
from train_models import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#visulizion of the data
features[" education"].value_counts().plot(kind='barh')
plt.show()

features[" self_employed"].value_counts().plot(kind='barh')
plt.show()

sns.countplot(data=loan_dataset, x=" loan_status", color="red")
plt.title("loan_status distribution")
plt.show()

y_pred_RF = pipeline_rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred_RF)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest")
plt.show()


y_pred_log = pipeline_lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Logistic Regression")
plt.show()