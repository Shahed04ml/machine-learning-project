from data_preprocessing import *
from train_models import *
from evaluation import *
sns.countplot(data=loan_dataset, x="education", color="pink")
plt.title("education distribution")
plt.show()

sns.countplot(data=loan_dataset, x="self_employed", color="purple")
plt.title("self_employed distribution")
plt.show()

sns.countplot(data=loan_dataset, x="loan_status", color="red")
plt.title("loan_status distribution")
plt.show()

cm = confusion_matrix(y_test, y_pred_RF)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Approved','Rejected'],
            yticklabels=['Approved','Rejected'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("random forest Confusion Matrix")
plt.show()

cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Approved','Rejected'],
            yticklabels=['Approved','Rejected'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()