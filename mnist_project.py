import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

class MNISTModels:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}

    def training(self, name, model):
        try:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"{name} Accuracy: {acc} - mnist_project.py:38")
            print("Confusion Matrix:\n - mnist_project.py:39", cm)
            print("Classification Report:\n - mnist_project.py:40", classification_report(self.y_test, y_pred))
            self.models[name] = model
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Init Error at line {ex_line.tb_lineno}: {ex_msg} - mnist_project.py:44")

    def model(self):
        self.training("KNN", KNeighborsClassifier(n_neighbors=5))
        self.training("Naive Bayes", GaussianNB())
        self.training("Logistic Regression", LogisticRegression())
        self.training("Decision Tree", DecisionTreeClassifier(criterion='entropy'))
        self.training("Random Forest", RandomForestClassifier(n_estimators=5, criterion='entropy'))
        self.training("AdaBoost", AdaBoostClassifier(estimator=LogisticRegression(), n_estimators=5))
        self.training("Gradient Boosting", GradientBoostingClassifier(n_estimators=5))
        self.training("XGBoost (Baseline)", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        self.training("SVM", SVC(kernel='rbf', probability=True))

    def xgb_grid(self):
        try:
            xgb_reg = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            param_grid = {
                'n_estimators': [ 100],
                'max_depth': [ 5],
                'learning_rate': [0.1]
            }
            grid = GridSearchCV(
                estimator=xgb_reg,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            grid.fit(self.X_train, self.y_train)
        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Init Error at line {ex_line.tb_lineno}: {ex_msg} - mnist_project.py:75")

    def plot_roc_curves(self):
        try:
            y_test_bin = label_binarize(self.y_test, classes=list(range(10)))
            plt.figure(figsize=(12, 10))

            for name, model in self.models.items():
                y_score = model.predict_proba(self.X_test)
                fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves for All Models on MNIST")
            plt.legend(loc=0)
            plt.show()


        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            print(f"Init Error at line {ex_line.tb_lineno}: {ex_msg} - mnist_project.py:98")


if __name__ == "__main__":
    df = pd.read_csv('D:\mnist_project\mnist_test.csv.zip')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    obj = MNISTModels(X_train, y_train, X_test, y_test)
    obj.training
    obj.model()
    obj.xgb_grid()
    obj.plot_roc_curves()