from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('/content/drive/MyDrive/creditcard.csv')

dataset.head().append(dataset.tail())

dataset.info()

print("Fraudulent Cases: " + str(len(dataset[dataset["Class"] == 1])))
print("Valid Transactions: " + str(len(dataset[dataset["Class"] == 0])))

data_p = dataset.copy()
data_p[" "] = np.where(data_p["Class"] == 1 ,  "Fraud", "Genuine")

data_p[" "].value_counts().plot(kind="pie", figsize=(11.7, 8.27))

print("Non-missing values: " + str(dataset.isnull().shape[0]))
print("Missing values: " + str(dataset.shape[0] - dataset.isnull().shape[0]))

df = dataset

df[['Amount', 'Time']].describe()

fig = px.box(df[['Time']])
fig.show()

scaler = RobustScaler().fit(dataset[["Time", "Amount"]])
dataset[["Time", "Amount"]] = scaler.transform(dataset[["Time", "Amount"]])

dataset.head().append(dataset.tail())

y = dataset["Class"]
X = dataset.iloc[:,0:30]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

kf = StratifiedKFold(n_splits=3, random_state = None, shuffle = False)

res_table = pd.DataFrame(columns=['classifiers', 'best classifier', 'fpr', 'tpr', 'auc', 'precision', 'recall', 'accuracy'])

def get_model_best_estimator_and_metrics(estimator, params, kf=kf, X_train=X_train,
                                         y_train=y_train, X_test=X_test,
                                         y_test=y_test, is_grid_search=True,
                                         sampling=NearMiss(), scoring="f1",
                                         n_jobs=2):

    print(f'started {estimator}')
    if sampling is None:
        pipeline = make_pipeline(estimator)
    else:
        pipeline = make_pipeline(sampling, estimator)

    estimator_name = estimator.__class__.__name__.lower()
    new_params = {f'{estimator_name}__{key}': params[key] for key in params}

    if is_grid_search:
        search = GridSearchCV(pipeline, param_grid=new_params, cv=kf, return_train_score=True, n_jobs=n_jobs, verbose=2)
    else:
        search = RandomizedSearchCV(pipeline, param_distributions=new_params,
                                    cv=kf, scoring=scoring, return_train_score=True,
                                    n_jobs=n_jobs, verbose=1)

    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.named_steps[estimator_name].predict(X_test)

    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'finished {estimator}')

    cm = confusion_matrix(y_test, y_pred)

    cm = pd.DataFrame(cm, index=['Not Fraud', 'Fraud'], columns=['Not Fraud', 'Fraud'])

    sns.heatmap(cm, linecolor='black', linewidth=1, annot=True, fmt='' , xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    cv_score = cross_val_predict(search, X_test, y_test, cv=kf)

    print(classification_report(y_test, y_pred, target_names = ['Not Fraud', 'Fraud'], digits=3))

    # y_proba = search.best_estimator_.named_steps[estimator_name].predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, cv_score)
    auc = roc_auc_score(y_test, cv_score)

    return {
        "y_pred": y_pred,
        "best_estimator": search.best_estimator_,
        "estimator_name": estimator_name,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

classifiers = {
    'RandomForestClassifier': {
        'estimator': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 10, 12],
            'random_state': [13]
        },
    },
    'SVC': {
        'estimator': SVC(),
        'params': {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    },
    'DecisionTreeClassifier': {
        'estimator': DecisionTreeClassifier(),
        'params': {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
    },
}

result_smote = []

for key, value in classifiers.items():
  res_class = get_model_best_estimator_and_metrics(
      estimator=value['estimator'],
      params=value['params'],
      sampling=SMOTE(),
      scoring="f1",
      is_grid_search=True,
      n_jobs=-1,
  );

  result_smote.append({
                        'classifiers': res_class["estimator_name"],
                        'best classifier': res_class['best_estimator'],
                        'precision': res_class['precision'],
                        'recall': res_class['recall'],
                        'accuracy': res_class['accuracy'],
                        'fpr': res_class["fpr"],
                        'tpr': res_class["tpr"],
                        'auc': res_class["auc"]
                      }, ignore_index=True)

result_original = []

for key, value in classifiers.items():
  res_class = get_model_best_estimator_and_metrics(
      estimator=value['estimator'],
      params=value['params'],
      sampling=None,
      scoring="f1",
      is_grid_search=False,
      n_jobs=-1,
  );

  result_original.append({
                        'classifiers': res_class["estimator_name"],
                        'best classifier': res_class['best_estimator'],
                        'precision': res_class['precision'],
                        'recall': res_class['recall'],
                        'accuracy': res_class['accuracy'],
                        'fpr': res_class["fpr"],
                        'tpr': res_class["tpr"],
                        'auc': res_class["auc"]
                      }, ignore_index=True)
