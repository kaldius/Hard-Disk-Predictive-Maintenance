import pandas as pd
import numpy as np
import sklearn
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("./Datasets/Train_dataset.csv")
data.drop(columns = ['date_x','serial_number','model','failure_x','date_actual_fail'], inplace=True)
x = data.drop(columns="failure_actual_fail")
y = data[['failure_actual_fail']]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=1)

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)
# undersample = RandomUnderSampler()
# x_train, y_train = undersample.fit_resample(x_train, y_train)

# Hyperparameters:
# n-estimators: [10, 2000] increments of 10 or 100?
# learning rate [0.1, 2.0], increments of 0.1
# Base estimator: logistic regression?
# best 1.4 lr, 1000 estimators
# recall: Best: 0.657700 using {'learning_rate': 1.5, 'n_estimators': 1100}

model = AdaBoostClassifier()
grid = dict()
grid['n_estimators'] = [1100, 1500, 2000]
grid['learning_rate'] = [1.75, 2.0, 2.5]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
print("Starting grid search")
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=cv, scoring='recall')
grid_result = grid_search.fit(x_test, np.ravel(y_test))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# n_estimators = 100
# clf = AdaBoostClassifier(n_estimators=n_estimators)
# scores = cross_val_score(clf, x_train, y_train, cv=5)
# scores.mean()
# clf.fit(x_train, y_train)

