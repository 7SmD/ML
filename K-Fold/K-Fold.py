import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))
print('\n')

#SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))
print('\n')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
print('\n')

#KFOLD
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
print(kf)
print('\n')

KFold(n_splits=3, random_state=None, shuffle=False)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

print(scores_logistic)
print('\n')

print(scores_svm)
print('\n')

print(scores_rf)
print('\n')

#cross_val_score function
from sklearn.model_selection import cross_val_score

#Logistic regression model performance using cross_val_score
print(cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3))
print('\n')

#svm model performance using cross_val_score
print(cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3))
print('\n')

#random forest performance using cross_val_score
print(cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3))
print('\n')

#Parameter tunning using k fold cross validation
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),digits.data, digits.target, cv=10)
print(np.average(scores1))
print('\n')

scores2 = cross_val_score(RandomForestClassifier(n_estimators=20),digits.data, digits.target, cv=10)
print(np.average(scores2))
print('\n')

scores3 = cross_val_score(RandomForestClassifier(n_estimators=30),digits.data, digits.target, cv=10)
print(np.average(scores3))
print('\n')

scores4 = cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target, cv=10)
print(np.average(scores4))
print('\n')
