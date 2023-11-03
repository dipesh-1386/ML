# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:05:52 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=1000, n_features=10,n_informative=5,n_redundant=5,n_classes=3,random_state=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(multi_class='multinomial',solver='lbfgs')
classifier.fit(X_train,y_train)
##Prediction
y_pred=classifier.predict(X_test)
##Prediction Probability
y_pred_p=classifier.predict_proba(X_test)
##Confusion MAtrix,accuracy score,classification report
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
##Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
para={'penalty':('l1','l2','elasticnet',None),'C':[1,2,3,5,8,10]}
gsc=GridSearchCV(classifier,param_grid=para,cv=5)
##Split of train and validation data
gsc.fit(X_train,y_train)
print(gsc.best_params_)
##Again fitting models with best parameters(Hyperparameter Tuning)
classifier=LogisticRegression(multi_class='multinomial',solver='lbfgs',C=1,penalty='l2')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
