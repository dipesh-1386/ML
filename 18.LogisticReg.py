# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:35:39 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
dataset=load_iris()
print(dataset)
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
##Here we just want to take only 2 classes for O/P features so we are dropping datapoints which lie in 3rd class 
df_copy=df[df['target']!=2]
X=df_copy.iloc[:,:-1]
y=df_copy.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
##Prediction
y_pred=classifier.predict(X_test)
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
para={'penalty':('l1','l2','elasticnet',None),'C':[1,10,20]}
gsc=GridSearchCV(classifier,param_grid=para,cv=5)
##Split of train and validation data
gsc.fit(X_train,y_train)
print(gsc.best_params_)
##Again fitting models with best parameters(Hyperparameter Tuning)
classifier=LogisticRegression(C=1,penalty='l2')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
para={'penalty':('l1','l2','elasticnet',None),'C':[1,10,20]}
rsc=RandomizedSearchCV(classifier,param_distributions=para,cv=5,n_iter=3)
##Split of train and validation data
rsc.fit(X_train,y_train)
print(rsc.best_params_)
##Again fitting models with best parameters(Hyperparameter Tuning)
classifier=LogisticRegression(C=1,penalty='l2')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)