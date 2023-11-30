# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:51:43 2023

@author: ds448
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
X,y=make_regression(n_samples=1000,n_features=2,n_targets=1,noise=3)
sns.scatterplot(x=pd.DataFrame(X)[0],y=pd.DataFrame(X)[1],hue=y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.25)
from sklearn.svm import SVR
svr=SVR(kernel='linear')
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
param={
       'C':[0.1,1,10,100,1000],
       'gamma':[1,0.1,0.01,0.001,0.0001],
       'epsilon':[0.1,0.2,0.3]
       }
from sklearn.model_selection import GridSearchCV
gsv=GridSearchCV(svr, param_grid=param,cv=5,refit=True,verbose=3)
gsv.fit(X_train,y_train)
print(gsv.best_params_)
y_pred4=gsv.predict(X_test)
score=r2_score(y_test,y_pred)
print(score)