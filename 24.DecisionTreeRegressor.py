# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:31:20 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
from sklearn.model_selection import GridSearchCV
param={'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
       'splitter':['best', 'random'],
       'max_features':['auto', 'sqrt', 'log2'],
       'max_depth':[1,2,3,4,5,6,7,10,12,15]
       }
gsv=GridSearchCV(regressor,param_grid=param,cv=2,scoring='neg_mean_squared_error')
gsv.fit(X_train,y_train)
print( gsv.best_params_ )
reg2=DecisionTreeRegressor(criterion='absolute_error', max_depth= 10, max_features='log2',splitter= 'best')
reg2.fit(X_train,y_train)
y_pred=reg2.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)