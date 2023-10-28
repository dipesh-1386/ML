# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:57:04 2023

@author: ds448
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
cal=fetch_california_housing()
dataset=pd.DataFrame(cal.data,columns=cal.feature_names)
dataset['Price']=cal.target
"""sns.heatmap(dataset.corr(),annot=True)"""

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1];
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=10)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train_sc,y_train)
from sklearn.metrics import r2_score
y_pred=regressor.predict(X_test_sc)
r2=r2_score(y_test, y_pred)
adr2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r2,adr2)