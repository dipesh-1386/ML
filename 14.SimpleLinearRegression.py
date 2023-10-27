# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:00:14 2023

@author: ds448
"""
"""
1. Import dataset
2. Divide into independent and dependent feature(X,y)
3. Split into train and test data(train_test_split)
4. Feature Scailing of X_train, X_test
5.Train the model
6. Plot the data points of train data with predicted points(best fit line)
7. Find error(mse,mae,rmse)
8. Find accuracy of model
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('weight-height.csv')
X=df[['Weight']]
Y=df['Height']
plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression() 
reg.fit(X_train,y_train)
reg.intercept_
reg.coef_
plt.scatter(X_train,y_train)
plt.plot(X_train,reg.predict(X_train))
y_pred=reg.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
rms=np.sqrt(mse)
print(mse,mae,rms)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
adjusted_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r2,adjusted_r2)

