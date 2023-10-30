# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:00:25 2023

@author: ds448
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Algerian.csv')
##To get null values
df[df.isnull().any(axis=1)]
##There are 2 regions in dataset so we divide it according to region as 0 and 1
df.loc[:122,"Region"]=0
df.loc[122:,"Region"]=1
df.info()
##Coverting region from float to int
df[["Region"]]=df[["Region"]].astype(int)
##Get count of null values in each column
df.isnull().sum()
##Dropping rows with null values
df=df.dropna().reset_index(drop=True)
df.isnull().sum()
##Dropping 122th index as its value contains repeated column names
df=df.drop(122).reset_index(drop=True)
## REmoving blank spaces from column name
df.columns=df.columns.str.strip()
##converting type
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']]=df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)
df[['Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']]=df[['Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float)
df.info()
df.to_csv('Algerian_ff_cleaned.csv',index=False)
df_copy=df.drop(['day','month','year'],axis=1)
df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)
df_copy['Classes']=df_copy['Classes'].astype(int)
df_copy['Classes'].value_counts()
##Plotting histogram of every featue
plt.style.use('seaborn')
df_copy.hist(bins=50,figsize=(20,15))
##Plotting pie chart
percentage=df_copy['Classes'].value_counts(normalize=True)*100
classlabels=["Fire","Not fire"]
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabels,autopct='%1.1f%%')
plt.show()
X=df_copy.drop('FWI',axis=1)
y=df_copy['FWI']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)
#Feature Selection
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features=correlation(X_train,0.85)
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)
#Feature scailing
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title('X_train Before Scaling')
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title('X_train After Scaling')
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train_scaled,y_train)
y_pred=linreg.predict(X_test_scaled)
from sklearn.metrics import r2_score, mean_absolute_error
r2=r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Linear reg=> ","R2 score",r2,"Mean absolute error",mae)
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred=lasso.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Lasso Mean absolute error", mae)
print("Lasso R2 Score", score)
plt.scatter(y_test,y_pred)
from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred=ridge.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Ridge Mean absolute error", mae)
print("Ridge R2 Score", score)
plt.scatter(y_test,y_pred)
from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred=elastic.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("ElasticNet Mean absolute error", mae)
print("ElasticNet R2 Score", score)
plt.scatter(y_test,y_pred)

##Pickling
import pickle
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(ridge,open('ridge.pkl','wb'))
pickle.dump(linreg,open('linearregression.pkl','wb'))
