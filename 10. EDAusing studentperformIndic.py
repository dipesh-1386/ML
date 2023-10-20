# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 09:51:30 2023

@author: ds448
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('StudentsPerformance.csv')
df.head()
#1. To check missing values
df.isnull().sum()
#2. To check duplicates
df.duplicated().sum()
#3. To check data types
df.info()
#4. To check no. of unique value of each column
df.nunique()
#5. To check stats of dataset
df.describe()
#Segregate features
numerical_features=[feature for feature in df.columns if df[feature].dtype!='O']
categorical_features=[feature for feature in df.columns if df[feature].dtype!='O']
#Aggregate total and mean marks of astudent

df['total_score']=df['math score']+df['reading score']+df['writing score']
df['mean_score']=df['total_score']/3
df.head()
fig,axis=plt.subplots(1,3,figsize=(33,8))
plt.subplot(141)
sns.histplot(data=df,x='mean_score',bins=30,kde='True',color='g')
plt.subplot(142)
sns.histplot(data=df,x='mean_score',bins=30,kde='True',hue='gender')
plt.subplot(143)
sns.histplot(data=df,x='mean_score',bins=30,kde='True',hue='race/ethnicity')