# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:45:49 2023

@author: ds448
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('googleplaystore.csv')
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
#observation 1 -  Dataset has missing value
df[~df['Reviews'].str.isnumeric()]
df_copy=df.copy()
df_copy=df_copy.drop(df_copy.index[10472])
df_copy[~df_copy['Reviews'].str.isnumeric()]
df_copy.shape
df_copy['Size']=df_copy['Size'].str.replace('M','000')
df_copy['Size']=df_copy['Size'].str.replace('k','')
df_copy['Size']=df_copy['Size'].replace('Varies with device',np.nan)
df_copy['Size']=df_copy['Size'].astype(float)
df_copy.info()
df_copy['Price']=df_copy['Price'].str.replace('$','')
df_copy['Installs']=df_copy['Installs'].str.replace(',','')
df_copy['Installs']=df_copy['Installs'].str.replace('+','')
df_copy['Installs']=df_copy['Installs'].astype(int)
df_copy['Price']=df_copy['Price'].astype(float)
df_copy['Last Updated']=pd.to_datetime(df_copy['Last Updated'])
df_copy['Day']=df_copy['Last Updated'].dt.day
df_copy['Month']=df_copy['Last Updated'].dt.month
df_copy['Year']=df_copy['Last Updated'].dt.year
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
df_copy=pd.concat([df_copy,pd.DataFrame(encoder.fit_transform(df[['Content Rating']]).toarray(),columns=encoder.get_feature_names_out())],axis=1)
df_copy=df_copy.drop_duplicates(subset=['App'],keep='first')
#Exploring more data
#plt.figure(figsize=(15,16))
#sns.countplot(data=df_copy,x='Category')
#df_copy['Category'].value_counts().plot.pie(y=df_copy['Category'],figsize=(15,15),autopct='%1.1f')
#Observation - Family apps have more category
category=pd.DataFrame(df_copy['Category'].value_counts())
category.rename(columns={'Category':'Count'},inplace=True)
plt.figure(figsize=(85,6))
sns.histplot(data=df_copy,x='Installs',kde='True',hue='Category')

 
 
