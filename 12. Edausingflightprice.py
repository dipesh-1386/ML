# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 19:45:32 2023

@author: ds448
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_excel('flight_price.xlsx')
df.head()
df['Date']=df['Date_of_Journey'].str.split('/').str[0].astype(int) 
df['Month']=df['Date_of_Journey'].str.split('/').str[1].astype(int)
df['Year']=df['Date_of_Journey'].str.split('/').str[2].astype(int)
df.drop('Date_of_Journey',axis=1,inplace=True)
df['Arrival_Hours']=df['Arrival_Time'].str.split(' ').str[0].str.split(':').str[0].astype(int)
df['Arrival_Minutes']=df['Arrival_Time'].str.split(' ').str[0].str.split(':').str[1].astype(int)
df.drop('Arrival_Time',axis=1,inplace=True)
df['Departure_Hours']=df['Dep_Time'].str.split(':').str[0].astype(int)
df['Departure_Minutes']=df['Dep_Time'].str.split(':').str[1].astype(int)
df.drop('Dep_Time',axis=1,inplace=True)
df.drop('Route',axis=1,inplace=True)
df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, np.nan:1, '4 stops':4})
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
df2=pd.DataFrame(encoder.fit_transform(df[['Airline','Source','Destination']]).toarray(),columns=encoder.get_feature_names_out())
maindf=pd.concat([df,df2],axis=1)
maindf.drop('Airline',axis=1,inplace=True)
maindf.drop('Source',axis=1,inplace=True)
maindf.drop('Destination',axis=1,inplace=True)
maindf[maindf['Duration']=='5m']
maindf.drop(6474,axis=0,inplace=True)
maindf['DurationH']=maindf['Duration'].str.split(' ').str[0].str.split('h').str[0].astype(int)
maindf['Durationm']=maindf['Duration'].str.split(' ').str[1].str.split('m').str[0].astype(float)
maindf['Durationm']=maindf['Durationm'].replace(np.nan,0)
maindf.drop('Duration',axis=1,inplace=True)
maindf['Duration']=(maindf['DurationH']*60+maindf['Durationm']).astype(int)
maindf.drop('DurationH',axis=1,inplace=True)
maindf.drop('Durationm',axis=1,inplace=True)   

        
