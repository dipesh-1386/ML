# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:23:16 2023

@author: ds448
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.DataFrame({
    'color':['red','blue','green','green','red','blue']
    })
encoder=LabelEncoder()
#Fit encoder to datafram and transform the categorical variable
encoder.fit_transform(df['color'])
from sklearn.preprocessing import OrdinalEncoder
df2=pd.DataFrame({
    'size':['small','medium','large','medium','small','large']
    })
encoder2=OrdinalEncoder(categories=[['small','medium','large']])
encoder2.fit_transform(df2[['size']])