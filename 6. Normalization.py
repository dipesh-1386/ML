# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:04:03 2023

@author: ds448
"""


import numpy as np
import seaborn as sns
import pandas as pd
df=sns.load_dataset('taxis')
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
min_max.fit_transform(df[['distance','fare','tip']]) 
#For new data
min_max.transform([[1,3,4]])
