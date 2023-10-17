# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:55:10 2023

@author: ds448
"""

import numpy as np
import seaborn as sns
import pandas as pd
df=sns.load_dataset('tips')
#From Scratch
total_bill=list(df['total_bill'])
mean=np.mean(total_bill)
sd=np.std(total_bill)
normalized_data=[]
for i in total_bill:
    ans=(i-mean)/sd
    normalized_data.append(ans)
#print(normalized_data)
sns.histplot(normalized_data)
from sklearn.preprocessing import StandardScaler
sscaler=StandardScaler()
sscaler.fit(df[['total_bill','tip']])
sscaler.transform(df[['total_bill','tip']])