# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:08:50 2023

@author: ds448
"""

import numpy as np
import pandas as pd
#Creating dataset with 1000 values where 900 are 0s and 100 are 1s(Imbalance)
np.random.seed(123)
n_samples=1000
class_0_ratio=0.9
n_class_0=int(n_samples*class_0_ratio)
n_class_1=n_samples-n_class_0
class_0=pd.DataFrame({
    'feature_1':np.random.normal(loc=0,scale=1,size=n_class_0),
    'feature_2':np.random.normal(loc=0,scale=1,size=n_class_0),
    'target':[0]*n_class_0
    })
class_1=pd.DataFrame({
    'feature_1':np.random.normal(loc=0,scale=1,size=n_class_1),
    'feature_2':np.random.normal(loc=0,scale=1,size=n_class_1),
    'target':[1]*n_class_1
    })
df=pd.concat([class_0,class_1]).reset_index(drop=True)
df_minority=df[df['target']==1]
df_majority=df[df['target']==0]
#Upsampling
from sklearn.utils import resample
#df_minority_upsample=resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
#df_minority_upsample.shape
#df_upsampled=pd.concat([df_majority,df_minority_upsample])
#df_upsampled['target'].value_counts()
df_majority_downsample=resample(df_majority,replace=False,n_samples=len(df_minority),random_state=42)
df_majority_downsample.shape
df_downsampled=pd.concat([df_majority_downsample,df_minority])
df_downsampled['target'].value_counts()
