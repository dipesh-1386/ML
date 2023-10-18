# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:32:08 2023

@author: ds448
"""

import numpy as np
import seaborn as sns
import pandas as pd
df=sns.load_dataset('iris')
from sklearn.preprocessing import normalize
#To convert to data frame
pd.DataFrame(normalize(df[['sepal_length' , 'sepal_width',  'petal_length', 'petal_width']]),columns=['sepal_length' , 'sepal_width',  'petal_length', 'petal_width'])

