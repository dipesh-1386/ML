# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:52:19 2023

@author: ds448
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
marks=[49,35,65,37,45,47,58,59,69,63,88,79,82,1000,1100]
minimum,Q1,Q2,Q3,maximum=np.quantile(marks,[0.0,0.25,0.5,0.75,1.0])
IQR=Q3-Q1
lower_fence=Q1-1.5*(IQR)
upper_fence=Q3+1.5*(IQR)
print(upper_fence,lower_fence)
for i in marks:
    if i<lower_fence or i>upper_fence:
        print(i)
        
sns.boxplot(marks)