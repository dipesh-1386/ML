# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:05:25 2023

@author: ds448
"""

import numpy as np
import matplotlib.pyplot as plt
# 1. Interpolate using linear interpolation
xlp=np.array([1,2,3,4,5])
ylp=np.array([2,4,6,8,10])
x_new=np.linspace(1,5,10)
y_interp=np.interp(x_new,xlp,ylp)
#plt.scatter(x_new,y_interp)
# 2. Interpolate using cubic interpolation
xcp=np.array([1,2,3,4,5])
ycp=np.array([1,8,27,64,125])
from scipy.interpolate import interp1d
#create cubic interpolation function
f=interp1d(xcp, ycp,kind='cubic')
x_newcp=np.linspace(1,5,10)
y_intercp=f(x_newcp)
plt.scatter(x_newcp,y_intercp)
# 3. Interpolate using polynomial interpolation
xpp=np.array([1,2,3,4,5])
ypp=np.array([1,4,9,16,25])
#interpolate data using polynomial interpolation
p=np.polyfit(xpp,ypp,2)# 2 is degree of polynomial
x_newpp=np.linspace(1, 5,10)
y_interpp=np.polyval(p,x_newpp)
#plt.scatter(x_newpp,y_interpp)