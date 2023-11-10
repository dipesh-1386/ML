# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:19:46 2023

@author: ds448
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=2,n_redundant=0)
sns.scatterplot(x=pd.DataFrame(X)[0],y=pd.DataFrame(X)[1],hue=y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.25)
from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
print(svc.coef_)
y_pred=svc.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
param={
       'C':[0.1,1,10,100,1000],
       'gamma':[1,0.1,0.01,0.001,0.0001]
       }
from sklearn.model_selection import GridSearchCV
gsv=GridSearchCV(svc, param_grid=param,cv=5,refit=True,verbose=3)
gsv.fit(X_train,y_train)
print(gsv.best_params_)
y_pred4=gsv.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred4)
acc=accuracy_score(y_test, y_pred4)
cla=classification_report(y_test, y_pred4)
print(con)
print(acc)
print(cla)