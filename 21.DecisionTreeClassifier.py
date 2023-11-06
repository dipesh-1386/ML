# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:55:33 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
dataset=load_iris()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
X=df.iloc[:,:-1] 
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn.tree import DecisionTreeClassifier
treeclassifier=DecisionTreeClassifier()
treeclassifier.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier,filled=True)
y_pred=treeclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
##post prunning
treeclassifier=DecisionTreeClassifier(max_depth=2)
treeclassifier.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier,filled=True)
y_pred=treeclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
