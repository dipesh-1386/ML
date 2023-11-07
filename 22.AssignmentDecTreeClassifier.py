# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:53:23 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target']=dataset.target
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier,filled=True)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
##Post Prunning
classifier2=DecisionTreeClassifier(max_depth=3)
classifier2.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier2,filled=True)
y_pred2=classifier2.predict(X_test)
con=confusion_matrix(y_test,y_pred2)
acc=accuracy_score(y_test, y_pred2)
cla=classification_report(y_test, y_pred2)
print(con)
print(acc)
print(cla)