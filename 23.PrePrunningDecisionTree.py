# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 21:52:12 2023

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
##Accuracy Without using Prunning
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
con=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test, y_pred)
cla=classification_report(y_test, y_pred)
print(con)
print(acc)
print(cla)
##PrePrunning using Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
paramet = {
   'criterion':["gini", "entropy", "log_loss"],
   'splitter':["best", "random"],
   'max_depth':[1,2,3,4,5],
   'max_features':["auto", "sqrt", "log2"]
         }
gsv=GridSearchCV(treeclassifier, param_grid=paramet,cv=5,scoring='accuracy')
gsv.fit(X_train,y_train)
print(gsv.best_params_)
classifier2=DecisionTreeClassifier(criterion= 'entropy', max_depth= 4,max_features='log2', splitter='best')
classifier2.fit(X_train,y_train)
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier2,filled=True)
y_pred2=classifier2.predict(X_test)
##Accuracy with PrePrunning
con=confusion_matrix(y_test,y_pred2)
acc=accuracy_score(y_test, y_pred2)
cla=classification_report(y_test, y_pred2)
print(con)
print(acc)
print(cla)
