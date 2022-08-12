# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:48:01 2022
Penguin Dataset classification model using Random Forest
@author: Soura
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("G:\\Python\\ML and Data Science\\UNZIP_FOR_NOTEBOOKS_FINAL\\DATA\\penguins_size.csv")
df= df.dropna()
X= pd.get_dummies(df.drop('species', axis= 1), drop_first=(True))
y= df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.ensemble import RandomForestClassifier 
 
rfc= RandomForestClassifier(n_estimators=10,max_features='auto',random_state=101)

rfc.fit(X_train,y_train)
preds= rfc.predict(X_test) 
preds 

from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

plot_confusion_matrix(rfc, X_test, y_test)
print(classification_report(y_test, preds))

rfc.feature_importances_





########## More rigorous model (Doing Grid search)





df= pd.read_csv('G:\\Python\\ML and Data Science\\UNZIP_FOR_NOTEBOOKS_FINAL\\DATA\\data_banknote_authentication.csv')
sns.pairplot(data=df, hue='Class')

X= df.drop("Class", axis =1)
y= df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

n_estimators= [64,100,128,200]
max_features= [2,3,4]
bootstrap= [True, False]
oob_score= [True, False]         ## It is not a substitute for accuracy, but only used in 
#case we are using bootstrap samples, so that we can utilize the unused rows for oob score 

param_grid={'n_estimators':n_estimators, 'max_features':max_features, 'bootstrap':bootstrap, 'oob_score':oob_score} 

# Base model
rfc= RandomForestClassifier()
grid= GridSearchCV(rfc, param_grid)

grid.fit(X_train, y_train)
grid.best_params_ 
 
### Building our model
rfc= RandomForestClassifier(n_estimators=200, max_features=2,bootstrap=True, oob_score=True)
rfc.fit(X_train, y_train)

rfc.oob_score_
predictions= rfc.predict(X_test)

from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score
plot_confusion_matrix(rfc, X_test, y_test)
print(classification_report(y_test, predictions))



errors=[]
misclassifications=[]
for n in range(1,200):
    rfc= RandomForestClassifier(n_estimators=n,max_features=2, bootstrap=True, oob_score=True)
    rfc.fit(X_train, y_train)
    preds= rfc.predict(X_test)
    n_missed= np.sum(preds!=y_test)
    
    err= 1-accuracy_score(y_test, preds)
    errors.append(err)
    misclassifications.append(n_missed)

plt.plot(range(1,200), errors)
plt.plot(range(1,200),misclassifications)    








 
 
 
 
 
 
 