# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:38:12 2019

@author: Josue
"""

import os #module for the coding the directory
import pandas as pd #module for data frames

os.getcwd() 
data = pd.read_pickle('wind_pickle.pickle') #we read the file wind.pickle
data.loc[0:10] #Sample of the data set

#Data preprocessing 

#'steps', 'month', 'day','hour' are not useful, so we drop them 
data.drop(columns=['steps', 'month', 'day','hour'])
data.loc[0:10]

#The remaining variables are the input attributes. They are defined for the 25 
#locations in the grid. But we are going to use only those variables for 
#location 13 (Sotavento). Therefore, use Pandas for selecting the variables
#that end in .13.
new_columns=list(data.columns) #list with all the columns
sotavento=[] # empty list
counter=0
for column in new_columns: #go to all the name colums 
    if(column[-2:]!='13' and column!='energy' and column!='year'): # if it does not finish in 13 appends it to sotavento
        sotavento.append(column)
sotavento
for col in sotavento:
    data=data.drop(col,1) #drops all the columns who does not finish in 13
data.loc[0:10] #Sample of the data set  


#divide the data into a training set (for years 2005-08) and a test set 
#(for years 2009-2010)
train= data.loc[(data.year <= 2008)] #Store in train the part of data whose year lower than 209 ( 2005-2008)
test=data.loc[(data.year > 2008)] #Store in train the part of data whose year greater than 2008 ( 2009-2010)

#Removing attribute year
train=train.drop('year',1) #drop column year in train 
test=test.drop('year',1) #drop column year in test

#In order to apply the machine learning algorithms we must turn the train and 
#test into matrices. Dataframes are not allowed
import numpy as np
train=train.values
test=test.values

#hyper-parameter tuning with Random Search
from sklearn import tree #For decesion trees
from sklearn import metrics #For computing mean absolute error
from sklearn.model_selection import RandomizedSearchCV #Module in order to apply the random search

X_train=train[:,1:] #Takes all columns of train except the first
y_train=train[:,0] #Takes just the first column (response)
X_test=test[:,1:] #Takes all columns of test except the first
y_test=test[:,0] #Takes just the first column (response)

# Here, we define the type of training method (nothing happens yet)
clf = tree.DecisionTreeRegressor() #As our response is continuous we apply DecisionTreeRegressor not classifier
param_grid={'max_depth': range(2,20,2), #hyperparametres for the algorithm 
           'min_samples_split': range(2,300,10)}
rCV=RandomizedSearchCV(clf,param_grid,cv=5,n_iter=100) #We define the model for the hypermeters
rCV.fit(X_train,y_train) #we fit it to  the train

y_test_pred=rCV.predict(X_test) #We predict respect to the X_test in order to know if there is a considerable difference 
print(metrics.mean_absolute_error(y_test_pred, y_test)) #comparison the prediction with the test and prediction
                                                          #(mean absolute error)
                                                          
from sklearn import svm #For support vector machines

#model radial
param_SVM = {'kernel': ['rbf'], #Definition of hyperparameters
             'C':[9,27,81,243,729,2187,19683,59049,177147,531441,1594323,4782969], 
             'gamma': [4,8,16,32,64,128,256,512,1024,2048,4096,8192]}

clf = svm.SVR() #support vector machines regressor 
svm_radial = RandomizedSearchCV(clf,param_distributions=param_SVM,cv = 10,n_iter = 10) #model 

svm_radial.fit(X_train, y_train) #we fit it to the train part

y_test_pred = svm_radial.predict(X_test) #Now we predict the fit respect to the test
print(metrics.mean_absolute_error(y_test_pred, y_test))   #comparison the prediction with the test and prediction
                                                          #(mean absolute error)
                                                          

# model linear
param_SVM = {'kernel': ['linear'],
             'C':[9,27,81,243,729,2187]}

clf = svm.SVR()
svm_radial = RandomizedSearchCV(clf,param_distributions=param_SVM,cv = 10,n_iter = 5)

svm_radial.fit(X_train, y_train)

svm_radial.fit(X_train, y_train)
y_test_pred = svm_radial.predict(X_test)

print(metrics.mean_absolute_error(y_test_pred, y_test))