#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 08:42:49 2018

@author: titan
"""

#import plotly
import csv
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix



#Reading and sptiting data
dataSet = pd.read_csv('/home/titan/Desktop/features.csv')
X = np.array(dataSet.drop(['Class'],1))
y = np.array(dataSet['Class'])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,y,test_size=0.2)
dataSet.head()

#Will print out text for later use
test = "Test Score : "
train = "Training score : "
cross = "Cross validation score : "
min_cross ="Min Cross validation : "
max_cross = "Max Cross validation : "
sd_cross = "Standard Deviation Cross validation : "
mean_cross ="Mean Cross validation : "


#KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=5 , leaf_size= 30)
knn_pred =knn.fit(X_train, Y_train).predict(X_test)
print( confusion_matrix(Y_test,knn_pred))
knn_test_accurracy = knn.score(X_test, Y_test)
print(test , knn_test_accurracy)
knn_train_accurracy = knn.score(X_train, Y_train)
print(train, knn_train_accurracy)
#Cross validation
knn_cross_scr = cross_val_score(knn,X_train, Y_train ,  cv=10)
print(cross,knn_cross_scr)
print(min_cross,min(knn_cross_scr))
print(max_cross,max(knn_cross_scr))
print(sd_cross,np.std(knn_cross_scr))
print(mean_cross,np.mean(knn_cross_scr))
knn_min_cross=min(knn_cross_scr)
knn_max_cross=max(knn_cross_scr)
knn_sd_cross = np.std(knn_cross_scr)
knn_mean_cross = np.mean(knn_cross_scr)





#SVM
SVM = svm.SVC()
SVM.fit(X_train, Y_train)
svm_pred = SVM.predict(X_test)
print( confusion_matrix(Y_test,svm_pred))
svm_test_accurracy = SVM.score(X_test, Y_test)
print(test , svm_test_accurracy)
svm_train_accurracy = SVM.score(X_test, Y_test)
print(train , svm_train_accurracy)
#Cross validation
svm_cross_scr = cross_val_score(SVM,X_train, Y_train ,  cv=10)
print(cross , svm_cross_scr)
print(min_cross,min(svm_cross_scr))
print(max_cross,max(svm_cross_scr))
print(sd_cross,np.std(svm_cross_scr))
print(mean_cross,np.mean(svm_cross_scr))
svm_min_cross=min(svm_cross_scr)
svm_max_cross=max(svm_cross_scr)
svm_sd_cross = np.std(svm_cross_scr)
svm_mean_cross = np.mean(svm_cross_scr)



#Naive Bayes (Gaussian )
G = GaussianNB()
G.fit(X_train, Y_train)
y_pred = G.predict(X_test)
print( confusion_matrix(Y_test,y_pred))
G_test_accurracy = G.score(X_test, Y_test)
print(test,G_test_accurracy)
G_train_accurracy = G.score(X_train, Y_train)
print(train,G_train_accurracy)
#Cross validation
G_cross_scr = cross_val_score(G,X_train, Y_train ,  cv=5)
print(cross,G_cross_scr)
print(min_cross,min(G_cross_scr))
print(max_cross,max(G_cross_scr))
print(sd_cross,np.std(G_cross_scr))
print(mean_cross,np.mean(G_cross_scr))
G_min_cross=min(G_cross_scr)
G_max_cross=max(G_cross_scr)
G_sd_cross = np.std(G_cross_scr)
G_mean_cross = np.mean(G_cross_scr)





#Naive Bayes(Bernoulli)
bnb = BernoulliNB()
bnb.fit(X_train, Y_train)
y_pred = bnb.predict(X_test)
print( confusion_matrix(Y_test,y_pred))
bnb_test_accurracy = bnb.score(X_test, Y_test)
print(test,bnb_test_accurracy)
bnb_train_accurracy = bnb.score(X_train, Y_train)
print(train,bnb_train_accurracy)
#Cross validation
bnb_cross_scr = cross_val_score(bnb,X_train, Y_train ,  cv=5)
print(cross,bnb_cross_scr)
print(min_cross,min(bnb_cross_scr))
print(max_cross,max(bnb_cross_scr))
print(sd_cross,np.std(bnb_cross_scr))
print(mean_cross,np.mean(bnb_cross_scr))
bnb_min_cross=min(bnb_cross_scr)
bnb_max_cross=max(bnb_cross_scr)
bnb_sd_cross = np.std(bnb_cross_scr)
bnb_mean_cross = np.mean(bnb_cross_scr)
