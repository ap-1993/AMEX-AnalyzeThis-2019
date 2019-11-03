# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#loading packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from __future__ import absolute_import, division, print_function, unicode_literals

#loading the data
data = pd.read_csv('your path/development_dataset.csv')

#looking at the dataset
data.head()

#data pre-processing
data_1 = data.drop(columns = 'VAR1') #this is just serial no. of the data point
data_1.head()
df = data_1.fillna(data_1.median()) #fill the median values in the dataset to replace NaN(you can use your own metric like mean or mode or fill zeroes)
df_final = df.drop(columns = 'VAR14') #
df_final.head()

#encoding the target variable
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
df_final["VAR21"] = lb_make.fit_transform(df_final["VAR21"])
df_final[["VAR21"]].head(11)

#train test split
x = df_final.drop(columns = 'VAR21')
y = df_final['VAR21']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2) 
len(x_train)

#fitting the model
clf = MLPClassifier(solver='adam', alpha=1e-5, activation = 'tanh', momentum = 0.70, hidden_layer_sizes=(10,10,3), max_iter = 1000, learning_rate_init = 0.001)
clf.fit(x_train, y_train)

#model accuracy score
y_hat = clf.predict(x_test)

print("accuracy: " + str(accuracy_score(y_test, y_hat)))

#dimensionality reduction using PCA
pca_dims = PCA()
pca_dims.fit(x_train)
cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 #d is the no. of features that explain 95% of variance in the data

#transforming the data using PCA
pca = PCA(n_components=d)
X_reduced = pca.fit_transform(x_train)
X_recovered = pca.inverse_transform(X_reduced)

#fitting the model on the reduced dataset
clf_reduced = MLPClassifier(solver='sgd', alpha=1e-5, activation = 'tanh', momentum = 0.70, hidden_layer_sizes=(10,10,3), max_iter = 1000, learning_rate_init = 0.02)
clf_reduced.fit(X_reduced, y_train)

#accuracy of reduced model
X_test_reduced = pca.transform(x_test)

y_hat_reduced = clf_reduced.predict(X_test_reduced)

print("accuracy: " + str(accuracy_score(y_test, y_hat_reduced)))