#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:18:10 2021

@author: ruian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()
#print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:66] + "\n...")
#print("Target names: {}".format(iris_dataset['target_names']))
#print("Shape of data: {}".format(iris_dataset['data'].shape))
#print("First ten rows of data:\n{}".format(iris_dataset['data'][:10]))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
#print("X_train shape: {}".format(X_train.shape))
#print("y_train shape: {}".format(y_train.shape))

# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
#grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap='rainbow')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
#print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
#print("Prediction: {}".format(prediction))
#print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".fformat(knn.score(X_test, y_test)))