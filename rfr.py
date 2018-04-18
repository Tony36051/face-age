import pandas as pd
import os
import numpy as np
import argparse
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn
from sklearn.ensemble import RandomForestRegressor


data_dir = "/home/tony/data"
hog_file = "hog.txt"

train_data = pd.read_csv(os.path.join(data_dir, hog_file), header=None)
X_train, X_test, y_train, y_test = train_test_split(train_data.values[:, 0:-1], train_data.values[:, -1],
                                                    test_size=0.1, random_state=2018, shuffle=True)

# PCA
# pca = PCA(n_components=100, whiten=True)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)
# print(X_train.shape)
# print(X_test.shape)


# svc
regr = RandomForestRegressor(n_estimators=320, n_jobs=32, random_state=2018, verbose=1)
regr.fit(X_train, y_train)
pre = regr.predict(X_test)

# metrics
mae = sklearn.metrics.mean_absolute_error(y_test, pre)
print("mae: %f" % mae)