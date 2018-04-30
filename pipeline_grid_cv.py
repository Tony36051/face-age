from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import os
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


def load_data():
    data_dir = "d:/data"
    hog_npy = os.path.join(data_dir, "hog.npy")
    lbp_npy = os.path.join(data_dir, "lbp.npy")
    vgg_npy = os.path.join(data_dir, "vgg-fea.npy")

    hog_data = np.load(hog_npy)
    y = hog_data[:, -1]
    hog_data = hog_data[:, :-1]

    lbp_data = np.load(lbp_npy)[:, :-1]
    vgg_data = np.load(vgg_npy)  # [:, :-1]

    return hog_data, lbp_data, vgg_data, np.array([y]).T


def only(train_data, feature_name, regr_name="SVR"):
    X_train, X_test, y_train, y_test = train_test_split(train_data[:, 0:-1], train_data[:, -1],
                                                        test_size=0.1, random_state=2018, shuffle=True)
    model = None
    models = {
        "LogisticRegression": LogisticRegression(),
        "LinearSVR": LinearSVR(),
        "SVR": SVR(),  # rbf kernel
        "AdaBoostRegressor": AdaBoostRegressor(random_state=2018),
        "BaggingRegressor": BaggingRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor(),
        "RandomForestRegressor": RandomForestRegressor(n_jobs=-1)
    }
    regr_name = "LinearSVR"
    model = models[regr_name]
    if model is None:
        return None

    model.fit(X_train, y_train)
    pre = model.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, pre)
    print("%s on %s(%d) resulting mae: %f" % (regr_name, feature_name, X_train.shape[1], mae))
    return mae


def pca(X, n_components=512, k=None):
    pca = PCA(n_components=n_components, whiten=True)
    if k:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        skb = SelectKBest(f_regression, k=k)
        skb2 = SelectKBest(mutual_info_regression, k=k)
        combined_features = FeatureUnion([("pca", pca), ("lda", lda), ("skb", skb), ("skb2", skb2)])
        return combined_features.fit_transform(X, y.ravel())
    else:
        return pca.fit_transform(X)


def union_pca_skb(train_data, n_components=30, k=30):
    X = train_data[:, 0:-1]
    y = train_data[:, -1]
    X = pca(X, n_components, k)
    return np.hstack((X, np.array([y]).T))


if __name__ == '__main__':
    hog_data, lbp_data, vgg_data, y = load_data()
    hog_data = pca(hog_data)  # pca to 512
    lbp_data = pca(lbp_data)  # pca to 512
    train_data = np.hstack((hog_data, lbp_data, vgg_data, y))
    X_train, X_test, y_train, y_test = train_test_split(train_data[:, 0:-1], train_data[:, -1],
                                                        test_size=0.1, random_state=2018, shuffle=True)
    print(X_train.shape, y_train.shape)

    combined_features = FeatureUnion([
        ("skb", SelectKBest()),
    ])
    pipeline = Pipeline([
        ('features', combined_features),
        ("StandardScaler", StandardScaler()),
        ('LinearSVR', LinearSVR())
    ])


    param_grid = {
        'features__skb__k': [10, 100, 300],
        'features__skb__score_func': [f_regression,mutual_info_regression],
        'LinearSVR__C': [0.1, 1, 10],
    }
    clf = GridSearchCV(
        pipeline, param_grid=param_grid, n_jobs=-1, verbose=1, cv=3, scoring='neg_mean_absolute_error'
    )
    cvscore = clf.fit(X_train, y_train)
    pre = clf.predict(X_test)
    # metrics
    mae = mean_absolute_error(y_test, pre)
    print("final mae:" + str(mae))

    score = "mae"
    print("# Tuning hyper-parameters for %s" % score)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (-mean, std * 2, params))
    print()
