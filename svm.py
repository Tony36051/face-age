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


def load_data():
    data_dir = "/home/tony/data"
    hog_npy = os.path.join(data_dir, "hog.npy")
    lbp_npy = os.path.join(data_dir, "lbp.npy")
    vgg_npy = os.path.join(data_dir, "vgg-fea-tar.npy")

    hog_data = np.load(hog_npy)[:, :-1]
    lbp_data = np.load(lbp_npy)[:, :-1]
    vgg_data = np.load(vgg_npy)[:, :-1]
    y = np.load(vgg_npy)[:, -1]
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
    # hog_data = pca(hog_data)
    # lbp_data = pca(lbp_data)
    # # hog only
    # train_data = np.hstack((hog_data, y))
    # hog_only = only(train_data, "hog", "RandomForestRegressor")
    #
    # # lbp only
    # train_data = np.hstack((lbp_data, y))
    # lbp_only = only(train_data, "lbp", "RandomForestRegressor")
    #
    # # vgg only
    # train_data = np.hstack((vgg_data, y))
    # vgg_only = only(train_data, "vgg", "SVR")

    # hog-lbp
    train_data = np.hstack((hog_data, vgg_data, y))
    train_data = union_pca_skb(train_data)
    hog_lbp = only(train_data, "hog_lbp", "SVR")

    # hog-vgg
    train_data = np.hstack((hog_data, vgg_data, y))
    train_data = union_pca_skb(train_data)
    hog_vgg = only(train_data, "hog_vgg", "SVR")

    # lbp-vgg
    train_data = np.hstack((lbp_data, vgg_data, y))
    train_data = union_pca_skb(train_data)
    lbp_vgg = only(train_data, "lbp_vgg", "SVR")

    # hog-lbp-vgg
    train_data = np.hstack((hog_data, lbp_data, vgg_data, y))
    train_data = union_pca_skb(train_data)
    hog_lbp_vgg = only(train_data, "hog_lbp_vgg", "SVR")
