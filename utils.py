# Copyright (c) 2021 Paul Irofti <paul@irofti.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, normalize

from dictlearn.methods._omp import omp


def gen_synth_fixed_supp(n_features, n_components_normal, n_components_outlier,
                         n_overlap, n_nonzero_coefs,
                         n_samples, n_samples_anomaly):
    # original dictionary
    D_inlier = normalize(np.random.randn(n_features, n_components_normal))
    D_outlier = normalize(np.random.randn(n_features, n_components_outlier))
    if n_overlap != 0:
        D_outlier = np.c_[D_inlier[:, :n_overlap], D_outlier[:, n_overlap:]]

    # Synthetic: fixed support with noise
    Inliers_orig = np.empty((n_features, n_samples))
    for i in range(n_samples):
        pos = np.random.permutation(n_components_normal)[:n_nonzero_coefs]
        coeff = np.random.normal(0, 1, n_nonzero_coefs)
        Inliers_orig[:, i] = np.sum(coeff * D_inlier[:, pos], axis=1)
    # Noise = np.random.normal(0, 0.1, n_samples)
    Inliers = (Inliers_orig).T

    Outliers = np.empty((n_samples_anomaly, n_features))
    for i in range(n_samples_anomaly):
        pos = np.random.permutation(n_components_outlier)[:n_nonzero_coefs]
        coeff = np.random.normal(0, 1, n_nonzero_coefs)
        Outliers[i] = np.sum(coeff * D_outlier[:, pos], axis=1)

    return Inliers, Outliers, np.c_[D_inlier, D_outlier]

def is_anomaly(X, used_set):
    y = np.empty(X.shape[0])
    for i, x in enumerate(X):
        cur_set = set(np.nonzero(x)[0])
        if not cur_set.issubset(used_set):
            y[i] = -1
        else:
            y[i] = 1
    return y

def fpfn(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def test_fpfn(X, y_true, dictionary, used_set, n_nonzero_coefs, params):
    Y, _ = omp(X.T, dictionary, n_nonzero_coefs)
    y_pred = is_anomaly(Y.T, used_set)
    return fpfn(y_pred, y_true)

def test_fpfn_ocsvm(X, y_true, gamma='scale', kernel='rbf'):
    clf = make_pipeline(StandardScaler(),
                        OneClassSVM(gamma=gamma,kernel=kernel))
    y_pred = clf.fit_predict(X)
    return fpfn(y_pred, y_true)

def test_fpfn_lof(X, y_true, n_neighbors=20,
                  leaf_size=30, metric='minkowski', p=2):
    clf = make_pipeline(StandardScaler(),
                        LocalOutlierFactor(n_neighbors=n_neighbors,
                                           leaf_size=leaf_size,
                                           metric=metric,
                                           p=p))
    y_pred = clf.fit_predict(X)
    return fpfn(y_pred, y_true)

def test_fpfn_iforrest(X, y_true, n_estimators=100,
                       bootstrap=False, max_features=1.0):
    clf = make_pipeline(StandardScaler(),
                        IsolationForest(random_state=0,
                                        n_estimators=n_estimators,
                                        bootstrap=bootstrap,
                                        max_features=max_features,
                                        warm_start=True))
    y_pred = clf.fit_predict(X)
    return fpfn(y_pred, y_true)
