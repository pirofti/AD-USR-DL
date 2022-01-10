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

import copy
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
import sys
import warnings
warnings.filterwarnings('ignore')

from dictlearn import DictionaryLearning

from ksvd_supp import ksvd_supp
from utils import gen_synth_fixed_supp
from utils import test_fpfn, test_fpfn_ocsvm, test_fpfn_lof, test_fpfn_iforrest


def iterate_lambdas():
    params['supp_lambda'] = supp_lambda_factor
    dl = DictionaryLearning(
        n_components=n_components,
        max_iter=n_iterations,
        fit_algorithm=ksvd_supp,
        n_nonzero_coefs=n_nonzero_coefs,
        dict_init=D0,
        params=params,
        data_sklearn_compat=False
    )
    dl.fit(Y)
    dictionary = dl.D_
    code = dl.X_

    used_atoms = np.unique(np.nonzero(code)[0])
    used_set = set(used_atoms)
    n_test_nonzero_coefs = len(used_atoms)
    zerod_atoms_percent = n_test_nonzero_coefs/n_components
    print(f'{supp_lambda_factor}: supp is at {zerod_atoms_percent}')

    # TESTING: ANOMALY DETECTION
    sensitivity, specificity = test_fpfn(X, y_truth, dictionary, used_set,
                                         n_nonzero_coefs, params)
    print(f'DL-SUPP: sensitivity {sensitivity}, specificity {specificity}')

    sensitivity, specificity = test_fpfn_ocsvm(X, y_truth)
    print(f'OCSVM: sensitivity {sensitivity}, specificity {specificity}')

    sensitivity, specificity = test_fpfn_lof(X, y_truth)
    print(f'LOF: sensitivity {sensitivity}, specificity {specificity}')

    sensitivity, specificity = test_fpfn_iforrest(X, y_truth)
    print(f'IForrest: sensitivity {sensitivity}, specificity {specificity}')


# LEARNING
if len(sys.argv) == 1:
    n_components_in = 8
    n_components_out = 4
else:
    n_components_in = int(sys.argv[1])
    n_components_out = int(sys.argv[2])

for n_samples_anomaly_percent in np.arange(0.01, 0.2, 0.02):
    max_overlap = int(np.minimum(n_components_in, n_components_out) / 2)
    for n_overlap in np.arange(0, max_overlap, 1):
        # data parameters
        n_features = 64                     # signal dimension (m)
        n_components = 2 * n_features       # number of atoms (n)
        n_iterations = 20                   # number of DL iterations (K)
        n_nonzero_coefs = int(np.round(0.2 * np.sqrt(n_features)))  # sparsity (s)

        n_samples = 25 * n_components        # number of signals (N)
        n_samples_train_percent = 0.9       # percentage of training signals
        n_samples_train = int(np.round(n_samples_train_percent * n_samples))
        n_samples_test = n_samples - n_samples_train
        n_samples_anomaly = int(np.round(n_samples_test * n_samples_anomaly_percent))

        # learning parameters
        params = {
            'replatoms': 2,    # NO
            'supp_reg': 0,     # 0: L0, 1: L1, 2: clipped
            'supp_lambda': 0.02,
            'supp_thres': 0,    # will be updated later
        }
        supp_thres_factor = 4

        # DATA
        # init dictionary
        D0 = normalize(np.random.randn(n_features, n_components))

        Inliers, Outliers, _ = gen_synth_fixed_supp(n_features,
                                                    n_components_in,
                                                    n_components_out,
                                                    n_overlap,
                                                    n_nonzero_coefs,
                                                    n_samples,
                                                    n_samples_anomaly)
        Y = Inliers.T[:, :n_samples_train]
        Y_normal_test = Inliers.T[:, n_samples_train:]
        Y_anomaly_test = Outliers.T
        Y_norm = np.linalg.norm(Y, ord='fro')
        print(f'signals norm {Y_norm}')

        X_fit = Y.T
        X = np.c_[Y_normal_test, Y_anomaly_test].T
        y_truth = np.r_[np.ones(n_samples_test), -1*np.ones(n_samples_anomaly)]

        params['supp_thres'] = supp_thres_factor * np.mean(
            np.linalg.norm(Y, axis=0))

        for supp_lambda_factor in np.arange(50, 80, 5):
            iterate_lambdas()
