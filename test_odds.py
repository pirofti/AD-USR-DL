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
import pandas as pd
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from dictlearn import DictionaryLearning
from sklearn.preprocessing import normalize

from ksvd_supp import ksvd_supp
from utils import test_fpfn

# LEARNING
filename = sys.argv[1] # ex. 'satellite.csv' from odds.cs.stonybrook.edu

# data parameters
Y = pd.read_csv(filename, header=None)
n_features = np.shape(Y)[1]-1       # signal dimension (m)
n_components = 2 * n_features       # number of atoms (n)
n_iterations = 20                   # number of DL iterations (K)
n_nonzero_coefs = int(np.round(0.2 * np.sqrt(n_features)))  # sparsity (s)

y_truth = Y.iloc[:, -1:]
y_truth = 2 * y_truth - 1

Y_norm = np.linalg.norm(Y, ord='fro')

Inliers = Y.loc[Y[np.shape(Y)[1]-1] == 0]
Inliers = Inliers/Y_norm
Outliers = Y.loc[Y[np.shape(Y)[1]-1] == 1]
Outliers = Outliers/Y_norm

Y = Y.iloc[:, :-1]
Inliers = Inliers.iloc[:, :-1]
Outliers = Outliers.iloc[:, :-1]

n_samples = np.shape(Inliers)[0]        # number of signals (N)
n_samples_train_percent = 0.9       # percentage of training signals
n_samples_train = int(np.round(n_samples_train_percent * n_samples))
n_samples_test = n_samples - n_samples_train
n_samples_anomaly = np.shape(Outliers)[0]

# learning parameters
params = {
    'replatoms': 2,    # NO
    'supp_reg': 1,    # 0: L0, 1: L1, 2: clipped
    'supp_lambda': 0.02,
}
# Good supp_lambda values found via grid-search
# L0
#  satellite.csv : 0.009695128297758569
#  shuttle.csv   : 0.023854930000000826
#  pendigits.csv : 0.01102320000000024
#  speech.csv    : 0.00047483167617358473
#  mnist.csv     : 0.0013071141248941365
# L1
#  satellite.csv : 0.09378937464459605
#  shuttle.csv   : 0.12765840000000167
#  pendigits.csv : 0.12071539999999897
#  speech.csv    : 0.02088700000000009
#  mnist.csv     : 0.042566299438476105

# DATA
# init dictionary
D0 = normalize(np.random.randn(n_features, n_components))

Inliers = Inliers.to_numpy()
Outliers = Outliers.to_numpy()

Y = Inliers.T[:, :n_samples_train]
Y_normal_test = Inliers.T[:, n_samples_train:]
Y_anomaly_test = Outliers.T
Y_norm = np.linalg.norm(Y, ord='fro')
print(f'signals norm {Y_norm}')

X_fit = Y.T
X = np.c_[Y_normal_test, Y_anomaly_test].T
y_truth = np.r_[np.ones(n_samples_test), -1*np.ones(n_samples_anomaly)]

start_time = time.time()
# dictionary, code, rmse, error_extra = (
#     dictionary_learning(Y, copy.deepcopy(D0), n_nonzero_coefs,
#                         n_iterations, None, ksvd_supp, params))
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

now = int(time.time() - start_time)
used_atoms = np.unique(np.nonzero(code)[0])
used_set = set(used_atoms)
n_test_nonzero_coefs = len(used_atoms)
zerod_atoms_percent = n_test_nonzero_coefs/n_components

# TESTING: ANOMALY DETECTION
sensitivity, specificity = test_fpfn(X, y_truth, dictionary, used_set,
                                     n_nonzero_coefs, params)
balanced_accuracy = (sensitivity + specificity) / 2
print(f'[{now:06}] '
      f'supp {zerod_atoms_percent:.4f}, '
      f'TPR {sensitivity:.4f}, TNR {specificity:.4f}, '
      f'BA {balanced_accuracy:.4f}')
