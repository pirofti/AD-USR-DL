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

from dictlearn.methods._atom import _update_atom

def _ksvd_supp_update_atom(F, D, X, atom_index, atom_usages, params):
    U, S, Vh = np.linalg.svd(F, full_matrices=False)
    d = U[:, 0]
    x = S[0] * Vh[0, :]

    if params['supp_reg'] == 0:        # L0-regularization
        if (np.linalg.norm(F, 'fro')**2 <
            np.linalg.norm(F - np.outer(d, x), 'fro')**2 +
            params['supp_lambda']):

            # d = np.zeros_like(d)
            x = np.zeros_like(x)
    elif params['supp_reg'] == 1:      # L1-regularization
        if S[0] - params['supp_lambda'] <= 0:
            # d = np.zeros_like(d)
            x = np.zeros_like(x)
        else:
            x = (S[0] - params['supp_lambda']) * Vh[0, :]
    elif params['supp_reg'] == 2:      # clipped-L1 regularization
        if np.linalg.norm(x) >= params['supp_thres']:
            if S[0] - params['supp_lambda'] <= 0:
                # d = np.zeros_like(d)
                x = np.zeros_like(x)
            else:
                x = (S[0] - params['supp_lambda']) * Vh[0, :]

    return d, x


def ksvd_supp(Y, D, X, params):
    '''
    K-SVD algorithm with coherence reduction
    INPUTS:
        Y -- training signals set
        D -- current dictionary
        X -- sparse representations
    OUTPUTS:
        D -- updated dictionary
    '''
    D, X = _update_atom(Y, D, X, params, _ksvd_supp_update_atom)
    return D, X
