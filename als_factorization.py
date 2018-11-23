# Based on Implicit ALS tutorial:
# https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

import os
import pickle as pkl
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Matrix Factorization.')
parser.add_argument('--idf', default='id0', help='identifier')
args = parser.parse_args()

# Load training data
print('| Loading training data...')
train_rating = pd.read_csv('cache/train_rating.csv')

# convert to sparse matrix
X = train_rating.pivot(index='userId', columns='movieId', values='rating')\
        .reset_index(drop=True)
X.fillna(0, inplace=True)
# X.columns = np.arange(len(X.columns))
X = sparse.csr_matrix(X.values)

# Factorize using ALS
def als(X, k=20, lam=0.1, alpha=40.0, num_iters=10):
    n, m = X.shape
    print('| users: {}, items: {}'.format(n, m))

    # Confidence matrix
    Conf = alpha*X[:, :]

    # Initialize latent matrices
    P = sparse.csr_matrix(np.random.randn(n, k))
    Q = sparse.csr_matrix(np.random.randn(m, k))

    # Compute identity matrices
    Ik = sparse.eye(k)
    In = sparse.eye(n)
    Im = sparse.eye(m)
    lam_Ik = lam*Ik

    for itr in range(num_iters):
        print('Iteration {}/{}:'.format(itr, num_iters))

        # Precompute PTP and QTQ
        PTP = P.T.dot(P)
        QTQ = Q.T.dot(Q)

        print('Estimating user matrix P...')
        for u in range(n):
            row = Conf[u, :].toarray()
            X_u = row.copy()
            X_u[X_u != 0 ] = 1.0

            # Compute Cu and Cu - I 
            CuI = sparse.diags(row, [0])
            Cu = CuI + Im
            
            # Final formula
            QT_CuI_Q = Q.T.dot(CuI).dot(Q) # kxm x mxm x mxk = kxk
            QT_Cu_Xu = Q.T.dot(Cu).dot(X_u.T) # kxm x mxm x mx1 = kx1
            P[u] = spsolve(QTQ + QT_CuI_Q + lam_Ik, QT_Cu_Xu) # kxk, kx1

        print('Estimating item matrix Q...')
        for v in range(m):
            col = Conf[:, v].T.toarray()
            X_v = col.copy()
            X_v[X_v != 0 ] = 1.0

            # Compute Cv and Cv - I 
            CvI = sparse.diags(col, [0])
            Cv = CvI + In
            
            # Final formula
            PT_CvI_P = P.T.dot(CvI).dot(P) # kxn x nxn x nxk = kxk
            PT_Cv_Xv = P.T.dot(Cv).dot(X_v.T) # kxn x nxn x nx1 = kx1
            Q[v] = spsolve(PTP + PT_CvI_P + lam_Ik, PT_Cv_Xv) # kxk, kx1
    return P, Q

# Cache latent factors
fname = 'cache/latentm_{}.pkl'.format(args.idf)
if not os.path.exists(fname):
    latentm = {}
    latentm['P'], latentm['Q'] = als(X)
    if not os.path.exists('cache'):
        os.makedirs('cache')
    f = open(fname, 'wb')
    pkl.dump(latentm, f)
    f.close()










