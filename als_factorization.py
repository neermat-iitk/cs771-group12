import os
import pickle as pkl
import argparse
import pandas as pd
import numpy as np

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Matrix Factorization.')
parser.add_argument('--idf', default='id0', help='identifier')
args = parser.parse_args()

# Load training data
train_rating = pd.read_csv('movielens/train_rating_100k.csv')
X_train = train_rating.pivot(index='userId', columns='movieId', values='implicit')

# Factorize using ALS
def als(X, k=20, lam=0.1, max_iter=10):
    X = np.array(X)
    n, m = X.shape
    P = np.random.randn(n, k)
    Q = np.random.randn(m, k)
    I = np.eye(k)
    for itr in range(max_iter):
        print('Iteration {}/{}:'.format(itr, max_iter))
        print('Estimating user matrix P...')
        for u in range(n):
            sigma_ru = ~np.isnan(X[u])
            X_ = X[u:u+1, sigma_ru]
            Q_ = Q[sigma_ru,:]
            P[u] = np.dot(
                    np.linalg.inv(np.dot(Q_.T, Q_) 
                    + lam*I),
                    np.dot(Q_.T, X_.T)).squeeze()
        print('Estimating user matrix Q...')
        for v in range(m):
            sigma_cm = ~np.isnan(X[:,v])
            X_ = X[sigma_cm, v:v+1]
            P_ = P[sigma_cm,:]
            Q[v] = np.dot(np.linalg.inv(np.dot(P_.T, P_)
                    + lam*I),
                    np.dot(P_.T, X_)).squeeze()
    return P, Q

# Cache latent factors
fname = 'cache/latentm_{}.pkl'.format(args.idf)
if not os.path.exists(fname):
    latentm = {}
    latentm['P'], latentm['Q'] = als(X_train)
    if not os.path.exists('cache'):
        os.makedirs('cache')
    f = open(fname, 'wb')
    pkl.dump(latentm, f)
    f.close()
else:
    with open(fname, 'rb') as f:
        latentm = pkl.load(f)










