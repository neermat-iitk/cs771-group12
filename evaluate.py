import os
import pandas as pd
import numpy as np
import pickle as pkl
import argparse

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Evaluating Matrix Factorization Model.')
parser.add_argument('--idf', default='id0', help='identifier')

# Set rand=True to evaluate for the random baseline
rand=False

args = parser.parse_args()

# Load latent matrices
fname = 'cache/latentm_{}.pkl'.format(args.idf)
with open(fname, 'rb') as f:
    latentm = pkl.load(f)
# Convert sparse matrices to numpy arrays
P = latentm['P'].toarray()
Q = latentm['Q'].toarray()

# Load lists of movieIds and userIds
movie = pd.read_csv('cache/movieId_ftrd.csv')
user = pd.read_csv('cache/userId_ftrd.csv')
movieId_ar = movie['movieId'].values # movie Ids as array

def get_new_userId(userId):
    return np.where(user['userId'] == userId)[0][0]

def get_new_movieId(movieId):
    return np.where(movie['movieId'] == movieId)[0][0]

def get_movie_liked_mask(userId):
    movieId_liked = (train_rating[train_rating['userId']==userId])['movieId'].values
    movieId_liked_mask = np.any(movieId_liked == movieId_ar[:,None], axis=-1)
    return movieId_liked_mask


# Load test data
test_rating = pd.read_csv('cache/test_rating.csv')
test_rating = test_rating[['userId', 'movieId']]

# Load train data
train_rating = pd.read_csv('cache/train_rating.csv')
train_rating = train_rating[['userId', 'movieId']]

K = 20
ndcg_at_K = np.zeros((len(test_rating),))
discount_K = 1/np.log2(np.arange(1, K+1) + 1)
ndisp = 10000

for idx, row in test_rating.iterrows():
    if idx%ndisp==0:
        print('| Evaluating {}-{}/{} test interactions...'.format(
            idx, idx+ndisp, len(test_rating)))

    #TODO precompute the following mapping
    # new indices 
    uId = get_new_userId(row['userId'])
    mId = get_new_movieId(row['movieId'])

    # Compute user preference
    user_vec = P[uId, :]
    user_pref = user_vec.dot(Q.T)

    # Don't recommend movies already liked by the user
    mask = get_movie_liked_mask(row['userId'])
    user_pref = ~mask*user_pref

    if rand:
        # Random baseline
        sort_idx = np.random.choice(np.where(~mask)[0], size=K, replace=False)
    else:
        # Pick topN list
        sort_idx = np.argsort(user_pref)[::-1][:K]

    j = np.where(mId == sort_idx)[0]
    if j.size > 0:
        ndcg_at_K[idx] = discount_K[j]

print('| NDCG@20: {}'.format(ndcg_at_K.mean()))

