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

# Load distance matrices
with open('cache/jaccard_dist.pkl', 'rb') as f:
    jaccard_dist = pkl.load(f)
with open('cache/cosine_dist.pkl', 'rb') as f:
    cosine_dist = pkl.load(f)

# Compute diversity metrics
K = 20
test_users = test_rating['userId'].unique()
jacEILD_at_k = np.zeros_like(test_users).astype(float)
cosEILD_at_k = np.zeros_like(test_users).astype(float)
ndisp = 5000
for idx, userId in enumerate(test_users):
    if idx%ndisp==0:
        print('| Evaluating {}-{}/{} users...'.format(
            idx, idx+ndisp, len(test_users)))

    # Get new index
    uId = get_new_userId(userId)

    # Compute user preference
    user_vec = P[uId, :]
    user_pref = user_vec.dot(Q.T)

    # Don't recommend movies already liked by the user
    mask = get_movie_liked_mask(userId)
    user_pref = ~mask*user_pref
    
    # Pick top_K
    if rand:
        # Random baseline
        sort_idx = np.random.choice(np.where(~mask)[0], size=K, replace=False)
    else:
        # Pick top_K list
        sort_idx = np.argsort(user_pref)[::-1][:K]
    jac_mat = jaccard_dist[sort_idx, :][:, sort_idx]
    cos_mat = cosine_dist[sort_idx, :][:, sort_idx]
    jacEILD_at_k[idx] = jac_mat.sum()/(K*(K-1))
    cosEILD_at_k[idx] = cos_mat.sum()/(K*(K-1))
    # cosEILD_at_k[idx] = .mean()
    # print((jaccard_dist[sort_idx, :][:, sort_idx]).shape)

print('| EILD@20 (jaccard): {}, {}, {}'.format(jacEILD_at_k.mean(), jacEILD_at_k.min(), jacEILD_at_k.max()))
print('| EILD@20 (cosine): {}, {}, {}'.format(cosEILD_at_k.mean(), cosEILD_at_k.min(), cosEILD_at_k.max()))

