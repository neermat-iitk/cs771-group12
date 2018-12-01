import os
import pandas as pd
import numpy as np
import pickle as pkl
import argparse

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Evaluating Matrix Factorization Model.')
parser.add_argument('--idf', default='lam_400', help='identifier')
parser.add_argument('--uId', default=90, help='user ID', type=int)

args = parser.parse_args()

# Load latent matrices
fname = 'cache/latentm_{}.pkl'.format(args.idf)
with open(fname, 'rb') as f:
    latentm = pkl.load(f)

P = latentm['P'].toarray()
Q = latentm['Q'].toarray()

# Load lists of movieIds and userIds
movie = pd.read_csv('cache/movieId_ftrd.csv')
user = pd.read_csv('cache/userId_ftrd.csv')

# Load training data
train_rating = pd.read_csv('cache/train_rating.csv')
test_rating = pd.read_csv('cache/test_rating.csv')

# Pick a user
userId = user.loc[args.uId].values[0]
uId = np.where(user['userId'] == userId)[0][0]

# Movies liked by the user
movies_liked = (train_rating[train_rating['userId']==userId])['movieId'].values
all_movie_ids = movie['movieId'].values
movies_liked_idx = np.any(movies_liked == all_movie_ids[:,None], axis=-1)
print('-----------'*10)
print('|    Movies liked by the user in the train set:'.format())
print('-----------'*10)
print(movie.iloc[movies_liked_idx])
print(userId)
movie.iloc[movies_liked_idx].to_csv('user_train.csv')



# Movies liked by the user in the test
movies_liked_test = (test_rating[test_rating['userId']==userId])['movieId'].values
movies_liked_test_idx = np.any(movies_liked_test == all_movie_ids[:,None], axis=-1)
print('-----------'*10)
print('|    Movies liked by the user in the test set:'.format())
print('-----------'*10)
print(movie.iloc[movies_liked_test_idx])
movie.iloc[movies_liked_test_idx].to_csv('user_test.csv')

# Estimate the preference by the user
user_vec = P[uId, :]
# pref = user_vec.dot(Q.T)
pref = user_vec.dot(Q.T)
pref = pref*(1-movies_liked_idx)
print(pref.shape)
sort_idx = np.argsort(pref)[::-1][:20]
print('')
print('-----------'*10)
print('|   Movies recommended for the user:'.format())
print('-----------'*10)
top_recs = movie.iloc[sort_idx].copy()
top_recs['score'] = pref[sort_idx]
print(top_recs)
top_recs.to_csv('user_recs.csv')

