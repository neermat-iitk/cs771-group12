import os
import pandas as pd
import numpy as np
import pickle as pkl
import argparse

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Evaluating Matrix Factorization Model.')
parser.add_argument('--idf', default='id0', help='identifier')
parser.add_argument('--mId', default=90, help='movie ID', type=int)

args = parser.parse_args()

# Load latent matrices
fname = 'cache/latentm_{}.pkl'.format(args.idf)
with open(fname, 'rb') as f:
    latentm = pkl.load(f)

P = latentm['P'].toarray()
Q = latentm['Q'].toarray()

# Pick a movie
movieId = args.mId
movie = pd.read_csv('cache/movieId_ftrd.csv')
print('Movie: {}'.format(movie.loc[movieId]['title']))

# Movie vector
movie_vec = Q[movieId].T
movie_vec = movie_vec/np.linalg.norm(movie_vec)

# Normalize Q
Q_norm = Q / np.linalg.norm(Q, axis=-1)[:, np.newaxis]

# Compute Similarities
scores = Q_norm.dot(movie_vec).reshape(1,-1)[0]
top_10 = np.argsort(scores)[::-1][:10]

sim_movies = []
movie_genres = []
movie_scores= []

# Get and print the actual artists names and scores
for idx in top_10:
    sim_movies.append(movie.loc[idx]['title'])
    movie_genres.append(movie.loc[idx]['genres'])
    movie_scores.append(scores[idx])

similar = pd.DataFrame({'movies': sim_movies, 'genres': movie_genres,'score': movie_scores} )
print(similar)
