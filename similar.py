import os
import pandas as pd
import numpy as np
import pickle as pkl
import argparse

# Parse command line arguments 
parser = argparse.ArgumentParser(description='Evaluating Matrix Factorization Model.')
parser.add_argument('--idf', default='id0', help='identifier')
args = parser.parse_args()

# Load latent matrices
fname = 'cache/latentm_{}.pkl'.format(args.idf)
with open(fname, 'rb') as f:
    latentm = pkl.load(f)

P = latentm['P']
Q = latentm['Q']

# Pick a movie
movieId = 358 
movie = pd.read_csv('cache/movieId_ftrd.csv')
print('Movie: {}'.format(movie.loc[movieId]['title']))

movie_vec = Q[movieId].T

# Compute Similarities
scores = Q.dot(movie_vec).toarray().reshape(1,-1)[0]
top_10 = np.argsort(scores)[::-1][:10]

sim_movies = []
movie_scores= []

# Get and print the actual artists names and scores
for idx in top_10:
    sim_movies.append(movie.loc[idx]['title'])
    movie_scores.append(scores[idx])

similar = pd.DataFrame({'movies': sim_movies, 'score': movie_scores})
print(similar)
