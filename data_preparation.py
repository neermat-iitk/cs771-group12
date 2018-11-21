## In this script, we explore the dataset Movielens
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('| Dataset: {}'.format('Movielens'))
print('')

# Movies/Items
movie = pd.read_csv('movielens/movie.csv')
print('| Total no. of movies: {}'.format(len(movie['movieId'].unique())))
# Movies with listed genre
ftr1 = (movie['genres']!='(no genres listed)') # Some genres are listed
ftr2 = (movie['genres'] != 'IMAX') # IMAX is not the only genre listed
movie = movie[ftr1 & ftr2]
print('| Total no. of movies with some genre listed: {}'.format(len(movie['movieId'].unique())))

# Genres
genres_list = []
for genres in movie['genres']:
    genres_list += genres.split('|')
genres_unq  = set(genres_list) - set(['IMAX'])
print('')
print('| Total no. of genres: {}'.format(len(genres_unq)))
print('| Genres: {}'.format(genres_unq))

# Ratings
downsample = True
sample_size = 100000
if downsample:
    rating = pd.read_csv('movielens/rating.csv').sample(n=sample_size)
else:
    rating = pd.read_csv('movielens/rating.csv')

# Implicitize the ratings
rating['implicit'] = (rating['rating'] == 5.0).astype(int)

# Filter
rating = pd.merge(rating[['movieId', 'userId', 'implicit']], movie[['movieId']], how='inner')
print('')
print('| Total no. of movies: {}'.format(len(rating['movieId'].unique())))
print('| Total no. of users: {}'.format(len(rating['userId'].unique())))
n_5ratings = (rating['implicit'].sum())
print('| Total no. of 5-ratings: {}/{} ({:.2f}%)'.format(
    n_5ratings, rating.shape[0], n_5ratings*100.0/rating.shape[0] ))

# Split train and test sets
sample_perm = np.random.permutation(rating.shape[0])
Ntrain = int(len(sample_perm)*0.8)
train_rating = rating.iloc[sample_perm[:Ntrain]]
test_rating = rating.iloc[sample_perm[Ntrain:]]
Npos = (train_rating['implicit'].sum())
print('| Total no. of 5-ratings in train set: {}/{} ({:.2f}%)'.format(
    Npos , train_rating.shape[0], Npos*100.0/train_rating.shape[0] ))
Npos = (test_rating['implicit'].sum())
print('| Total no. of 5-ratings in test set: {}/{} ({:.2f}%)'.format(
    Npos , test_rating.shape[0], Npos*100.0/test_rating.shape[0] ))

train_rating.to_csv('train_rating_100k.csv', index=None)
test_rating.to_csv('test_rating_100k.csv', index=None)

train_df = pd.read_csv('train_rating_100k.csv')
print(train_df)


