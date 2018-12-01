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
sample_size = 5000000
if downsample:
    rating = pd.read_csv('movielens/rating.csv').sample(n=sample_size, random_state=1234)
    rating.to_csv('cache/rating_5m.csv')
else:
    rating = pd.read_csv('movielens/rating.csv')

# Data cleaning 
print('')
print('| Data Cleaning')
print('| Total interactions: {}'.format(rating.shape[0]))
rating = rating[rating['rating'] == 5.0]
rating['rating'] = 1
print('| Total interactions with 5 rating: {}'.format(rating.shape[0]))

# Filter movies: # users rated >= some thresh
movie_thresh = 100
nInts_by_movie = rating.groupby('movieId')['rating'].size()
movieId_ftrd = pd.DataFrame(nInts_by_movie[nInts_by_movie>= movie_thresh].index)
rating = pd.merge(rating[['userId', 'movieId', 'rating']], movieId_ftrd, how='inner')
print('| Total interactions after removing movies with low interactions: {}'.format(rating.shape[0]))

# Filter users: # movies rated >= some thresh
user_thresh = 10
nInts_by_user = rating.groupby('userId')['rating'].size()
userId_ftrd = pd.DataFrame(nInts_by_user[nInts_by_user>= user_thresh].index)
rating = pd.merge(rating, userId_ftrd)
print('| Total interactions after removing users with low interactions: {}'.format(rating.shape[0]))

# Filter
print('')
print('| Total no. of movies: {}'.format(len(rating['movieId'].unique())))
print('| Total no. of users: {}'.format(len(rating['userId'].unique())))
print('| Min. no. of users a movie has been rated by: {}'.format(
    rating.groupby('movieId')['rating'].size().min()))
print('| Min. no. of movies a user has rated : {}'.format(
    rating.groupby('userId')['rating'].size().min()))

# Split train and val sets
np.random.seed(321)
sample_perm = np.random.permutation(rating.shape[0])
Ntrain = int(len(sample_perm)*0.8)
train_rating = rating.iloc[sample_perm[:Ntrain]]
val_rating = rating.iloc[sample_perm[Ntrain:]]

# Check if train ratings has all the users and movies
print(rating['movieId'].unique().shape)
print(train_rating['movieId'].unique().shape)
print(val_rating['movieId'].unique().shape)
print(rating['userId'].unique().shape)
print(train_rating['userId'].unique().shape)
print(val_rating['userId'].unique().shape)

# Save train and val data 
if not os.path.exists('cache'):
    os.makedirs('cache')
userId_ftrd = pd.DataFrame({'userId': np.sort(rating['userId'].unique())})
movieId_ftrd = pd.DataFrame({'movieId': np.sort(rating['movieId'].unique())})
movieId_ftrd = movieId_ftrd.merge(movie, how='inner')
movieId_ftrd.to_csv('cache/movieId_ftrd.csv', index=None)
userId_ftrd.to_csv('cache/userId_ftrd.csv', index=None)
train_rating.to_csv('cache/train_rating.csv', index=None)
val_rating.to_csv('cache/val_rating.csv', index=None)
