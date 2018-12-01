import numpy as np
import pandas as pd

# Load lists of movieIds and userIds
movie = pd.read_csv('cache/movieId_ftrd.csv')
movieId_ar = movie['movieId'].values # movie Ids as array
user = pd.read_csv('cache/userId_ftrd.csv')

# Load 5m ratings
rating_5m = pd.read_csv('cache/rating_5m.csv', index_col=0)
rating_20m = pd.read_csv('movielens/rating.csv')
# Take a 1m sample from all ratings
sample_size = 1000000
rating_20m = rating_20m.sample(n=sample_size, random_state=4321)
# remove 5m ratings used in train and val set
rating_15m = pd.concat([rating_20m, rating_5m, rating_5m]).drop_duplicates(keep=False)
test_rating = rating_15m
del rating_20m
del rating_5m
del rating_15m

# keep users only from train/val set
test_rating = test_rating[test_rating['movieId'].isin(movieId_ar)]
test_rating = test_rating[test_rating['userId'].isin(user['userId'])]
print('| Total test ratings: {}'.format(test_rating.shape[0]))
print('| Total test users: {}'.format(test_rating['userId'].unique().shape[0]))
print('| Total test movies: {}'.format(test_rating['movieId'].unique().shape[0]))
test_rating = test_rating[test_rating['rating'] == 5]
test_rating['rating'] = 1
print('| Total test ratings: {}'.format(test_rating.shape[0]))
print('| Total test users: {}'.format(test_rating['userId'].unique().shape[0]))
print('| Total test movies: {}'.format(test_rating['movieId'].unique().shape[0]))
test_rating.to_csv('cache/test_rating.csv', index=None)




