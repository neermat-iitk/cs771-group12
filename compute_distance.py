import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
import pickle as pkl

# Load list of movies
df = pd.read_csv('cache/movieId_ftrd.csv')

# List of genres is the vocab
vocab = ['Action', 'Adventure','Animation','Children',
        'Comedy','Crime','Documentary','Drama',
        'Fantasy','Film-Noir','Horror','Musical',
        'Mystery','Romance','Sci-Fi','Thriller',
        'War','Western']

# Vectorize
vectorizer = CountVectorizer(
        lowercase = False,
        vocabulary=vocab,
        token_pattern="(?u)\\b[\\w-]+\\b",  # prevents splitting of hyphenated words
        tokenizer=None,
        binary=True
        )
X = vectorizer.fit_transform(df['genres']).toarray() # nMovies x nGenres

# Compute intersection
intersection = X.dot(X.T)

# Compute union
union = np.zeros_like(intersection)
for a in range(len(df)):
    for b in range(len(df)):
        union[a, b] = ((X[a]+X[b]).clip(0, 1).sum())


# Compute cosine similarity
jaccard_sim_mat = intersection/union.astype(float)
jaccard_dist = 1-jaccard_sim_mat

# Normalize X
X_norm = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]

# Compute cosine similarity
cosine_sim_mat = X_norm.dot(X_norm.T)
cosine_dist = 1-cosine_sim_mat

# Display cosine and jaccard distance for a random pair
disp=True
if disp:
    a = np.random.randint(len(df))
    b = np.random.randint(len(df))
    print('')
    print('| Example movie pair:')
    print(df.iloc[[a,b]])
    print('')
    print('| Intersection: {}, Union: {}'.format(intersection[a, b], union[a, b]))
    print('| Jaccard distance: {}'.format(jaccard_dist[a, b]))
    print('| Cosine distance: {}'.format(cosine_dist[a, b]))


with open('cache/cosine_dist.pkl', 'wb') as f:
    pkl.dump(cosine_dist, f)
with open('cache/jaccard_dist.pkl', 'wb') as f:
    pkl.dump(jaccard_dist, f)


