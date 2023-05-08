# K-MEANS

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Prepare the list of unique genres
unique_genres = list(set([genre for genres in data['genres'] for genre in genres])) + list(set([genre for genres in data['favourite_genres'] for genre in genres]))

# Convert the genres to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(unique_genres)

# Apply K-means clustering
num_clusters = 100  # Adjust this value according to your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vectorized)
labels = kmeans.labels_

# Group the genres based on the cluster labels
clustered_genres = {}
for i, label in enumerate(labels):
    if label not in clustered_genres:
        clustered_genres[label] = []
    clustered_genres[label].append(unique_genres[i])

# Map genres to their cluster label
genre_to_cluster = {genre: label for label, genres in clustered_genres.items() for genre in genres}

# Define the function for mapping genres to simpler forms
def map_genre(genre):
    cluster_label = genre_to_cluster[genre]
    representative_genre = clustered_genres[cluster_label][0]  # Use the first genre in the cluster as the representative
    return representative_genre

# Apply the mapping function to both 'genres' and 'favourite_genres' columns
data['genres'] = data['genres'].apply(lambda x: [map_genre(genre) for genre in x])
data['favourite_genres'] = data['favourite_genres'].apply(lambda x: [map_genre(genre) for genre in x])

# Delete not unique genres
data['genres'] = data['genres'].apply(lambda x: list(set(x)))
data['favourite_genres'] = data['favourite_genres'].apply(lambda x: list(set(x)))

data.head(10)